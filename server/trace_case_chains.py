#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_case_chains.py
====================
Lần chuỗi bản án từ cấp cao nhất xuống sơ thẩm trong dataset tmquan/cbba-toaan-gov-vn
(config `documents`, các file documents-*.parquet):

        Tái thẩm (Retrial)  →  [Giám đốc thẩm]  →  [Phúc thẩm]  →  Sơ thẩm   (dừng)

Cách làm
--------
1. Đọc metadata toàn bộ dataset (KHÔNG đọc markdown) → index theo (số, năm, mã bản án).
   Cache ra <out>/doc_index.parquet cho các lần chạy sau.
2. Văn bản gốc: instance_level = Retrial (mặc định; đổi bằng --levels, VD "Retrial,Cassation").
3. Trong markdown của văn bản gốc, trích mọi bản án/quyết định được nêu:
       "Bản án dân sự phúc thẩm số 16/2023/DS-PT ngày 09/3/2023 của Tòa án nhân dân tỉnh Bình Định"
   → số hiệu, giai đoạn (ST/PT/GĐT/TT), ngày, TÒA ÁN NÀO, và phần quyết định của văn bản gốc
   xử lý nó thế nào (hủy / giữ nguyên / sửa / đình chỉ). Bổ sung từ cột `citations_case` nếu sót.
   Tự loại số thụ lý, kháng nghị, nghị quyết, thông tư, hợp đồng, án lệ, số của chính văn bản.
4. Tìm từng bản án đó trong dataset. Số bản án KHÔNG duy nhất giữa các tòa (hàng trăm tòa huyện
   cùng có "12/2019/DS-ST"), nên ứng viên được chấm theo ngày ban hành + tên tòa:
       high   : khớp ngày + khớp tòa
       medium : khớp ngày (tòa không rõ) | lệch ≤7 ngày + khớp tòa | khớp tòa khi trích dẫn thiếu ngày
       low    : chỉ khớp (số, năm, mã), không có gì để kiểm chứng
   Lệch ngày > 7 hôm, hoặc khác tòa / khác cấp tòa → loại. Hỗ trợ tên tòa ẩn danh ("huyện DH").
5. Bản án tìm được nếu CHƯA phải sơ thẩm (GĐT, PT) thì đọc tiếp markdown của nó, lặp lại bước 3–4,
   cho đến khi gặp bản án sơ thẩm thì dừng (--max-depth chỉ là giới hạn an toàn).
6. Xuất:
       refs.parquet   mọi tham chiếu trích được ở mọi tầng + trạng thái đối chiếu + ứng viên
       edges.csv      cạnh (văn bản → bản án nó nêu) đã khớp
       chains.csv     mỗi dòng 1 đường đi: root → gdt_* → pt_* → st_* (+ outcome, link, confidence)
       chains.jsonl   dạng lồng: mỗi root 1 dòng, các đường đi kèm metadata từng bản án
       summary.json   thống kê

Ví dụ
-----
    pip install duckdb pyarrow rapidfuzz
    python trace_case_chains.py --data "/data/cbba/documents-*.parquet" --out out_tt
    python trace_case_chains.py --data "/data/cbba/documents-*.parquet" --out out_tt_test --limit 200
    python trace_case_chains.py --data "/data/cbba/documents-*.parquet" --out out_all --levels Retrial,Cassation
    python trace_case_chains.py --data "/data/cbba/documents-*.parquet" --out out_tt --explain <doc_name>
    # Kaggle / không tải về: DuckDB đọc thẳng từ HF
    python trace_case_chains.py --data "hf://datasets/tmquan/cbba-toaan-gov-vn/documents-*.parquet" --out out
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ============================== CẤU HÌNH – SỬA Ở ĐÂY ==============================
DATA_PATH = "/path/to/cbba-toaan-gov-vn"        # thư mục (lấy MỌI file .parquet bên trong) | 1 file .parquet | glob
LOAD_ALL_FIRST = True                           # nạp hết mọi file vào DuckDB trước rồi mới chạy
LOAD_TO = None                                  # None = nạp vào RAM | "/local/disk/cbba.duckdb" = nạp ra đĩa, lần sau dùng lại
OUT_DIR = "out_chains"                          # thư mục kết quả
LEVELS = ["Retrial"]                            # cấp xét xử của văn bản gốc
LIMIT = None                                    # None hoặc 0 = chạy toàn bộ
# ==================================================================================

import glob  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq

try:
    import duckdb
except ImportError:  # pragma: no cover
    sys.exit("Thiếu thư viện: pip install duckdb pyarrow rapidfuzz")

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover
    _rf_fuzz = None

log = logging.getLogger("trace_case_chains")

# --------------------------------------------------------------------------------------
# Hằng số
# --------------------------------------------------------------------------------------
# thứ bậc từ lớn → nhỏ: Tái thẩm > Giám đốc thẩm > Phúc thẩm > Sơ thẩm
LEVEL_RANK = {"First-instance": 1, "Appellate": 2, "Cassation": 3, "Retrial": 4}
LEVEL_SHORT = {"Retrial": "tt", "Cassation": "gdt", "Appellate": "pt", "First-instance": "st"}
STAGE_LEVEL = {"ST": "First-instance", "PT": "Appellate", "GĐT": "Cassation", "TT": "Retrial"}
STAGE_PHRASES = [("giám đốc thẩm", "GĐT"), ("tái thẩm", "TT"), ("phúc thẩm", "PT"), ("sơ thẩm", "ST")]
DOMAIN_PHRASES = [
    ("kinh doanh thương mại", "KDTM"), ("hôn nhân và gia đình", "HNGĐ"), ("hôn nhân gia đình", "HNGĐ"),
    ("hình sự", "HS"), ("dân sự", "DS"), ("hành chính", "HC"), ("lao động", "LĐ"),
    ("kinh tế", "KT"), ("phá sản", "PS"),
]

STAGE_ATOMS = {"ST", "PT", "GĐT", "TT"}
DOMAIN_ATOMS = {"HS", "DS", "HC", "KDTM", "KT", "LĐ", "HNGĐ", "PS", "VDS"}
BASE_ATOMS = STAGE_ATOMS | DOMAIN_ATOMS | {"QĐ"}
ATOM_ALIASES = {"HNGD": "HNGĐ", "LD": "LĐ", "QD": "QĐ", "GDT": "GĐT"}
# Ký hiệu KHÔNG phải bản án/quyết định của tòa (văn bản luật, thụ lý, kháng nghị, hợp đồng...)
CODE_BLACKLIST = {
    "NQ", "HĐTP", "HDTP", "QH", "CP", "NĐ", "ND", "UBND", "BTP", "TANDTC", "VKSNDTC", "VKS", "TTLT",
    "KN", "TL", "TLST", "TLPT", "TLDS", "TLVA", "TLHS", "TLHC", "TB", "CV", "HĐ", "HD", "BB", "VC",
    "AL", "QĐKN", "KNGĐT", "KNTT", "BCA", "BQP", "BTC", "BTNMT", "NHNN", "TTG", "QĐ-TTG", "CT", "UB",
}
NON_JUDGMENT_KINDS = {"thụ lý", "thông báo", "nghị quyết", "hợp đồng", "văn bản", "công văn", "án lệ",
                      "giấy", "biên bản", "kết luận", "cáo trạng", "thông tư", "nghị định", "luật"}

TIER_NAME = {3: "high", 2: "medium", 1: "low", 0: "rejected"}
TIER_OF = {v: k for k, v in TIER_NAME.items()}

# --------------------------------------------------------------------------------------
# Chuẩn hóa văn bản
# --------------------------------------------------------------------------------------
_OLD_TONE = {  # kiểu bỏ dấu cũ ở âm tiết mở cuối từ: "Toà" -> "Tòa", "Thuỷ" -> "Thủy"
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}
_OLD_TONE.update({k[0].upper() + k[1:]: v[0].upper() + v[1:] for k, v in list(_OLD_TONE.items())})
_OLD_TONE_RE = re.compile("(" + "|".join(map(re.escape, _OLD_TONE)) + r")(?![^\W\d_])")


def fix_tones(s: str) -> str:
    return _OLD_TONE_RE.sub(lambda m: _OLD_TONE[m.group(1)], s)


def normalize_text(s: str) -> str:
    """Văn bản phẳng (1 dòng) để regex: NFC, bỏ ký hiệu markdown, sửa dấu, gộp khoảng trắng."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("Ð", "Đ").replace("ð", "đ")  # U+00D0 hay bị dùng nhầm cho Đ trong PDF
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", s)  # markdown escape: DS\-ST -> DS-ST
    s = re.sub(r"[*_`#|>]+", " ", s)
    s = fix_tones(s)
    s = re.sub(r"(\d{4}\s?/\s?[A-ZĐ][A-ZĐ\-–—.]{0,24}[A-ZĐ])(?=(?:ngày|của|tại|về|đã|bị|và)\b)", r"\1 ", s)
    s = re.sub(r"Bà Rịa\s*[-–—]\s*Vũng Tàu", "Bà Rịa Vũng Tàu", s, flags=re.I)
    s = re.sub(r"\b(?:TP|Tp|tp)\s*\.\s*", "thành phố ", s)
    s = re.sub(r"\b(?:TX|Tx)\s*\.\s*", "thị xã ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_diacritics(s: str) -> str:
    s = s.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------------------
# Ký hiệu bản án ("DS-ST", "HSPT", "QĐST-HNGĐ" ...) → khóa chuẩn
# --------------------------------------------------------------------------------------
def clean_code(code: str) -> str:
    s = unicodedata.normalize("NFC", str(code)).replace("Ð", "Đ").replace("ð", "đ").upper()
    s = re.sub(r"[–—‐‑−]", "-", s)
    s = re.sub(r"[\s.&]", "", s)
    s = s.replace("HN-GĐ", "HNGĐ").replace("KD-TM", "KDTM")
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _decompose(chunk: str, atoms: tuple):
    if not chunk:
        return []
    for a in atoms:
        if chunk.startswith(a):
            rest = _decompose(chunk[len(a):], atoms)
            if rest is not None:
                return [ATOM_ALIASES.get(a, a)] + rest
    return None


class CodeNormalizer:
    """Tách ký hiệu thành tập 'atom' không phụ thuộc thứ tự / gạch nối:
    'DS-ST' == 'DSST' == 'ST-DS' → ('DS','ST');  'QĐST-HNGĐ' → ('HNGĐ','QĐ','ST')."""

    def __init__(self, extra_chunks=()):
        self.atoms = tuple(sorted(BASE_ATOMS | set(ATOM_ALIASES), key=len, reverse=True))
        self.extra = {c for c in extra_chunks if c not in CODE_BLACKLIST}
        self._cache: dict = {}

    def key(self, code, strict=True):
        if code is None:
            return None
        ck = (code, strict)
        if ck in self._cache:
            return self._cache[ck]
        res = self._key(code, strict)
        self._cache[ck] = res
        return res

    def _key(self, code, strict):
        c = clean_code(code)
        if not c:
            return None
        chunks = [x for x in c.split("-") if x]
        atoms = []
        for ch in chunks:
            if ch in CODE_BLACKLIST:
                return None if strict else ("RAW:" + c,)
            d = _decompose(ch, self.atoms)
            if d is None:
                if ch in self.extra or not strict:
                    d = [ch]
                else:
                    return None
            atoms.extend(d)
        if strict and not (set(atoms) & BASE_ATOMS) and not any(ch in self.extra for ch in chunks):
            return None
        return tuple(sorted(set(atoms)))


def stage_of(atoms) -> str | None:
    if not atoms:
        return None
    for s in ("GĐT", "TT", "PT", "ST"):
        if s in atoms:
            return s
    return None


def domain_of(atoms) -> str | None:
    if not atoms:
        return None
    for d in ("KDTM", "HNGĐ", "HS", "DS", "HC", "LĐ", "KT", "PS", "VDS"):
        if d in atoms:
            return d
    return None


# --------------------------------------------------------------------------------------
# Trích tham chiếu từ văn bản
# --------------------------------------------------------------------------------------
_CH = r"[A-Za-zĐđ]{1,12}(?![^\W\d_])"
CASE_RE = re.compile(
    r"(?<![\d/.,])(?P<num>\d{1,5})\s?/\s?(?P<year>(?:19[89]\d|20[0-4]\d))\s?/\s?"
    r"(?P<code>" + _CH + r"(?:(?:\s?[-–—]\s?|\.)" + _CH + r"){0,3}(?:\s(?:ST|PT|GĐT|TT)(?![^\W\d_]))?)"
)
# số kiểu cũ không có năm: "Bản án hình sự sơ thẩm số 123/HSST ngày 12/5/2003" → năm lấy từ ngày
CASE_NOYEAR_RE = re.compile(
    r"\b(?:bản án|quyết định)\b[^.;:\d]{0,80}?\bsố\s*:?\s*(?P<num>\d{1,5})\s?/\s?"
    r"(?P<code>" + _CH + r"(?:(?:\s?[-–—]\s?|\.)" + _CH + r"){0,3})(?=[\s,;.)])", re.I)
KIND_RE = re.compile(
    r"\b(bản án|quyết định|án lệ|thụ lý|thông báo|nghị quyết|hợp đồng|văn bản|công văn|giấy|biên bản|"
    r"kết luận|cáo trạng|thông tư|nghị định|(?<!pháp )luật)\b", re.I)
DATE_RE = re.compile(
    r"^(?:[.\-–—/]?[A-ZĐ]{1,5}\b)?[\s,;:(\-–]*(?:được\s+|đã\s+)?(?:tuyên\s+án\s+|tuyên\s+|ban hành\s+|xét xử\s+)?(?:vào\s+)?"
    r"ngày\s*:?\s*(?P<d>\d{1,2})\s*(?:[/\-.]\s*(?P<m>\d{1,2})\s*[/\-.]\s*(?P<y>(?:19|20)\d{2})"
    r"|tháng\s*(?P<m2>\d{1,2})\s*năm\s*(?P<y2>(?:19|20)\d{2}))", re.I)
COURT_START_RE = re.compile(
    r"^[\s,;:(\-–]*(?:của\s+|do\s+)?(?P<court>(?:Ủy ban Thẩm phán\s+|Hội đồng Thẩm phán\s+|"
    r"Hội đồng (?:xét xử\s+)?(?:giám đốc thẩm|tái thẩm|phúc thẩm|sơ thẩm)\s+(?:của\s+)?|Tòa phúc thẩm\s+)?"
    r"(?:Tòa án|Tòa|TANDTC|TANDCC|TAND)\b.{0,180})", re.I)
COURT_STOP_RE = re.compile(
    r"\s(?:quyết định|đã|xử|xét xử|tuyên|về|đối với|và|nêu trên|thì|theo|giải quyết|với|bị|có|là|trong|"
    r"nhưng|cũng|không|giữa|ra|kháng cáo|kháng nghị|hủy|giữ nguyên|sửa|đề nghị|cho|để|nay|được|mà|phần|"
    r"ngày|thụ lý|số|thuộc|này|đó|sau|trước|khi|lần|xác định|nhận định|chấp nhận|buộc|áp dụng|"
    r"tuyên bố|ở|như sau|đều|vẫn|chỉ|bản án|quyết định)\b"
    r"|(?<!cấp cao)(?<!tối cao)\stại\b|[;:()\[\]\"“”'‘’]|\.\s|\.$|\s[-–—]\s", re.I)
COURT_PREFIX_RE = re.compile(r"^(?:Ủy ban Thẩm phán|Hội đồng\b.*?)\s+(?:của\s+)?(?=Tòa|TAND)", re.I)
ADMIN_LEAD_RE = re.compile(r"^(?:tỉnh|thành phố|tp|quận|huyện|thị xã)\b", re.I)
GENERIC_COURT_RE = re.compile(
    r"^(?:tòa án|tòa)(?:\s+nhân dân)?(?:\s+cấp\s+(?:sơ thẩm|phúc thẩm|giám đốc thẩm|tái thẩm|huyện|tỉnh|"
    r"dưới|trên))?(?:\s+(?:nói trên|trên|đó))?$", re.I)

OUTCOME_PATTERNS = [("huy", re.compile(r"\bhủy\b", re.I)),
                    ("giu_nguyen", re.compile(r"\bgiữ nguyên\b", re.I)),
                    ("sua", re.compile(r"\bsửa\b", re.I)),
                    ("dinh_chi", re.compile(r"\bđình chỉ\b", re.I))]
DECISION_HEAD_RE = re.compile(
    r"QUYẾT ĐỊNH\s*:|\b(?:TUYÊN XỬ|XỬ)\s*:|QUYẾT ĐỊNH(?=\s+(?:\d+\s*[.)/]|[-–]\s|Chấp nhận|Không chấp nhận|Hủy|Giữ nguyên|Sửa|"
    r"Đình chỉ|Căn cứ))")


def _to_iso(d, m, y):
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except (TypeError, ValueError):
        return None


def _parse_date_at(s: str):
    m = DATE_RE.match(s)
    if not m:
        return None, 0
    iso = _to_iso(m["d"], m["m"] or m["m2"], m["y"] or m["y2"])
    return iso, m.end()


def _clean_court(raw: str) -> str | None:
    s = COURT_PREFIX_RE.sub("", raw, count=1)
    m = COURT_STOP_RE.search(s)
    if m:
        s = s[: m.start()]
    parts = [p.strip() for p in s.split(",")]
    keep = [parts[0]]
    for p in parts[1:]:
        if ADMIN_LEAD_RE.match(p):
            keep.append(p)
        else:
            break
    s = ", ".join(x for x in keep if x).strip(" ,.-")
    if not s or GENERIC_COURT_RE.match(s) or len(s) < 6:
        return None
    return s[:150]


def decision_section_start(text: str) -> int:
    last = None
    for m in DECISION_HEAD_RE.finditer(text):
        last = m
    if last:
        return last.start()
    nd = text.rfind("NHẬN ĐỊNH")
    if nd >= 0:
        q = text.find("QUYẾT ĐỊNH", nd)
        if q >= 0:
            return q
    return len(text) + 1


def _outcome_before(text: str, pos: int, lo: int) -> str | None:
    seg = text[max(lo, pos - 400): pos]
    cut = max(seg.rfind(". "), seg.rfind("; "), seg.rfind(": "))
    if cut >= 0:
        seg = seg[cut + 2:]
    best, best_pos = None, -1
    for name, rx in OUTCOME_PATTERNS:
        for mm in rx.finditer(seg):
            if mm.start() > best_pos:
                best, best_pos = name, mm.start()
    return best


def _forward_info(text: str, end: int):
    """Sau số hiệu: (ngày ISO, tòa, số ký tự dùng cho phần ngày)."""
    fwd = text[end: end + 260]
    iso, used = _parse_date_at(fwd)
    rest = fwd[used:]
    court = None
    cm = COURT_START_RE.match(rest)
    if cm:
        court = _clean_court(cm["court"])
        if court and not iso:
            after = rest[cm.start("court") + len(court):]
            iso2, _ = _parse_date_at(after)
            iso = iso or iso2
    return iso, court, (used if iso else 0)


def _kind_before(text: str, start: int):
    """Nhìn ngược ≤140 ký tự: (kind, phrase, raw_start, bị_loại?)."""
    back = text[max(0, start - 140): start]
    kinds = list(KIND_RE.finditer(back))
    if not kinds:
        return None, "", start, False
    k = kinds[-1]
    kind = k.group(1).lower()
    phrase = back[k.start():]
    reject = (kind in NON_JUDGMENT_KINDS and len(back) - k.start() < 60) or \
             ("kháng nghị" in phrase.lower() and kind == "quyết định")
    return kind, phrase, max(0, start - 140) + k.start(), reject


def _stage_domain(atoms, phrase):
    stage, domain = stage_of(atoms), domain_of(atoms)
    pl = phrase.lower()
    if not stage and pl:
        stage = next((st for p, st in STAGE_PHRASES if p in pl), None)
    if not domain and pl:
        domain = next((d for p, d in DOMAIN_PHRASES if p in pl), None)
    return stage, domain


def _valid_atoms(code_raw: str, codenorm: CodeNormalizer):
    """Thử cả mã đầy đủ lẫn các tiền tố (bỏ mảnh rác cuối: 'DS-ST-TAND' → 'DS-ST')."""
    atoms = codenorm.key(code_raw, strict=True)
    if atoms:
        return atoms, clean_code(code_raw)
    chunks = [c for c in re.split(r"\s?[-–—.]\s?|\s", code_raw) if c]
    for k in range(len(chunks) - 1, 0, -1):
        c = "-".join(chunks[:k])
        atoms = codenorm.key(c, strict=True)
        if atoms:
            return atoms, clean_code(c)
    return None, None


def _make_mention(text, start, end, num, year, code, atoms, decision_start, source, **extra):
    kind, phrase, raw_start, _ = _kind_before(text, start)
    stage, domain = _stage_domain(atoms, phrase)
    iso, court, used = _forward_info(text, end)
    raw = text[raw_start: end + used] + (" … " + court if court else "")
    in_dec = start >= decision_start
    outcome = _outcome_before(text, raw_start, decision_start) if in_dec else None
    m = dict(num=num, year=year, code=code, atoms=atoms, stage=stage, domain=domain, kind=kind, date=iso,
             court=court, pos=start, in_decision=in_dec, outcome=outcome, outcome_via="decision" if outcome else None,
             raw=raw[:400], source=source, cc_confirmed=False, cc_role=None, year_inferred=False)
    m.update(extra)
    return m


def extract_mentions(text: str, codenorm: CodeNormalizer, decision_start: int | None = None):
    """Mọi lần xuất hiện số hiệu bản án/quyết định hợp lệ trong TOÀN văn bản (kể cả phần QUYẾT ĐỊNH)."""
    if decision_start is None:
        decision_start = decision_section_start(text)
    out, seen_pos = [], set()
    for m in CASE_RE.finditer(text):
        atoms, code = _valid_atoms(m["code"], codenorm)
        if atoms is None:
            continue
        if _kind_before(text, m.start())[3]:
            continue
        seen_pos.add(m.start())
        out.append(_make_mention(text, m.start(), m.end(), int(m["num"]), int(m["year"]), code, atoms,
                                 decision_start, "markdown"))
    # số kiểu cũ không có năm → lấy năm từ ngày ban hành ngay sau
    for m in CASE_NOYEAR_RE.finditer(text):
        st = m.start("num")
        if st in seen_pos:
            continue
        atoms, code = _valid_atoms(m["code"], codenorm)
        if atoms is None or not stage_of(atoms):
            continue
        iso, _, _ = _forward_info(text, m.end())
        if not iso:
            continue
        out.append(_make_mention(text, st, m.end(), int(m["num"]), int(iso[:4]), code, atoms,
                                 decision_start, "markdown", year_inferred=True))
    out.sort(key=lambda x: x["pos"])
    return out


# "Hủy Bản án dân sự phúc thẩm nêu trên" (không ghi số) → gán kết quả cho bản án PT gần nhất phía trước
ANAPHORA_RE = re.compile(
    r"\b(?P<verb>hủy|giữ nguyên|sửa|đình chỉ)\b(?:\s+(?:toàn bộ|một phần))?[^.;:]{0,30}?\b(?P<kind>bản án|quyết định)\b"
    r"(?P<mid>(?:\s+[^\s.;:]+){0,7}?)\s+(?:nêu trên|nói trên|đã nêu|trên|này|đó)\b", re.I)
_VERB_OUTCOME = {"hủy": "huy", "giữ nguyên": "giu_nguyen", "sửa": "sua", "đình chỉ": "dinh_chi"}


def resolve_anaphora(text: str, mentions: list, decision_start: int):
    if decision_start >= len(text):
        return
    for m in ANAPHORA_RE.finditer(text, decision_start):
        mid = (m["mid"] or "").lower()
        if re.search(r"\d", mid):
            continue  # có số hiệu → đã được bắt trực tiếp
        stage = next((st for p, st in STAGE_PHRASES if p in mid), None)
        domain = next((d for p, d in DOMAIN_PHRASES if p in mid), None)
        kind = m["kind"].lower()
        cands = [x for x in mentions if 0 <= x["pos"] < m.start() and x.get("kind") in (None, kind)
                 and (stage is None or x["stage"] == stage) and (domain is None or x["domain"] in (None, domain))]
        if not cands:
            continue
        # "nêu trên" thường chỉ bản án đang bị xem xét: được nhắc nhiều lần nhất; hòa → bản án mới nhất
        cnt = Counter((x["num"], x["year"]) for x in cands)
        tgt = max(cands, key=lambda x: (cnt[(x["num"], x["year"])], x["year"], x["pos"]))
        if tgt["outcome"] is None:
            tgt["outcome"] = _VERB_OUTCOME[m["verb"].lower()]
            tgt["outcome_via"] = "anaphora" if len(cnt) == 1 else "anaphora_ambiguous"
            tgt["in_decision"] = True


def _span_start(span):
    """span trong citations_case có thể là [s,e], [[s,e],...], {'start':..}, hoặc None."""
    try:
        if span is None:
            return None
        if isinstance(span, (list, tuple)) and span:
            first = span[0]
            if isinstance(first, (list, tuple)):
                return int(first[0])
            return int(first)
        if isinstance(span, dict):
            return int(span.get("start"))
    except (TypeError, ValueError):
        return None
    return None


_CAT2ATOM = {"civil": "DS", "criminal": "HS", "administrative": "HC", "commercial": "KDTM", "labor": "LĐ",
             "marriage & family": "HNGĐ", "bankruptcy": "PS", "economic": "KT", "dân sự": "DS", "hình sự": "HS",
             "hành chính": "HC", "kinh doanh thương mại": "KDTM", "lao động": "LĐ", "hôn nhân và gia đình": "HNGĐ"}
_LVL2ATOM = {"first-instance": "ST", "appellate": "PT", "cassation": "GĐT", "retrial": "TT", "st": "ST",
             "pt": "PT", "gđt": "GĐT", "gdt": "GĐT", "tt": "TT", "sơ thẩm": "ST", "phúc thẩm": "PT",
             "giám đốc thẩm": "GĐT", "tái thẩm": "TT"}


def parse_cc_item(it, codenorm: CodeNormalizer):
    """1 phần tử citations_case → dict(num, year, code, atoms, start, role) hoặc None.
    Chịu được: number là chuỗi, id dạng '16/2023/DS-PT', code thiếu phần cấp/lĩnh vực (lấy từ level/domain)."""
    if isinstance(it, str):
        it = {"id": it}
    if not isinstance(it, dict):
        return None

    def g(*ks):
        return next((it.get(k) for k in ks if it.get(k) not in (None, "", [])), None)

    num = year = code = None
    for cand in (g("id"), g("ref"), g("raw"), g("text"), g("number"), g("code")):
        if isinstance(cand, str) and "/" in cand:
            mm = CASE_RE.search(normalize_text(cand))
            if mm:
                num, year, code = int(mm["num"]), int(mm["year"]), mm["code"]
                break
    if num is None:
        try:
            num = int(float(str(g("number")).strip()))
            year = int(float(str(g("year")).strip()))
        except (TypeError, ValueError):
            return None
        code = g("code")
    if code and any(ch in CODE_BLACKLIST for ch in clean_code(code).split("-")):
        return None
    atoms = {a for a in (codenorm.key(code, strict=False) or ()) if not a.startswith("RAW:")} if code else set()
    lv = _LVL2ATOM.get(str(g("level") or "").strip().lower())
    dm = str(g("domain") or "").strip()
    dm_atom = _CAT2ATOM.get(dm.lower()) or (clean_code(dm) if dm and clean_code(dm) in DOMAIN_ATOMS else None)
    if lv and not (atoms & STAGE_ATOMS):
        atoms.add(lv)
    if dm_atom and not (atoms & DOMAIN_ATOMS):
        atoms.add(dm_atom)
    if not atoms:
        return None
    atoms = tuple(sorted(atoms))
    return dict(num=num, year=year, code=clean_code(code) if code else "-".join(atoms), atoms=atoms,
                start=_span_start(g("span", "spans")), role=g("role"))


def atoms_compatible(a, b) -> bool:
    sa, sb, da, db = stage_of(a), stage_of(b), domain_of(a), domain_of(b)
    return (not sa or not sb or sa == sb) and (not da or not db or da == db)


def _locate_in_text(c, flat, codenorm, decision_start):
    """Regex bỏ sót nhưng citations_case có → tìm số hiệu trong văn bản phẳng để lấy ngày/tòa."""
    pat = re.compile(rf"(?<![\d/.,])0*{c['num']}\s?/\s?{c['year']}\s?/\s?")
    for mm in pat.finditer(flat):
        tail = re.compile(r"[A-Za-zĐđ][A-Za-zĐđ\-–—.]{0,24}").match(flat, mm.end())
        end = tail.end() if tail else mm.end()
        if _kind_before(flat, mm.start())[3]:
            continue
        return _make_mention(flat, mm.start(), end, c["num"], c["year"], c["code"], c["atoms"], decision_start,
                             "citations_case", cc_confirmed=True, cc_role=c["role"])
    return None


def mentions_from_citations_case(items, raw_markdown: str, flat: str, codenorm: CodeNormalizer, mentions: list,
                                 decision_start: int):
    """Đối chiếu cột citations_case với mention từ markdown:
    - trùng (cùng số/năm, cấp & lĩnh vực không mâu thuẫn) → đánh dấu cc_confirmed, bổ sung mã đầy đủ hơn
    - markdown bỏ sót → tìm lại trong văn bản để lấy ngày/tòa; không thấy thì giữ bản ghi trơn."""
    out = []
    by_ny = defaultdict(list)
    for m in mentions:
        by_ny[(m["num"], m["year"])].append(m)
    for it in items or []:
        c = parse_cc_item(it, codenorm)
        if not c:
            continue
        if str(c["role"] or "").strip().lower() in ("self", "own", "current", "this"):
            continue
        same = [m for m in by_ny[(c["num"], c["year"])] if atoms_compatible(m["atoms"], c["atoms"])]
        if same:
            for m in same:
                m["cc_confirmed"] = True
                m["cc_role"] = m["cc_role"] or c["role"]
                richer = tuple(sorted(set(m["atoms"]) | set(c["atoms"])))
                if richer != m["atoms"] and atoms_compatible(richer, m["atoms"]):
                    m["atoms"] = richer
                    if len(c["code"]) > len(m["code"]):
                        m["code"] = c["code"]
                    m["stage"] = m["stage"] or stage_of(richer)
                    m["domain"] = m["domain"] or domain_of(richer)
            continue
        found = None
        if c["start"] is not None and raw_markdown:
            snip = normalize_text(raw_markdown[max(0, c["start"] - 200): c["start"] + 500])
            for mm in extract_mentions(snip, codenorm, decision_start=10 ** 9):
                if (mm["num"], mm["year"]) == (c["num"], c["year"]) and atoms_compatible(mm["atoms"], c["atoms"]):
                    found = mm
                    break
            if found is not None:  # vị trí thật trong văn bản phẳng để biết có nằm ở phần QUYẾT ĐỊNH không
                loc = _locate_in_text(c, flat, codenorm, decision_start)
                found.update(source="citations_case", cc_confirmed=True, cc_role=c["role"],
                             pos=loc["pos"] if loc else -1, in_decision=bool(loc and loc["in_decision"]),
                             outcome=loc["outcome"] if loc else None)
        if found is None:
            found = _locate_in_text(c, flat, codenorm, decision_start)
        if found is None:
            found = dict(num=c["num"], year=c["year"], code=c["code"], atoms=c["atoms"], stage=stage_of(c["atoms"]),
                         domain=domain_of(c["atoms"]), kind=None, date=None, court=None, pos=-1, in_decision=False,
                         outcome=None, outcome_via=None, raw=str(it.get("id") if isinstance(it, dict) else it)[:200],
                         source="citations_case", cc_confirmed=True, cc_role=c["role"], year_inferred=False)
        out.append(found)
        by_ny[(c["num"], c["year"])].append(found)
    return out


# kết quả chung của phần QUYẾT ĐỊNH (dùng khi không gán được cho từng bản án)
DEC_ANNUL_RE = re.compile(r"\bhủy\b(?:\s+(?:toàn bộ|một phần))?[^.;:]{0,40}?\b(?:bản án|quyết định)\b[^.;:]{0,80}?"
                          r"\b(?:sơ thẩm|phúc thẩm|giám đốc thẩm|tái thẩm)\b", re.I)
DEC_TERMINATE_APPEAL_RE = re.compile(r"\bđình chỉ\b[^.;:]{0,20}?\bxét xử (?:phúc thẩm|giám đốc thẩm|tái thẩm)\b", re.I)
DEC_UPHOLD_RE = re.compile(r"\bgiữ nguyên\b[^.;:]{0,40}?\b(?:bản án|quyết định)\b", re.I)
DEC_AMEND_RE = re.compile(r"\bsửa\b[^.;:]{0,40}?\b(?:bản án|quyết định)\b", re.I)
_NOT_DECISION_CTX = re.compile(r"đề nghị|yêu cầu|theo hướng", re.I)


def decision_flags(flat: str) -> dict:
    """Phần QUYẾT ĐỊNH có hủy / giữ nguyên / sửa / đình chỉ bản án cấp dưới không (bỏ qua '... đề nghị hủy ...')."""
    ds = decision_section_start(flat)
    dec = flat[ds:] if ds < len(flat) else ""

    def hit(rx):
        for m in rx.finditer(dec):
            seg = dec[max(0, m.start() - 80): m.start()]
            cut = max(seg.rfind(". "), seg.rfind("; "), seg.rfind(": "))
            if cut >= 0:
                seg = seg[cut + 2:]
            if not _NOT_DECISION_CTX.search(seg):
                return True
        return False

    return dict(annuls=hit(DEC_ANNUL_RE), upholds=hit(DEC_UPHOLD_RE), amends=hit(DEC_AMEND_RE),
                terminates=bool(DEC_TERMINATE_APPEAL_RE.search(dec)), found_decision_section=bool(dec))


def collect_mentions(raw_markdown: str, citations_case, codenorm: CodeNormalizer, use_cc: bool = True):
    """Gom tham chiếu từ markdown (toàn văn + phần QUYẾT ĐỊNH, kể cả 'bản án ... nêu trên') và citations_case."""
    flat = normalize_text(raw_markdown or "")
    dstart = decision_section_start(flat)
    ms = extract_mentions(flat, codenorm, dstart)
    if use_cc and citations_case is not None and len(citations_case):
        ms += mentions_from_citations_case(list(citations_case), raw_markdown or "", flat, codenorm, ms, dstart)
    resolve_anaphora(flat, ms, dstart)
    return flat, ms, dstart


def group_mentions(mentions, self_keys: set):
    """Gộp các lần nhắc cùng 1 bản án (cùng số/năm/mã; tách nếu khác ngày)."""
    groups = defaultdict(list)
    for m in mentions:
        k = (m["num"], m["year"], m["atoms"])
        if k in self_keys or any(k[0] == s[0] and k[1] == s[1] and atoms_compatible(k[2], s[2]) and
                                 stage_of(k[2]) == stage_of(s[2]) for s in self_keys):
            continue
        groups[k].append(m)
    refs = []
    for k, ms in groups.items():
        dates = Counter(m["date"] for m in ms if m["date"])
        if len(dates) <= 1:
            buckets = [ms]
        else:
            main = dates.most_common(1)[0][0]
            by = defaultdict(list)
            for m in ms:
                by[m["date"] or main].append(m)
            buckets = list(by.values())
        for b in buckets:
            courts = Counter(m["court"] for m in b if m["court"])
            court = max(courts, key=lambda c: (courts[c], c.count(","), len(c))) if courts else None
            dts = [m["date"] for m in b if m["date"]]
            stage = next((m["stage"] for m in b if m["stage"]), None)
            domain = next((m["domain"] for m in b if m["domain"]), None)
            dec = [m for m in b if m["outcome"]]
            outcome = dec[0]["outcome"] if dec else None
            outcome_via = dec[0].get("outcome_via") if dec else None
            best_raw = max(b, key=lambda m: (m["source"] == "markdown", bool(m["date"]), bool(m["court"]),
                                             -m["pos"]))["raw"]
            sources = set()
            for m in b:
                sources.add(m["source"])
                if m.get("cc_confirmed"):
                    sources.add("citations_case")
            positions = [m["pos"] for m in b if m["pos"] >= 0]
            refs.append(dict(
                number=k[0], year=k[1], code=b[0]["code"], code_key="-".join(k[2]), atoms=k[2],
                stage=stage, cited_level=STAGE_LEVEL.get(stage), domain=domain,
                cited_date=dts[0] if dts else None, cited_court=court, outcome=outcome, outcome_via=outcome_via,
                in_decision=any(m["in_decision"] for m in b), n_mentions=len(b),
                sources="+".join(sorted(sources)), from_citations_case="citations_case" in sources,
                cc_role=next((m.get("cc_role") for m in b if m.get("cc_role")), None),
                year_inferred=all(m.get("year_inferred") for m in b),
                first_pos=min(positions) if positions else 10 ** 9, raw=best_raw,
            ))
    refs.sort(key=lambda r: r["first_pos"])
    return refs


# --------------------------------------------------------------------------------------
# So khớp tên tòa
# --------------------------------------------------------------------------------------
_UNIT_RE = re.compile(
    r"(cấp cao tại|tối cao tại|thành phố|thị xã|huyện|quận|tỉnh|quân khu|khu vực)\s+(.+?)"
    r"(?=\s*,|\s+(?:cấp cao tại|thành phố|thị xã|huyện|quận|tỉnh)\s|$)")
_NAME_ALIAS = {"hcm": "hochiminh", "tphcm": "hochiminh", "hn": "hanoi", "brvt": "bariavungtau",
               "bariavt": "bariavungtau"}
_court_cache: dict = {}


def court_units(court: str):
    if court in _court_cache:
        return _court_cache[court]
    s = fix_tones(unicodedata.normalize("NFC", court))
    s = re.sub(r"Bà Rịa\s*[-–—]\s*Vũng Tàu", "Bà Rịa Vũng Tàu", s, flags=re.I)
    s = re.sub(r"\s+[-–—]\s+", ", ", s)
    s = re.sub(r"\b(?:TP|Tp|tp)\b\.?\s*", "thành phố ", s)
    s = re.sub(r"\b(?:TX|tx)\b\.?\s*", "thị xã ", s)
    s = re.sub(r"\bTANDTC\b", "Tòa án nhân dân tối cao", s, flags=re.I)
    s = re.sub(r"\bTANDCC\b", "Tòa án nhân dân cấp cao", s, flags=re.I)
    # bỏ tiền tố nhưng GIỮ chữ hoa/thường để nhận diện tên viết tắt ẩn danh ("huyện DH", "thị xã A")
    s2 = re.sub(r"(ủy ban thẩm phán|hội đồng thẩm phán|tòa phúc thẩm|tòa án nhân dân|tòa án quân sự|"
                r"\btand\b|tòa án|tòa)", " ", s, flags=re.I)
    s2 = re.sub(r"\s+", " ", s2).strip(" ,")
    low = s2.lower()
    units = []
    for m in _UNIT_RE.finditer(low):
        typ = m.group(1)
        orig = s2[m.start(2): m.end(2)].strip(" ,.")
        units.append((typ, orig.lower(), orig))
    if not units and "tối cao" in low:
        units.append(("tối cao", "", ""))
    _court_cache[court] = units
    return units


from functools import lru_cache


@lru_cache(maxsize=200_000)
def _nkey(name: str) -> str:
    k = re.sub(r"[^a-z0-9]", "", strip_diacritics(name).lower())
    k = re.sub(r"^(?:thanhpho|tinh)(?=[a-z])", "", k)
    return _NAME_ALIAS.get(k, k)


@lru_cache(maxsize=200_000)
def _initials(name: str) -> str:
    return "".join(w[0] for w in strip_diacritics(name).lower().split() if w)


def _is_anon(orig: str) -> bool:
    o = orig.replace(" ", "")
    return 1 <= len(o) <= 3 and o.isalpha() and o.upper() == o


_TYPE_CLASS = {"tỉnh": {"prov"}, "thành phố": {"prov", "dist"}, "huyện": {"dist"}, "quận": {"dist"},
               "thị xã": {"dist"}, "cấp cao tại": {"cc"}, "tối cao tại": {"tc"}, "tối cao": {"tc"},
               "quân khu": {"qk"}, "khu vực": {"kv"}}


def _type_compat(a, b) -> bool:
    return bool(_TYPE_CLASS.get(a, {a}) & _TYPE_CLASS.get(b, {b}))


def _same_unit(u, v) -> bool:
    (tu, nu, ou), (tv, nv, ov) = u, v
    if not _type_compat(tu, tv):
        return False  # "TAND thành phố Hà Nội" ≠ "TAND cấp cao tại Hà Nội"
    if not nu or not nv:
        return not nu and not nv
    if _nkey(nu) == _nkey(nv):
        return True
    if _is_anon(ou) and not _is_anon(ov):  # bản án ẩn danh: "huyện DH" ~ "huyện Đức Hòa"
        return _nkey(ou) == _initials(nv)[: len(_nkey(ou))]
    if _is_anon(ov) and not _is_anon(ou):
        return _nkey(ov) == _initials(nu)[: len(_nkey(ov))]
    return False


@lru_cache(maxsize=500_000)
def court_sim(ref_court: str | None, cand_court: str | None):
    """1.0 cùng tòa | 0.4 chỉ trùng 1 đơn vị phụ | 0.2 không so được | 0.1 khác cấp (tòa huyện vs tòa tỉnh
    cùng địa bàn) | 0.0 mâu thuẫn (cùng loại đơn vị, khác tên) | None không đủ thông tin."""
    if not ref_court or not cand_court:
        return None
    ru, cu = court_units(ref_court), court_units(cand_court)
    if not ru or not cu:
        if _rf_fuzz is not None:
            r = _rf_fuzz.token_set_ratio(strip_diacritics(ref_court).lower(), strip_diacritics(cand_court).lower())
            return 0.8 if r >= 92 else None
        return None
    pr, pc = ru[0], cu[0]
    if _same_unit(pr, pc):
        return 1.0
    if any(_same_unit(pr, v) for v in cu[1:]) or any(_same_unit(u, pc) for u in ru[1:]):
        return 0.1
    if _type_compat(pr[0], pc[0]):
        return 0.0
    if any(_same_unit(u, v) for u in ru for v in cu):
        return 0.4
    return 0.2


# --------------------------------------------------------------------------------------
# Index metadata
# --------------------------------------------------------------------------------------
class DocIndex:
    def __init__(self, table: pa.Table, codenorm: CodeNormalizer):
        cols = table.column_names
        g = lambda c: table.column(c).to_pylist() if c in cols else [None] * table.num_rows  # noqa: E731
        self.doc_name = g("doc_name")
        self.official_id = g("official_document_id")
        self.level = g("instance_level")
        self.court = g("court")
        self.court_level = g("court_level")
        self.category = g("category")
        raw_date = g("issued_date")
        self.date = [str(d)[:10] if d is not None else None for d in raw_date]
        nums, years, codes = g("number"), g("year"), g("code")
        self.codenorm = codenorm
        self.exact = defaultdict(list)
        self.loose = defaultdict(list)
        self.by_name = {}
        self.key_of = [None] * table.num_rows
        bad = 0
        for i in range(table.num_rows):
            self.by_name[self.doc_name[i]] = i
            num, year, code = nums[i], years[i], codes[i]
            oid = self.official_id[i]
            if (num is None or year is None or num != num or year != year or not code) and oid:
                mm = CASE_RE.search(normalize_text(str(oid)))
                if mm:
                    num, year, code = mm["num"], mm["year"], code or mm["code"]
            try:
                num, year = int(float(num)), int(float(year))
            except (TypeError, ValueError):
                bad += 1
                continue
            atoms = codenorm.key(code, strict=False) if code else None
            if atoms is None:
                bad += 1
                continue
            self.key_of[i] = (num, year, atoms)
            self.exact[(num, year, atoms)].append(i)
            self.loose[(num, year)].append(i)
        log.info("Index: %d văn bản, %d khóa (số/năm/mã), %d văn bản không có số hợp lệ",
                 table.num_rows, len(self.exact), bad)

    def meta(self, i):
        return dict(doc_name=self.doc_name[i], official_id=self.official_id[i], level=self.level[i],
                    court=self.court[i], court_level=self.court_level[i], date=self.date[i],
                    category=self.category[i])


def _date_state(a, b):
    if not a or not b:
        return None
    if a == b:
        return "eq"
    try:
        da, db = date.fromisoformat(a), date.fromisoformat(b)
        if abs((da - db).days) <= 7:
            return "near"
    except ValueError:
        pass
    return "conflict"


def evaluate(ref, m, mode):
    ds = _date_state(ref["cited_date"], m["date"])
    if ds == "conflict":
        return 0, -3.0, ["date:conflict"]
    cs = court_sim(ref["cited_court"], m["court"])
    lvl = ref["cited_level"]
    level_conflict = bool(lvl and m["level"] and LEVEL_RANK.get(lvl) != LEVEL_RANK.get(m["level"]))
    score = {"eq": 3, "near": 1, "conflict": -3, None: 0}[ds]
    if cs is not None:
        score += 2 if cs >= 0.99 else 1.5 if cs >= 0.6 else 0.5 if cs >= 0.35 else -1 if cs > 0.05 else -2
    score += -1 if level_conflict else (0.5 if lvl and m["level"] else 0)
    why = [f"date:{ds}" if ds else "date:?", f"court:{cs}" if cs is not None else "court:?"]
    if level_conflict:
        why.append("level_conflict")

    if ds == "conflict" or (cs is not None and cs < 0.35):
        tier = 0
    elif ds == "eq" and cs is not None and cs >= 0.6:
        tier = 3
    elif ds == "eq" and (cs is None or cs >= 0.35):
        tier = 2
    elif ds == "near" and cs is not None and cs >= 0.6:
        tier = 2
    elif ds is None and cs is not None and cs >= 0.99:
        tier = 2
    elif ds == "eq" or ds == "near" or (ds is None and cs is not None and cs >= 0.6):
        tier = 1
    elif ds is None and mode == "exact":  # không có ngày để kiểm; tòa không rõ / chỉ khớp một phần
        tier = 1
    else:
        tier = 0
    if mode == "loose":
        tier = min(tier, 2) if ds in ("eq", "near") else 0
    if level_conflict and tier > 1:
        tier -= 1
    return tier, score, why


def match_ref(ref, index: DocIndex, exclude_doc=None):
    key = (ref["number"], ref["year"], tuple(ref["atoms"]))
    cands = [i for i in index.exact.get(key, []) if index.doc_name[i] != exclude_doc]
    mode = "exact"
    if not cands:
        mode = "loose"
        for i in index.loose.get((ref["number"], ref["year"]), []):
            if index.doc_name[i] == exclude_doc:
                continue
            ka = index.key_of[i][2]
            st_c, st_r = stage_of(ka), ref["stage"]
            dm_c, dm_r = domain_of(ka), ref["domain"]
            if st_r and st_c and st_r != st_c:
                continue
            if dm_r and dm_c and dm_r != dm_c:
                continue
            cands.append(i)
    res = dict(status="not_found", match_doc=None, match_official_id=None, match_level=None,
               match_court=None, match_date=None, match_score=None, confidence=None, match_mode=None,
               match_reasons=None, n_candidates=len(cands), alt_candidates=None)
    if not cands:
        return res
    scored = []
    for i in cands:
        m = index.meta(i)
        tier, score, why = evaluate(ref, m, mode)
        scored.append((tier, score, i, why))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[0]
    alts = [dict(doc=index.doc_name[i], id=index.official_id[i], court=index.court[i], date=index.date[i],
                 tier=TIER_NAME[t], score=round(s, 2)) for t, s, i, _ in scored[:5]]
    res["alt_candidates"] = json.dumps(alts, ensure_ascii=False)
    res["match_mode"] = mode
    if top[0] == 0:
        res["status"] = "candidates_rejected"  # có số trùng nhưng sai ngày/tòa → bản án khác
        return res
    if len(scored) > 1 and scored[1][0] == top[0] and top[1] - scored[1][1] < 0.5:
        res["status"] = "ambiguous"
        res["confidence"] = TIER_NAME[top[0]]
        return res
    m = index.meta(top[2])
    res.update(status="matched", match_doc=m["doc_name"], match_official_id=m["official_id"],
               match_level=m["level"], match_court=m["court"], match_date=m["date"],
               match_score=round(top[1], 2), confidence=TIER_NAME[top[0]], match_reasons=",".join(top[3]))
    return res


# --------------------------------------------------------------------------------------
# Xử lý 1 văn bản
# --------------------------------------------------------------------------------------
def doc_self_keys(row, index, codenorm):
    keys = set()
    i = index.by_name.get(row["doc_name"]) if index is not None else None
    if i is not None and index.key_of[i]:
        keys.add(index.key_of[i])
    oid = row.get("official_document_id")
    if oid:
        mm = CASE_RE.search(normalize_text(str(oid)))
        if mm:
            a = codenorm.key(mm["code"], strict=False)
            if a:
                keys.add((int(mm["num"]), int(mm["year"]), a))
    return keys


def process_doc(row: dict, index: DocIndex, codenorm: CodeNormalizer, depth: int, use_cc: bool):
    doc = row["doc_name"]
    oid = row.get("official_document_id")
    i = index.by_name.get(doc)
    self_keys = doc_self_keys(row, index, codenorm)
    _, mentions, _ = collect_mentions(row.get("markdown") or "", row.get("citations_case"), codenorm, use_cc)
    refs = group_mentions(mentions, self_keys)
    src_meta = index.meta(i) if i is not None else dict(level=row.get("instance_level"), official_id=oid)
    src_rank = LEVEL_RANK.get(src_meta.get("level"))
    for r in refs:
        r.update(match_ref(r, index, exclude_doc=doc))
        r["source_doc"] = doc
        r["source_official_id"] = src_meta.get("official_id")
        r["source_level"] = src_meta.get("level")
        r["depth"] = depth
        tgt_rank = LEVEL_RANK.get(r["match_level"] or r["cited_level"])
        r["direction"] = (None if not (src_rank and tgt_rank) else
                          "lower" if tgt_rank < src_rank else "same" if tgt_rank == src_rank else "higher")
    return refs


# --------------------------------------------------------------------------------------
# I/O với DuckDB
# --------------------------------------------------------------------------------------
class LoadedTable(str):
    """Tên bảng DuckDB đã nạp sẵn toàn bộ dữ liệu (dùng thay cho đường dẫn parquet)."""
    sig = ""
    files: list = []


def _src(data) -> str:
    """data: bảng đã nạp (LoadedTable), 1 file .parquet, 1 glob, hoặc list các file."""
    if isinstance(data, LoadedTable):
        return str(data)
    if isinstance(data, (list, tuple)):
        inner = "[" + ", ".join("'" + str(x).replace("'", "''") + "'" for x in data) + "]"
    else:
        inner = "'" + str(data).replace("'", "''") + "'"
    return f"read_parquet({inner}, union_by_name=true)"


REQUIRED_COLS = {"doc_name", "instance_level", "markdown"}
USEFUL_COLS = {"official_document_id", "number", "year", "code", "court", "issued_date", "category"}


def check_columns(cols: set, data):
    """Thoát kèm danh sách cột nếu file parquet thiếu cột bắt buộc."""
    missing = REQUIRED_COLS - cols
    if missing:
        sys.exit(f"File parquet {data} thiếu cột bắt buộc {sorted(missing)}.\n"
                 f"  Các cột hiện có: {sorted(cols)}\n"
                 f"  -> Cần schema config `documents` của tmquan/cbba-toaan-gov-vn")
    weak = USEFUL_COLS - cols
    if weak:
        log.warning("Thiếu cột %s → khớp bản án sẽ kém chính xác hơn", sorted(weak))
    if not ({"official_document_id", "number"} & cols):
        sys.exit("Không có official_document_id lẫn number → không thể dựng index số bản án")


def data_signature(data) -> str:
    """Chữ ký của bộ dữ liệu (danh sách file + kích thước + thời gian sửa) để biết cache còn đúng không."""
    if isinstance(data, LoadedTable):
        return data.sig
    files = data if isinstance(data, (list, tuple)) else [data]
    items = []
    for f in files:
        try:
            st = os.stat(f)
            items.append([os.path.realpath(f), st.st_size, int(st.st_mtime)])
        except OSError:
            items.append([str(f), None, None])
    return json.dumps(sorted(items), ensure_ascii=False)


def load_index(con, data, out_dir: Path, rebuild: bool) -> pa.Table:
    path = out_dir / "doc_index.parquet"
    meta = out_dir / "doc_index.meta.json"
    sig = data_signature(data)
    if path.exists() and not rebuild and meta.exists() and meta.read_text(encoding="utf-8") == sig:
        log.info("Dùng index cache: %s", path)
        return pq.read_table(path)
    if path.exists():
        log.info("Dữ liệu đã thay đổi (hoặc --rebuild-index) → dựng lại index")
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_src(data)}").fetchall()]
    want = ["doc_name", "official_document_id", "number", "year", "code", "category", "instance_level",
            "court", "court_level", "issued_date"]
    sel = ", ".join(c if c in cols else f"NULL AS {c}" for c in want)
    t0 = time.time()
    log.info("Đọc metadata toàn bộ dataset (không đọc markdown)...")
    con.execute(f"COPY (SELECT {sel} FROM {_src(data)}) TO '{path}' (FORMAT parquet)")
    meta.write_text(sig, encoding="utf-8")
    log.info("  xong sau %.1fs → %s", time.time() - t0, path)
    return pq.read_table(path)


def iter_docs(con, data, names, batch_size, has_cc: bool):
    if not names:
        return
    cols = "doc_name, official_document_id, instance_level, markdown" + (", citations_case" if has_cc else "")
    con.register("want_tbl", pa.table({"doc_name": pa.array(sorted(names), pa.string())}))
    try:
        q = f"SELECT {cols} FROM {_src(data)} WHERE doc_name IN (SELECT doc_name FROM want_tbl)"
        for rb in _arrow_batches(con, q, batch_size):
            yield from rb.to_pylist()
    finally:
        try:
            con.unregister("want_tbl")
        except Exception:  # noqa: BLE001
            pass


def _arrow_batches(con, q, batch_size):
    """Đọc kết quả theo batch, tương thích DuckDB 0.10 → 2.x."""
    rel = con.sql(q)
    for name in ("to_arrow_reader", "fetch_arrow_reader", "record_batch"):
        f = getattr(rel, name, None)
        if f is not None:
            try:
                return f(batch_size)
            except TypeError:
                return f()
    return con.execute(q).fetch_record_batch(batch_size)


def learn_extra_code_chunks(meta: pa.Table, min_count: int) -> set:
    tmp = CodeNormalizer()
    cnt = Counter()
    vc = Counter(x for x in meta.column("code").to_pylist() if x)
    for code, n in vc.items():
        for ch in clean_code(code).split("-"):
            if ch and _decompose(ch, tmp.atoms) is None and ch.isalpha() and len(ch) <= 12:
                cnt[ch] += n
    extra = {c for c, n in cnt.items() if n >= min_count and c not in CODE_BLACKLIST}
    log.info("Học thêm %d ký hiệu từ cột `code` của dataset: %s", len(extra), sorted(extra)[:40])
    return extra


# --------------------------------------------------------------------------------------
# Dựng chuỗi: đi từ văn bản gốc xuống từng cấp cho đến khi gặp bản án sơ thẩm
# --------------------------------------------------------------------------------------
def build_chains(roots, edges_by_src, index: DocIndex, min_tier: int, max_hops: int, max_paths: int = 50):
    """Mỗi đường đi: gốc (VD TT) → [GĐT] → [PT] → ST. Chỉ đi theo cạnh 'lower' đã khớp với độ tin cậy
    ≥ min_tier. Nếu một nút (VD PT) không có bản án cấp dưới nào trong dataset, nhưng văn bản cấp trên
    của nó có nêu đúng 1 bản án cấp thấp hơn và hợp lý về thời gian, thì nối tiếp với link = via_ancestor."""

    def meta(d):
        i = index.by_name.get(d)
        return index.meta(i) if i is not None else {"doc_name": d}

    def rank(d):
        return LEVEL_RANK.get(meta(d).get("level"), 0)

    kids_cache = {}

    def children(d):
        if d not in kids_cache:
            best = {}
            for e in edges_by_src.get(d, []):
                if e["status"] == "matched" and e["direction"] == "lower" and TIER_OF[e["confidence"]] >= min_tier:
                    cur = best.get(e["match_doc"])
                    if cur is None or (TIER_OF[e["confidence"]], e["match_score"] or 0) > \
                            (TIER_OF[cur["confidence"]], cur["match_score"] or 0):
                        best[e["match_doc"]] = e
            kids_cache[d] = sorted(best.values(), key=lambda e: -LEVEL_RANK.get(e["match_level"], 0))
        return kids_cache[d]

    rows, nested = [], []
    for root in roots:
        rmeta = meta(root)
        paths = []

        def dfs(node, hops, visited):
            if len(paths) >= max_paths:
                return
            if hops and rank(node) == 1:
                paths.append((hops, True))
                return
            if len(hops) >= max_hops:
                paths.append((hops, False))
                return
            kids = [(e, "cited") for e in children(node) if e["match_doc"] not in visited]
            if not kids and hops:
                ancestors = [root] + [h["doc"] for h in hops[:-1]]
                nr, nd = rank(node), meta(node).get("date")
                for anc in reversed(ancestors):
                    same_lvl = [e for e in children(anc) if LEVEL_RANK.get(e["match_level"]) == nr]
                    if len(same_lvl) != 1:
                        continue
                    inh = [e for e in children(anc) if LEVEL_RANK.get(e["match_level"], 0) < nr
                           and e["match_doc"] not in visited
                           and not (nd and e["match_date"] and e["match_date"] > nd)]
                    if inh:
                        kids = [(e, "via_ancestor") for e in inh]
                        break
            if not kids:
                if hops:
                    paths.append((hops, False))
                return
            for e, link in kids:
                dfs(e["match_doc"], hops + [dict(doc=e["match_doc"], edge=e, link=link, parent=node)],
                    visited | {e["match_doc"]})

        dfs(root, [], {root})
        # bỏ đường đi bị chứa trong đường dài hơn (VD TT→ST khi đã có TT→PT→ST)
        docsets = [frozenset(h["doc"] for h in hp) for hp, _ in paths]
        keep = []
        for i, (hp, ok) in enumerate(paths):
            if any(j != i and docsets[i] < docsets[j] for j in range(len(paths))):
                continue
            if any(j < i and docsets[i] == docsets[j] for j in range(len(paths))):
                continue
            keep.append((hp, ok))

        node = dict(root=rmeta, reached_first_instance=any(ok for _, ok in keep), paths=[])
        for hp, ok in keep:
            chain_docs = [root] + [h["doc"] for h in hp]
            hop_out = []
            for k, h in enumerate(hp):
                e = h["edge"]
                also = [d for d in chain_docs[:k] if d != h["parent"]
                        and any(x["match_doc"] == h["doc"] for x in children(d))]
                hop_out.append(dict(level=e["match_level"], doc=meta(h["doc"]), cited_by=h["parent"], link=h["link"],
                                    also_cited_by=also, confidence=e["confidence"], outcome_by_parent=e["outcome"],
                                    cited_as=e["raw"], cited_court=e["cited_court"], cited_date=e["cited_date"]))
            node["paths"].append(dict(reached_first_instance=ok, hops=hop_out))
            rows.append(_chain_row(rmeta, hop_out, ok))
        if not keep:
            rows.append(_chain_row(rmeta, [], False))
        nested.append(node)
    return rows, nested


def _chain_row(rmeta, hops, reached):
    row = dict(root_doc=rmeta.get("doc_name"), root_id=rmeta.get("official_id"), root_level=rmeta.get("level"),
               root_court=rmeta.get("court"), root_date=rmeta.get("date"),
               n_hops=len(hops), reached_first_instance=reached,
               chain_confidence=(TIER_NAME[min(TIER_OF[h["confidence"]] for h in hops)] if hops else None),
               has_via_ancestor=any(h["link"] == "via_ancestor" for h in hops),
               path_ids=" → ".join([str(rmeta.get("official_id"))] + [str(h["doc"]["official_id"]) for h in hops]),
               path_levels=" → ".join([str(rmeta.get("level"))] + [str(h["level"]) for h in hops]))
    for lvl in ("Cassation", "Appellate", "First-instance"):
        p = LEVEL_SHORT[lvl]
        h = next((x for x in hops if x["level"] == lvl), None)
        row.update({f"{p}_doc": h and h["doc"]["doc_name"], f"{p}_id": h and h["doc"]["official_id"],
                    f"{p}_court": h and h["doc"]["court"], f"{p}_date": h and h["doc"]["date"],
                    f"{p}_outcome_by_parent": h and h["outcome_by_parent"], f"{p}_link": h and h["link"],
                    f"{p}_confirmed": bool(h and h["also_cited_by"]), f"{p}_confidence": h and h["confidence"]})
    return row


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
REF_COLS = ["source_doc", "source_official_id", "source_level", "depth", "raw", "number", "year", "code",
            "code_key", "stage", "cited_level", "domain", "cited_date", "cited_court", "outcome", "outcome_via",
            "in_decision", "n_mentions", "sources", "from_citations_case", "cc_role", "year_inferred", "direction", "status", "match_doc", "match_official_id",
            "match_level", "match_court", "match_date", "match_score", "confidence", "match_mode",
            "match_reasons", "n_candidates", "alt_candidates"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", "--data", dest="data", default=DATA_PATH,
                    help="thư mục chứa documents-*.parquet hoặc glob (mặc định DATA_PATH)")
    ap.add_argument("--out", default=OUT_DIR, help="thư mục output")
    ap.add_argument("--levels", default=",".join(LEVELS), help="instance_level của văn bản gốc, VD 'Retrial,Cassation'")
    ap.add_argument("--categories", default=None, help="lọc category, VD 'Civil,Criminal'")
    ap.add_argument("--max-depth", "--depth", dest="max_depth", type=int, default=6,
                    help="giới hạn an toàn số tầng đọc tiếp; bình thường dừng khi gặp sơ thẩm")
    ap.add_argument("--limit", type=int, default=LIMIT, help="chỉ lấy N văn bản gốc (chạy thử); 0 = toàn bộ")
    ap.add_argument("--seed", type=int, default=0, help="seed khi lấy mẫu --limit")
    ap.add_argument("--chain-min-confidence", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--min-code-count", type=int, default=20, help="ngưỡng học ký hiệu mới từ cột code")
    ap.add_argument("--no-citations-case", action="store_true", help="không dùng cột citations_case")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=0, help="số thread DuckDB (0 = mặc định)")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--no-load-all", dest="load_all", action="store_false", default=LOAD_ALL_FIRST,
                    help="không nạp trước, đọc thẳng từ file parquet")
    ap.add_argument("--load-to", default=LOAD_TO, help="file .duckdb để nạp dữ liệu ra đĩa thay vì RAM")
    ap.add_argument("--explain", nargs="*", default=None, help="doc_name cần in chi tiết rồi thoát")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con, args.data, cols = open_data(args.data, args.load_all, args.load_to, args.threads)
    has_cc = "citations_case" in cols and not args.no_citations_case

    meta = load_index(con, args.data, out, args.rebuild_index)
    codenorm = CodeNormalizer(learn_extra_code_chunks(meta, args.min_code_count))
    index = DocIndex(meta, codenorm)

    if args.explain is not None:
        for row in iter_docs(con, args.data, set(args.explain), args.batch_size, has_cc):
            refs = process_doc(row, index, codenorm, 1, has_cc)
            print(f"\n=== {row['doc_name']}  {row.get('official_document_id')}  ({row.get('instance_level')})")
            flat = normalize_text(row.get("markdown") or "")
            ds = decision_section_start(flat)
            print("--- phần QUYẾT ĐỊNH (tối đa 1500 ký tự):\n", flat[ds: ds + 1500] if ds < len(flat) else "(không tìm thấy)")
            cc = row.get("citations_case") or []
            print(f"--- citations_case: {len(cc)} phần tử")
            for it in list(cc)[:30]:
                print("   ", {k: v for k, v in (it.items() if isinstance(it, dict) else [("value", it)])
                              if k not in ("sentence_id",)}, "→", parse_cc_item(it, codenorm))
            print(f"--- {len(refs)} tham chiếu sau khi gộp:")
            for r in refs:
                print(json.dumps({k: r[k] for k in REF_COLS if k in r and k != "alt_candidates"},
                                 ensure_ascii=False, indent=1))
                if r["alt_candidates"]:
                    print("  candidates:", r["alt_candidates"])
        return

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    lv = meta.column("instance_level").to_pylist()
    cat = meta.column("category").to_pylist()
    names = meta.column("doc_name").to_pylist()
    cats = {x.strip() for x in args.categories.split(",")} if args.categories else None
    roots = [n for n, l, c in zip(names, lv, cat) if l in levels and (cats is None or c in cats)]
    log.info("Văn bản gốc (%s): %d", ",".join(levels), len(roots))
    if args.limit and len(roots) > args.limit:
        import random
        random.Random(args.seed).shuffle(roots)
        roots = roots[: args.limit]
        log.info("  lấy mẫu %d văn bản", len(roots))

    all_refs = []
    processed = set()
    frontier = set(roots)
    for depth in range(1, args.max_depth + 1):
        t0, n = time.time(), 0
        lv_count = Counter(index.level[index.by_name[d]] if d in index.by_name else None for d in frontier)
        log.info("Tầng %d: đọc %d văn bản %s", depth, len(frontier), dict(lv_count))
        for row in iter_docs(con, args.data, frontier, args.batch_size, has_cc):
            if row["doc_name"] in processed:
                continue
            processed.add(row["doc_name"])
            all_refs.extend(process_doc(row, index, codenorm, depth, has_cc))
            n += 1
            if n % 2000 == 0:
                log.info("  %d/%d văn bản, %.0f doc/s", n, len(frontier), n / (time.time() - t0 + 1e-9))
        log.info("  xong %d văn bản sau %.1fs", n, time.time() - t0)
        nxt = set()
        for r in all_refs:
            # đi tiếp xuống mọi bản án cấp dưới tìm được, trừ sơ thẩm (điểm dừng)
            if (r["depth"] == depth and r["status"] == "matched" and r["direction"] == "lower"
                    and LEVEL_RANK.get(r["match_level"], 0) > 1 and r["match_doc"] not in processed):
                nxt.add(r["match_doc"])
        frontier = nxt
        if not frontier:
            break

    # ---- ghi refs / edges ----
    refs_tbl = pa.Table.from_pylist([{k: r.get(k) for k in REF_COLS} for r in all_refs])
    if refs_tbl.num_rows:
        pq.write_table(refs_tbl, out / "refs.parquet")
    edges = [r for r in all_refs if r["status"] == "matched"]
    _write_csv(out / "edges.csv", edges, ["source_doc", "source_official_id", "source_level", "match_doc",
                                          "match_official_id", "match_level", "direction", "outcome",
                                          "confidence", "match_reasons", "cited_date", "match_date",
                                          "cited_court", "match_court", "depth", "raw"])
    edges_by_src = defaultdict(list)
    for e in edges:
        edges_by_src[e["source_doc"]].append(e)

    rows, nested = build_chains(roots, edges_by_src, index, TIER_OF[args.chain_min_confidence], args.max_depth)
    _write_csv(out / "chains.csv", rows, list(rows[0].keys()) if rows else [])
    with open(out / "chains.jsonl", "w", encoding="utf-8") as f:
        for nd in nested:
            f.write(json.dumps(nd, ensure_ascii=False) + "\n")

    # ---- thống kê ----
    d1 = [r for r in all_refs if r["depth"] == 1]
    st = Counter(r["status"] for r in d1)
    conf = Counter(r["confidence"] for r in d1 if r["status"] == "matched")
    lower_d1 = [r for r in d1 if r["status"] == "matched" and r["direction"] == "lower"]
    summary = dict(
        roots=len(roots), docs_processed=len(processed),
        refs_depth1=len(d1), refs_depth1_by_cited_stage=dict(Counter(r["stage"] for r in d1)),
        refs_depth1_status=dict(st), matched_depth1_confidence=dict(conf),
        roots_with_any_lower_match=len({r["source_doc"] for r in lower_d1}),
        roots_reached_first_instance=sum(1 for nd in nested if nd["reached_first_instance"]),
        chain_rows=len(rows), chain_min_confidence=args.chain_min_confidence,
        chain_hops=dict(Counter(r["n_hops"] for r in rows)),
        chain_level_patterns=dict(Counter(r["path_levels"].split(" → ", 1)[-1] if r["n_hops"] else "-"
                                          for r in rows).most_common(15)),
        outcome_counts=dict(Counter(r["outcome"] for r in lower_d1)),
        refs_depth1_by_source=dict(Counter(r["sources"] for r in d1)),
        refs_depth1_in_decision_section=sum(1 for r in d1 if r["in_decision"]),
        refs_depth1_outcome_via=dict(Counter(r["outcome_via"] for r in d1 if r["outcome_via"])),
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Tóm tắt:\n%s", json.dumps(summary, ensure_ascii=False, indent=2))


def find_parquet_files(p):
    """DATA_PATH: thư mục (quét đệ quy mọi *.parquet) | 1 file | glob | list các mục trên.
    Chỉ giữ file có schema `documents` (đủ cột doc_name, instance_level, markdown) → tự bỏ embeddings-*, reduces-*."""
    items = list(p) if isinstance(p, (list, tuple)) else [p]
    if len(items) == 1 and str(items[0]).startswith("hf://"):
        return str(items[0])
    files = []
    for it in items:
        it = os.path.expanduser(str(it))
        if os.path.isdir(it):
            files += glob.glob(os.path.join(it, "**", "*.parquet"), recursive=True)
        elif os.path.isfile(it):
            files.append(it)
        else:
            files += glob.glob(it, recursive=True)
    uniq, seen = [], set()
    for f in sorted(files):
        rp = os.path.realpath(f)
        if rp not in seen:  # cùng 1 file qua symlink (VD cache của huggingface_hub)
            seen.add(rp)
            uniq.append(f)
    if not uniq:
        sys.exit(f"Không thấy file .parquet nào ở: {p}  -> sửa DATA_PATH ở đầu file")
    keep, skipped = [], []
    for f in uniq:
        try:
            names = set(pq.read_schema(f).names)
        except Exception as e:  # noqa: BLE001
            skipped.append((f, f"không đọc được: {type(e).__name__}"))
            continue
        miss = REQUIRED_COLS - names
        if miss:
            skipped.append((f, f"không phải schema documents, thiếu {sorted(miss)}"))
            continue
        keep.append(f)
    log.info("Tìm thấy %d file .parquet → dùng %d file có schema `documents`", len(uniq), len(keep))
    for f, why in skipped[:8]:
        log.info("  bỏ qua %s (%s)", f, why)
    if len(skipped) > 8:
        log.info("  ... và %d file khác bị bỏ qua", len(skipped) - 8)
    if not keep:
        sys.exit(f"Không có file .parquet nào có đủ cột {sorted(REQUIRED_COLS)} ở: {p}")
    return keep


def resolve_data_path(p):
    return find_parquet_files(p)


LOAD_COLS = ["doc_name", "official_document_id", "number", "year", "code", "category", "instance_level", "court",
             "court_level", "issued_date", "web_url", "pdf_url", "markdown", "citations_case"]


def open_data(data_path, load_all: bool = True, load_to=None, threads: int = 0):
    """Mở dữ liệu. load_all=True: NẠP HẾT mọi file parquet vào 1 bảng DuckDB trước, các bước sau đọc từ bảng đó.
    load_to: None = nạp vào RAM; đường dẫn file .duckdb = nạp ra đĩa (ít RAM, lần sau dùng lại không phải nạp).
    Trả (con, data, cols): `data` đưa thẳng cho _src()."""
    import duckdb  # noqa: WPS433
    files = find_parquet_files(data_path)
    con = duckdb.connect(str(load_to) if load_to else ":memory:")
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    con.execute("SET preserve_insertion_order = false")
    if isinstance(files, str) and files.startswith("hf://"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_src(files)}").fetchall()}
    check_columns(cols, files if isinstance(files, str) else f"({len(files)} file)")
    if not load_all:
        return con, files, cols
    sig = data_signature(files)
    tbl = LoadedTable("docs")
    tbl.sig, tbl.files = sig, files if isinstance(files, list) else [files]
    if load_to:
        try:
            old = con.execute("SELECT sig FROM _load_meta").fetchone()[0]
            if old == sig:
                n = con.execute("SELECT count(*) FROM docs").fetchone()[0]
                log.info("Dùng dữ liệu đã nạp sẵn trong %s (%d văn bản)", load_to, n)
                return con, tbl, {r[0] for r in con.execute("DESCRIBE SELECT * FROM docs").fetchall()}
        except Exception:  # noqa: BLE001 — chưa nạp lần nào
            pass
    want = [c for c in LOAD_COLS if c in cols]
    t0 = time.time()
    log.info("Nạp toàn bộ %s vào DuckDB (%s)...", f"{len(tbl.files)} file" if isinstance(files, list) else files,
             load_to or "RAM")
    con.execute(f"CREATE OR REPLACE TABLE docs AS SELECT {', '.join(want)} FROM {_src(files)}")
    n, nd = con.execute("SELECT count(*), count(DISTINCT doc_name) FROM docs").fetchone()
    if nd < n:
        log.warning("Có %d dòng trùng doc_name giữa các file → giữ 1 dòng cho mỗi doc_name", n - nd)
        con.execute("CREATE OR REPLACE TABLE docs AS SELECT * FROM docs "
                    "QUALIFY row_number() OVER (PARTITION BY doc_name) = 1")
        n = nd
    by_lv = dict(con.execute("SELECT instance_level, count(*) FROM docs GROUP BY 1 ORDER BY 2 DESC").fetchall())
    log.info("  đã nạp %d văn bản sau %.0fs: %s", n, time.time() - t0, by_lv)
    if load_to:
        con.execute("CREATE OR REPLACE TABLE _load_meta AS SELECT ? AS sig", [sig])
    return con, tbl, set(want)


def _write_csv(path, rows, cols):
    import csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


if __name__ == "__main__":
    main()
