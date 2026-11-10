#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_refs_llm.py
===================
Dùng DeepSeek tự host (API kiểu OpenAI: vLLM / SGLang / LMDeploy ...) để đọc TOÀN VĂN các bản án
Tái thẩm trong tmquan/cbba-toaan-gov-vn và trích mọi bản án / quyết định của Tòa án được nhắc đến,
xuất ra 1 file JSON hoàn chỉnh.

Với mỗi bản án được tham chiếu, LLM trả: số hiệu, loại văn bản, cấp xét xử, lĩnh vực, ngày ban hành,
Tòa án ban hành, quan hệ với văn bản đang đọc (bị xem xét lại / bản án trước của cùng vụ / vụ khác),
kết quả xử lý ở phần QUYẾT ĐỊNH (hủy / giữ nguyên / sửa / đình chỉ) và câu trích nguyên văn.
Ngoài ra trích các quyết định kháng nghị.

Script tự kiểm tra đầu ra của LLM (không tin mù):
  * checks.number_in_text   : số hiệu có thật sự xuất hiện trong văn bản không (chống bịa)
  * checks.evidence_in_text : câu trích có đúng nguyên văn không
  * checks.regex_also_found : bộ trích regex (trace_case_chains.py) có cùng tìm thấy không
  * sources                 : nguồn tìm ra bản án đó: llm / markdown (regex) / citations_case
  * llm_missed              : LLM bỏ sót, được bổ sung từ regex hoặc cột citations_case

Chống bỏ sót: trước khi gọi LLM, script gom các số hiệu do regex tìm trong TOÀN văn (kể cả phần
QUYẾT ĐỊNH và câu "Hủy bản án phúc thẩm nêu trên") và trong cột citations_case, đưa vào prompt làm
gợi ý bắt buộc kiểm tra; sau đó gộp: cái nào LLM vẫn bỏ sót thì thêm từ regex / citations_case.
  * dataset_match           : bản án đó có trong dataset không (khớp số + ngày + tòa), doc_name nào

Chạy được tiếp nếu bị ngắt: kết quả từng văn bản ghi dần vào <out>/llm_records.jsonl, file JSON cuối
<out>/retrial_references.json được dựng lại từ đó (có thể chỉ dựng lại bằng --compile-only).

Cần file trace_case_chains.py nằm CÙNG THƯ MỤC.

Cách dùng
---------
    pip install duckdb pyarrow rapidfuzz openai
    1. Sửa phần CẤU HÌNH bên dưới (ít nhất DATA_PATH, BASE_URL, SERVED_NAME).
    2. python extract_refs_llm.py
       hoặc ghi đè nhanh không cần sửa file:
       python extract_refs_llm.py --data-path /data/cbba-toaan-gov-vn --limit 20
       python extract_refs_llm.py --limit 0                 # 0 = chạy toàn bộ
       python extract_refs_llm.py --compile-only            # chỉ dựng lại file JSON từ llm_records.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

# ============================== CẤU HÌNH – SỬA Ở ĐÂY ==============================
DATA_PATH = "/path/to/cbba-toaan-gov-vn"        # thư mục (lấy MỌI file .parquet bên trong) | 1 file .parquet | glob
LOAD_ALL_FIRST = True                           # nạp hết mọi file vào DuckDB trước rồi mới chạy
LOAD_TO = None                                  # None = nạp vào RAM | "/local/disk/cbba.duckdb" = nạp ra đĩa, lần sau dùng lại
OUT_DIR = "out_retrial_refs"                    # thư mục kết quả
LEVELS = ["Retrial"]                            # cấp xét xử của văn bản cần đọc: "Retrial", "Cassation", ...
LIMIT = 20                                      # None hoặc 0 = chạy toàn bộ

# Server model đã host sẵn
BASE_URL = "https://127.0.0.1:8000/v1"
SERVED_NAME = "deepseek-v4-flash"               # "auto" = lấy model đầu tiên server trả về ở /models
API_KEY = "empty"
VERIFY_SSL = True          # False nếu server https dùng chứng chỉ tự ký
EXTRA_BODY = {"chat_template_kwargs": {"thinking": False}}  # tắt thinking cho nhanh; None = không gửi thêm

# Agent
WORKERS = 8                # số request song song
MAX_TOKENS = 8192          # token đầu ra tối đa mỗi lần gọi
TEMPERATURE = 0.0
TIMEOUT = 900              # giây / request
STRUCTURED = "json_schema" # json_schema | guided_json | json_object | none (tự lùi nếu server không hỗ trợ)
MAX_CHARS = 60000          # văn bản dài hơn sẽ được cắt thành nhiều phần
LLM_INPUT = "full"         # "full": toàn văn | "condensed": đầu văn bản + các đoạn có số hiệu + toàn bộ phần QUYẾT ĐỊNH

MATCH_DATASET = True       # đối chiếu từng bản án được tham chiếu với dataset (doc_name nào)

# Chống bỏ sót
USE_CITATIONS_CASE = True  # dùng cột citations_case của dataset làm nguồn bổ sung
HINTS_TO_LLM = True        # đưa số hiệu regex + citations_case tìm được vào prompt để LLM kiểm tra
UNION_WITH_RULES = True    # số hiệu LLM bỏ sót nhưng regex / citations_case có → vẫn thêm vào kết quả
# ==================================================================================

import glob  # noqa: E402
import ssl  # noqa: E402
import urllib.request  # noqa: E402

import pyarrow as pa  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import trace_case_chains as T
except ImportError:  # pragma: no cover
    sys.exit("Cần file trace_case_chains.py nằm cùng thư mục với script này.")
try:
    import duckdb
except ImportError:  # pragma: no cover
    sys.exit("Thiếu thư viện: pip install duckdb pyarrow rapidfuzz openai")
try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, BadRequestError
    from openai import DefaultAsyncHttpxClient
except ImportError:  # pragma: no cover
    sys.exit("Thiếu thư viện: pip install openai")
try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:  # pragma: no cover
    _rf_fuzz = None

log = logging.getLogger("extract_refs_llm")
PROMPT_VERSION = "refs-v2"

# --------------------------------------------------------------------------------------
# Schema & prompt
# --------------------------------------------------------------------------------------
INSTANCE_LEVELS = ["First-instance", "Appellate", "Cassation", "Retrial", "Other"]
CATEGORIES = ["Civil", "Criminal", "Administrative", "Commercial", "Labor", "Marriage & Family", "Bankruptcy",
              "Economic"]
RELATIONS = ["under_review", "earlier_in_case", "related_case", "other"]
OUTCOMES = ["annulled", "partially_annulled", "upheld", "amended", "terminated", "not_decided"]

_NSTR = {"type": ["string", "null"]}
REF_SCHEMA = {
    "type": "object",
    "properties": {
        "case_number": {"type": "string"},
        "document_type": {"type": "string", "enum": ["Bản án", "Quyết định"]},
        "document_title": {"type": "string"},
        "instance_level": {"type": "string", "enum": INSTANCE_LEVELS},
        "case_category": {"type": ["string", "null"], "enum": CATEGORIES + [None]},
        "issued_date": _NSTR,
        "court": _NSTR,
        "relation": {"type": "string", "enum": RELATIONS},
        "outcome": {"type": "string", "enum": OUTCOMES},
        "evidence": {"type": "string"},
    },
    "required": ["case_number", "document_type", "document_title", "instance_level", "case_category",
                 "issued_date", "court", "relation", "outcome", "evidence"],
    "additionalProperties": False,
}
PROTEST_SCHEMA = {
    "type": "object",
    "properties": {
        "number": {"type": "string"},
        "issuer": _NSTR,
        "issued_date": _NSTR,
        "target_case_numbers": {"type": "array", "items": {"type": "string"}},
        "request": _NSTR,
    },
    "required": ["number", "issuer", "issued_date", "target_case_numbers", "request"],
    "additionalProperties": False,
}
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "annuls_lower_judgment": {"type": "boolean"},
        "upholds_lower_judgment": {"type": "boolean"},
        "amends_lower_judgment": {"type": "boolean"},
        "terminates_proceeding": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["annuls_lower_judgment", "upholds_lower_judgment", "amends_lower_judgment",
                 "terminates_proceeding", "summary"],
    "additionalProperties": False,
}
DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "referenced_judgments": {"type": "array", "items": REF_SCHEMA},
        "protests": {"type": "array", "items": PROTEST_SCHEMA},
        "decision": DECISION_SCHEMA,
    },
    "required": ["referenced_judgments", "protests", "decision"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Bạn là trợ lý pháp lý chuyên đọc bản án, quyết định của Tòa án Việt Nam. Nhiệm vụ của bạn là trích xuất "
    "CHÍNH XÁC các bản án, quyết định của Tòa án được nhắc đến trong văn bản. Chỉ dựa vào văn bản được cung cấp, "
    "không suy đoán, không bịa số hiệu. Chỉ trả về đúng một đối tượng JSON, không giải thích thêm."
)

USER_TEMPLATE = """Văn bản dưới đây là {doc_label}{own_part}{court_part}.{chunk_note}

Hãy trích xuất:

A. "referenced_judgments": MỌI bản án hoặc quyết định của Tòa án được nhắc đến trong văn bản (sơ thẩm, phúc thẩm, giám đốc thẩm, tái thẩm, quyết định công nhận sự thỏa thuận, quyết định đình chỉ, ...), dù của cùng vụ án hay vụ án khác. Mỗi bản án/quyết định chỉ liệt kê MỘT lần (gộp các lần nhắc lại). Mỗi phần tử gồm:
  - "case_number": số hiệu đúng như văn bản ghi, dạng "số/năm/ký hiệu", ví dụ "16/2023/DS-PT". Nếu văn bản chỉ viết "bản án phúc thẩm nêu trên" và không ghi số ở bất kỳ đâu thì để "".
  - "document_type": "Bản án" hoặc "Quyết định".
  - "document_title": tên như văn bản ghi, ví dụ "Bản án dân sự phúc thẩm".
  - "instance_level": "First-instance" (sơ thẩm), "Appellate" (phúc thẩm), "Cassation" (giám đốc thẩm), "Retrial" (tái thẩm) hoặc "Other".
  - "case_category": "Civil" (dân sự), "Criminal" (hình sự), "Administrative" (hành chính), "Commercial" (kinh doanh thương mại), "Labor" (lao động), "Marriage & Family" (hôn nhân và gia đình), "Bankruptcy" (phá sản), "Economic" (kinh tế), hoặc null.
  - "issued_date": ngày ban hành dạng YYYY-MM-DD; null nếu văn bản không ghi.
  - "court": Tòa án ban hành, ghi đầy đủ như văn bản, giữ nguyên tên viết tắt/ẩn danh (ví dụ "Tòa án nhân dân huyện DH, tỉnh Long An"); null nếu không ghi.
  - "relation": "under_review" nếu là bản án/quyết định bị kháng nghị và được xem xét lại trong chính văn bản này; "earlier_in_case" nếu là bản án/quyết định trước đó của cùng vụ án nhưng không bị xem xét lại trực tiếp; "related_case" nếu thuộc vụ án khác; "other".
  - "outcome": phần QUYẾT ĐỊNH của văn bản này xử lý nó thế nào: "annulled" (hủy toàn bộ), "partially_annulled" (hủy một phần), "upheld" (giữ nguyên), "amended" (sửa), "terminated" (đình chỉ), "not_decided" (không quyết định gì về nó).
  - "evidence": một đoạn trích NGUYÊN VĂN từ văn bản (tối đa 300 ký tự) có chứa số hiệu của nó.

B. "protests": các quyết định kháng nghị (của Chánh án hoặc Viện trưởng Viện kiểm sát) được nhắc đến, mỗi phần tử gồm "number" (số hiệu như văn bản ghi), "issuer" (người/cơ quan kháng nghị), "issued_date" (YYYY-MM-DD hoặc null), "target_case_numbers" (số hiệu các bản án/quyết định bị kháng nghị), "request" (tóm tắt đề nghị, tối đa 200 ký tự).

Đọc kỹ CẢ phần QUYẾT ĐỊNH ở cuối văn bản: mọi bản án/quyết định bị hủy, giữ nguyên, sửa, đình chỉ ở đó đều phải có trong kết quả; nếu phần QUYẾT ĐỊNH chỉ viết "bản án phúc thẩm nêu trên" thì dùng số hiệu của bản án đó ở phần trước.
{hints}
KHÔNG đưa vào "referenced_judgments": chính văn bản này{own_excl}; quyết định kháng nghị (để ở "protests"); văn bản pháp luật (Bộ luật, Luật, Nghị quyết, Nghị định, Thông tư, Án lệ, Công văn); số thụ lý; hợp đồng; kết luận giám định; cáo trạng; văn bản của cơ quan không phải Tòa án.

C. "decision": kết quả chung ở phần QUYẾT ĐỊNH của CHÍNH văn bản này đối với các bản án/quyết định cấp dưới:
  - "annuls_lower_judgment": true nếu hủy (toàn bộ hoặc một phần) bất kỳ bản án/quyết định nào của Tòa án cấp dưới.
  - "upholds_lower_judgment": true nếu giữ nguyên bản án/quyết định cấp dưới (kể cả "không chấp nhận kháng nghị/kháng cáo, giữ nguyên ...").
  - "amends_lower_judgment": true nếu sửa bản án/quyết định cấp dưới.
  - "terminates_proceeding": true nếu đình chỉ xét xử phúc thẩm / giám đốc thẩm / tái thẩm.
  - "summary": tóm tắt phần QUYẾT ĐỊNH (tối đa 300 ký tự).
  Câu "đề nghị hủy" của Viện kiểm sát/đương sự KHÔNG phải là kết quả. Nếu phần văn bản được cung cấp không có phần QUYẾT ĐỊNH thì để các giá trị false và summary = "".

Trả về JSON đúng dạng: {{"referenced_judgments": [...], "protests": [...], "decision": {{...}}}}

=== VĂN BẢN ===
{text}
=== HẾT VĂN BẢN ==="""

LEVEL_LABEL = {"Retrial": "bản án/quyết định tái thẩm", "Cassation": "bản án/quyết định giám đốc thẩm",
               "Appellate": "bản án/quyết định phúc thẩm", "First-instance": "bản án/quyết định sơ thẩm"}

# chuẩn hóa nếu model trả giá trị tiếng Việt thay vì enum
_LEVEL_MAP = {"sơ thẩm": "First-instance", "phúc thẩm": "Appellate", "giám đốc thẩm": "Cassation",
              "tái thẩm": "Retrial", "first instance": "First-instance"}
_OUTCOME_MAP = {"hủy": "annulled", "huỷ": "annulled", "hủy một phần": "partially_annulled", "giữ nguyên": "upheld",
                "sửa": "amended", "đình chỉ": "terminated", "không": "not_decided", "none": "not_decided"}
_CAT_MAP = {"dân sự": "Civil", "hình sự": "Criminal", "hành chính": "Administrative",
            "kinh doanh thương mại": "Commercial", "lao động": "Labor", "hôn nhân và gia đình": "Marriage & Family",
            "hôn nhân gia đình": "Marriage & Family", "phá sản": "Bankruptcy", "kinh tế": "Economic"}


# --------------------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------------------
def clean_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n").replace(" ", " ")
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


_NUM_LOOSE_RE = re.compile(r"\d{1,5}\s*/\s*(?:(?:19|20)\d{2}\s*/\s*)?[A-ZĐ]{2}")
_REF_PHRASE_RE = re.compile(r"\b(?:bản án|quyết định)\b[^.\n]{0,80}?\b(?:sơ thẩm|phúc thẩm|giám đốc thẩm|tái thẩm)\b",
                            re.I)


def condense_markdown(text: str, head: int = 3000, ctx: int = 700) -> str:
    """Giữ: phần đầu văn bản + các đoạn quanh số hiệu / cụm 'bản án ... sơ thẩm' + TOÀN BỘ phần QUYẾT ĐỊNH."""
    if len(text) <= head + 4000:
        return text
    ds = T.decision_section_start(text)
    spans = [(0, head)]
    if ds < len(text):
        spans.append((max(0, ds - 300), len(text)))
    for rx in (_NUM_LOOSE_RE, _REF_PHRASE_RE):
        for m in rx.finditer(text):
            spans.append((max(0, m.start() - ctx), min(len(text), m.end() + ctx)))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 50:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        # cắt ở ranh giới dòng cho dễ đọc
        if a > 0:
            nl = text.find("\n", a, min(b, a + 200))
            a = nl + 1 if nl >= 0 else a
        out.append(text[a:b].strip())
    return "\n[...]\n".join(x for x in out if x)


def split_chunks(text: str, max_chars: int, overlap: int):
    """Cắt văn bản dài theo ranh giới đoạn, các phần gối nhau `overlap` ký tự."""
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut > start:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_json_text(s: str):
    if s is None:
        raise ValueError("empty content")
    s = _THINK_RE.sub("", s).strip()
    if "</think>" in s:  # template chỉ in thẻ đóng
        s = s.split("</think>", 1)[1]
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("không tìm thấy JSON")
    frag = s[i: j + 1]
    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        frag = re.sub(r",\s*([}\]])", r"\1", frag)  # dấu phẩy thừa
        return json.loads(frag)


def norm_date(v):
    if not v or not isinstance(v, str):
        return None
    v = v.strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", v)
    if m:
        y, mo, d = m.groups()
    else:
        m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})", v) or \
            re.search(r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})", v)
        if not m:
            return None
        d, mo, y = m.groups()
    try:
        return date(int(y), int(mo), int(d)).isoformat()
    except ValueError:
        return None


def _enum(v, allowed, mapping, default):
    if isinstance(v, str):
        if v in allowed:
            return v
        k = T.fix_tones(v.strip().lower())
        for key, val in mapping.items():
            if key == k or (len(key) > 3 and key in k):
                return val
    return default


def coerce_output(obj) -> dict:
    """Ép đầu ra LLM về đúng shape, dù model trả thiếu/sai kiểu."""
    if isinstance(obj, list):
        obj = {"referenced_judgments": obj, "protests": []}
    if not isinstance(obj, dict):
        raise ValueError("JSON không phải object")
    refs = obj.get("referenced_judgments") or obj.get("references") or []
    prots = obj.get("protests") or []
    out_r, out_p = [], []
    for r in refs if isinstance(refs, list) else []:
        if not isinstance(r, dict):
            continue
        cat = r.get("case_category")
        out_r.append(dict(
            case_number=str(r.get("case_number") or "").strip(),
            document_type=r.get("document_type") if r.get("document_type") in ("Bản án", "Quyết định") else
            ("Quyết định" if "quyết định" in str(r.get("document_title", "")).lower() else "Bản án"),
            document_title=str(r.get("document_title") or "").strip(),
            instance_level=_enum(r.get("instance_level"), INSTANCE_LEVELS, _LEVEL_MAP, "Other"),
            case_category=None if cat in (None, "", "null") else _enum(cat, CATEGORIES, _CAT_MAP, None),
            issued_date=norm_date(r.get("issued_date")),
            court=(str(r["court"]).strip() or None) if r.get("court") else None,
            relation=_enum(r.get("relation"), RELATIONS, {}, "other"),
            outcome=_enum(r.get("outcome"), OUTCOMES, _OUTCOME_MAP, "not_decided"),
            evidence=str(r.get("evidence") or "").strip()[:600],
        ))
    for p in prots if isinstance(prots, list) else []:
        if not isinstance(p, dict):
            continue
        tg = p.get("target_case_numbers") or []
        out_p.append(dict(
            number=str(p.get("number") or "").strip(),
            issuer=p.get("issuer") or None,
            issued_date=norm_date(p.get("issued_date")),
            target_case_numbers=[str(x).strip() for x in (tg if isinstance(tg, list) else [tg]) if x],
            request=p.get("request") or None,
        ))
    dec = obj.get("decision")
    out_d = None
    if isinstance(dec, dict):
        b = lambda k: dec.get(k) is True or str(dec.get(k)).strip().lower() in ("true", "1", "yes", "có")  # noqa: E731
        out_d = dict(annuls_lower_judgment=b("annuls_lower_judgment"), upholds_lower_judgment=b("upholds_lower_judgment"),
                     amends_lower_judgment=b("amends_lower_judgment"), terminates_proceeding=b("terminates_proceeding"),
                     summary=str(dec.get("summary") or "").strip()[:600])
    return {"referenced_judgments": out_r, "protests": out_p, "decision": out_d}


def merge_decisions(decs):
    """Gộp kết quả chung giữa các phần của văn bản dài: OR các cờ, nối các summary khác rỗng."""
    decs = [d for d in decs if d]
    if not decs:
        return None
    keys = ("annuls_lower_judgment", "upholds_lower_judgment", "amends_lower_judgment", "terminates_proceeding")
    out = {k: any(d.get(k) for d in decs) for k in keys}
    out["summary"] = " | ".join(d["summary"] for d in decs if d.get("summary"))[:900]
    return out


# --------------------------------------------------------------------------------------
# Gọi LLM
# --------------------------------------------------------------------------------------
class TruncatedOutput(Exception):
    pass


class LLM:
    MODES = ["json_schema", "guided_json", "json_object", "none"]

    def __init__(self, args):
        self.args = args
        http_client = None if args.verify_ssl else DefaultAsyncHttpxClient(verify=False)
        self.client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0,
                                  http_client=http_client)
        self.model = args.model
        self.mode = args.structured
        self.extra_body = dict(args.extra_body or {})
        self._mode_lock = asyncio.Lock()

    async def resolve_model(self):
        if self.model and self.model != "auto":
            return self.model
        models = await self.client.models.list()
        ids = [m.id for m in models.data]
        if not ids:
            raise RuntimeError("Server không trả model nào ở /v1/models")
        self.model = ids[0]
        log.info("Model trên server: %s → dùng %s", ids, self.model)
        return self.model

    def _kwargs(self, messages, mode):
        kw = dict(model=self.model, messages=messages, temperature=self.args.temperature,
                  max_tokens=self.args.max_tokens)
        extra = dict(self.extra_body)
        if mode == "json_schema":
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "court_references", "schema": DOC_SCHEMA, "strict": True}}
        elif mode == "guided_json":
            extra["guided_json"] = DOC_SCHEMA
        elif mode == "json_object":
            kw["response_format"] = {"type": "json_object"}
        if extra:
            kw["extra_body"] = extra
        return kw

    async def _degrade(self, failed_mode, err):
        async with self._mode_lock:
            if self.mode == failed_mode and failed_mode != "none":
                nxt = self.MODES[self.MODES.index(failed_mode) + 1]
                log.warning("Server không nhận structured output '%s' (%s) → chuyển sang '%s'",
                            failed_mode, str(err)[:200], nxt)
                self.mode = nxt

    async def complete(self, messages):
        """Trả (content, usage, finish_reason). Tự lùi chế độ structured output nếu server báo 400."""
        attempt = 0
        while True:
            mode = self.mode
            try:
                resp = await self.client.chat.completions.create(**self._kwargs(messages, mode))
                ch = resp.choices[0]
                content = ch.message.content
                if not content:  # một số server để JSON trong reasoning_content khi bật thinking
                    content = getattr(ch.message, "reasoning_content", None) or ""
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                return content, usage, ch.finish_reason
            except BadRequestError as e:
                msg = str(e).lower()
                if mode != "none" and any(k in msg for k in ("response_format", "json_schema", "guided", "schema",
                                                             "grammar", "unsupported", "not support")):
                    await self._degrade(mode, e)
                    continue
                if "chat_template_kwargs" in msg and "chat_template_kwargs" in self.extra_body:
                    log.warning("Server không nhận chat_template_kwargs → bỏ khỏi EXTRA_BODY")
                    self.extra_body.pop("chat_template_kwargs", None)
                    continue
                if any(k in msg for k in ("maximum context", "context length", "too long", "max_model_len")):
                    raise TruncatedOutput("input quá dài so với context của model") from e
                raise
            except (APITimeoutError, APIConnectionError) as e:
                err = e
            except APIStatusError as e:
                if e.status_code not in (429, 500, 502, 503, 504):
                    raise
                err = e
            attempt += 1
            if attempt > self.args.retries:
                raise err
            await asyncio.sleep(min(60, 2 ** attempt + random.random()))


def format_hints(cands, text):
    """Gợi ý từ regex + citations_case; với văn bản bị cắt nhiều phần chỉ đưa số hiệu có trong phần đó."""
    if not cands:
        return ""
    ctext = re.sub(r"\s+", "", T.normalize_text(text)).lower()
    lines = []
    for c in cands:
        num = f"{c['number']}/{c['code']}" if c["year_inferred"] else f"{c['number']}/{c['year']}/{c['code']}"
        if not c["year_inferred"] and re.sub(r"\s+", "", num).lower() not in ctext and \
                not re.search(rf"(?<!\d)0*{c['number']}\s?/\s?{c['year']}\s?/", T.normalize_text(text)):
            if "markdown" in c["sources"]:
                continue  # số hiệu này nằm ở phần khác của văn bản
        info = [x for x in (c["cited_date"], c["cited_court"]) if x]
        tag = " [ở phần QUYẾT ĐỊNH]" if c["in_decision"] else ""
        src = " [từ citations_case]" if c["sources"] == "citations_case" else ""
        lines.append(f"- {num}" + (f" ({', '.join(info)})" if info else "") + tag + src)
        if len(lines) >= 80:
            break
    if not lines:
        return ""
    return ("\nGỢI Ý: bộ trích tự động (regex trên văn bản + cột citations_case) đã thấy các số hiệu sau. Danh sách "
            "có thể THIẾU hoặc THỪA. Số hiệu nào đúng là bản án/quyết định của Tòa án được nhắc trong văn bản thì "
            "PHẢI có trong kết quả; bổ sung cả những cái gợi ý bỏ sót; bỏ những cái không phải bản án/quyết định "
            "của Tòa án.\n" + "\n".join(lines) + "\n")


def build_prompt(row, text, part=None, cands=None):
    lvl = row.get("instance_level")
    oid = row.get("official_document_id")
    chunk_note = ""
    if part:
        i, n = part
        chunk_note = (f"\n(Đây là PHẦN {i}/{n} của văn bản; chỉ trích những gì xuất hiện trong phần này. "
                      f"Nếu phần này không chứa phần QUYẾT ĐỊNH thì để outcome = \"not_decided\".)")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            doc_label=LEVEL_LABEL.get(lvl, "bản án/quyết định của Tòa án"),
            own_part=f" số {oid}" if oid else "",
            court_part=f" của {row['court']}" if row.get("court") else "",
            own_excl=f" ({oid})" if oid else "",
            hints=format_hints(cands, text) if cands else "",
            chunk_note=chunk_note, text=text)},
    ]


async def extract_text(llm: LLM, row, text, part, depth, stats, cands=None):
    msgs = build_prompt(row, text, part, cands)
    try:
        content, usage, finish = await llm.complete(msgs)
    except TruncatedOutput:
        if depth < 3 and len(text) > 4000:  # input vượt context → chia nhỏ
            stats["splits"] += 1
            outs = [await extract_text(llm, row, h, part, depth + 1, stats, cands)
                    for h in split_chunks(text, len(text) // 2 + 1, 800)]
            return {"referenced_judgments": [r for o in outs for r in o["referenced_judgments"]],
                    "protests": [p for o in outs for p in o["protests"]],
                    "decision": merge_decisions([o.get("decision") for o in outs])}
        raise
    stats["calls"] += 1
    for k in ("prompt_tokens", "completion_tokens"):
        stats[k] += int(usage.get(k) or 0)
    if finish == "length":
        if depth < 2 and len(text) > 4000:  # output bị cắt → chia đôi phần văn bản rồi làm lại
            stats["splits"] += 1
            half = split_chunks(text, len(text) // 2 + 1, 800)
            outs = [await extract_text(llm, row, h, part, depth + 1, stats, cands) for h in half]
            return {"referenced_judgments": [r for o in outs for r in o["referenced_judgments"]],
                    "protests": [p for o in outs for p in o["protests"]],
                    "decision": merge_decisions([o.get("decision") for o in outs])}
        raise TruncatedOutput("output bị cắt (finish_reason=length); tăng --max-tokens")
    try:
        return coerce_output(parse_json_text(content))
    except (ValueError, json.JSONDecodeError) as e:
        stats["repairs"] += 1
        msgs2 = msgs + [{"role": "assistant", "content": (content or "")[:20000]},
                        {"role": "user", "content": "Câu trả lời trên không phải JSON hợp lệ theo yêu cầu. "
                                                    "Chỉ trả lại đúng một đối tượng JSON hợp lệ."}]
        content2, usage2, _ = await llm.complete(msgs2)
        stats["calls"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            stats[k] += int(usage2.get(k) or 0)
        try:
            return coerce_output(parse_json_text(content2))
        except (ValueError, json.JSONDecodeError):
            raise ValueError(f"LLM không trả JSON hợp lệ: {e}; đầu ra: {(content or '')[:300]!r}") from e


# --------------------------------------------------------------------------------------
# Hậu xử lý: chuẩn hóa số hiệu, gộp, kiểm tra, khớp dataset
# --------------------------------------------------------------------------------------
def parse_case_number(s: str, codenorm):
    if not s:
        return None
    m = T.CASE_RE.search(T.normalize_text(s))
    if not m:
        return None
    atoms = codenorm.key(m["code"], strict=True) or codenorm.key(m["code"], strict=False)
    if not atoms:
        return None
    return int(m["num"]), int(m["year"]), T.clean_code(m["code"]), atoms


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", T.normalize_text(s)).lower()


def merge_refs(refs, codenorm):
    """Gộp trùng (giữa các phần của văn bản dài, hoặc model liệt kê 2 lần)."""
    merged = {}
    order = []
    for r in refs:
        p = parse_case_number(r["case_number"], codenorm)
        key = (p[0], p[1], p[3], r["issued_date"]) if p else ("raw", r["case_number"] or None,
                                                              r["document_title"].lower(), r["court"], r["issued_date"])
        if p and r["issued_date"] is None:  # bản ghi thiếu ngày gộp vào bản ghi có ngày cùng số hiệu
            for k in merged:
                if k[0] == p[0] and k[1] == p[1] and k[2] == p[3]:
                    key = k
                    break
        if key not in merged:
            merged[key] = dict(r, _parsed=p)
            order.append(key)
            continue
        cur = merged[key]
        for f in ("issued_date", "court", "case_category", "document_title"):
            if not cur.get(f) and r.get(f):
                cur[f] = r[f]
        if cur["outcome"] == "not_decided" and r["outcome"] != "not_decided":
            cur["outcome"] = r["outcome"]
        if cur["relation"] in ("other",) and r["relation"] != "other":
            cur["relation"] = r["relation"]
        if r["relation"] == "under_review":
            cur["relation"] = "under_review"
        if len(r["evidence"]) > len(cur["evidence"]):
            cur["evidence"] = r["evidence"]
    return [merged[k] for k in order]


def rule_candidates(row, codenorm, index, use_cc=True):
    """Tham chiếu tìm bằng luật: regex trên toàn văn (kể cả phần QUYẾT ĐỊNH, câu 'nêu trên') + citations_case."""
    flat, mentions, _ = T.collect_mentions(row.get("markdown") or "", row.get("citations_case"), codenorm, use_cc)
    self_keys = T.doc_self_keys(row, index, codenorm)
    return flat, self_keys, T.group_mentions(mentions, self_keys)


_OUTCOME_FROM_RULE = {"huy": "annulled", "giu_nguyen": "upheld", "sua": "amended", "dinh_chi": "terminated"}
_DOMAIN2CAT = {"DS": "Civil", "HS": "Criminal", "HC": "Administrative", "KDTM": "Commercial", "LĐ": "Labor",
               "HNGĐ": "Marriage & Family", "PS": "Bankruptcy", "KT": "Economic", "VDS": "Civil"}


def _rule_sources(c):
    return [x for x in c["sources"].split("+") if x]


def ref_from_rule(c):
    """Bản án LLM bỏ sót → dựng bản ghi từ kết quả regex / citations_case."""
    ev = (c["raw"] or "").split(" … ")[0]
    title = ""
    m = re.match(r"^((?:bản án|quyết định)[^\d]{0,80}?)\s*số", ev, re.I)
    if m:
        title = m.group(1).strip()
    lvl = c["cited_level"] or "Other"
    outcome = _OUTCOME_FROM_RULE.get(c["outcome"], "not_decided")
    return dict(
        case_number=f"{c['number']}/{c['year']}/{c['code']}" if not c["year_inferred"] else f"{c['number']}/{c['code']}",
        document_type="Quyết định" if title.lower().startswith("quyết định") or (not title and "QĐ" in c["atoms"])
        else "Bản án",
        document_title=title, instance_level=lvl, case_category=_DOMAIN2CAT.get(c["domain"]),
        issued_date=c["cited_date"], court=c["cited_court"],
        relation="under_review" if (c["in_decision"] and outcome != "not_decided") else "other",
        outcome=outcome, evidence=ev[:400],
    )


def postprocess(row, raw, flat, self_keys, cands, codenorm, index, do_match, union=True):
    doc = row["doc_name"]
    compact_text = re.sub(r"\s+", "", flat).lower()

    # mọi số hiệu dạng N/YYYY/MÃ có trong văn bản (để kiểm LLM có bịa không)
    in_text = set()
    for m in T.CASE_RE.finditer(flat):
        a = codenorm.key(m["code"], strict=False)
        if a:
            in_text.add((int(m["num"]), int(m["year"]), a))
    cand_by_ny = {}
    for c in cands:
        cand_by_ny.setdefault((c["number"], c["year"]), []).append(c)
    used = set()

    def find_cand(p):
        for c in cand_by_ny.get((p[0], p[1]), []):
            if T.atoms_compatible(tuple(c["atoms"]), p[3]):
                return c
        return None

    refs, dropped = [], []
    for r in merge_refs(raw["referenced_judgments"], codenorm):
        p = r.pop("_parsed")
        key = (p[0], p[1], p[3]) if p else None
        if key and any(key[0] == s[0] and key[1] == s[1] and T.atoms_compatible(key[2], s[2])
                       and T.stage_of(key[2]) == T.stage_of(s[2]) for s in self_keys):
            dropped.append(dict(reason="self_reference", item=r))
            continue
        c = find_cand(p) if p else None
        if c is not None:
            used.add(id(c))
        ev = r["evidence"]
        ev_ok = None
        if ev:
            ce = _compact(ev)
            ev_ok = ce in compact_text
            if not ev_ok and _rf_fuzz is not None and len(ce) >= 20:
                ev_ok = _rf_fuzz.partial_ratio(ce, compact_text) >= 90
        num_ok = bool(key and key in in_text) or bool(r["case_number"] and _compact(r["case_number"]) in compact_text) \
            or c is not None
        filled = []
        if c is not None:  # LLM để trống mà luật có → bổ sung
            if r["outcome"] == "not_decided" and c["outcome"]:
                r["outcome"] = _OUTCOME_FROM_RULE.get(c["outcome"], r["outcome"])
                filled.append("outcome")
            if not r["issued_date"] and c["cited_date"]:
                r["issued_date"] = c["cited_date"]
                filled.append("issued_date")
            if not r["court"] and c["cited_court"]:
                r["court"] = c["cited_court"]
                filled.append("court")
        refs.append(_finalize(r, p, ["llm"] + (_rule_sources(c) if c else []), c, num_ok, ev_ok, filled,
                              llm_missed=False))

    rule_only = []
    for c in cands:
        if id(c) in used:
            continue
        r = ref_from_rule(c)
        p = (c["number"], c["year"], c["code"], tuple(c["atoms"]))
        if union:
            refs.append(_finalize(r, p, _rule_sources(c), c, True, True, [], llm_missed=True))
        else:
            rule_only.append(dict(r, sources=_rule_sources(c)))

    for out in refs:
        if do_match and index is not None:
            out["dataset_match"] = match_in_dataset(out, out.pop("_parsed"), index, doc)
        else:
            out.pop("_parsed", None)
    return refs, dropped, rule_only


def _finalize(r, p, sources, c, num_ok, ev_ok, filled, llm_missed):
    return dict(
        case_number=r["case_number"],
        case_number_normalized=f"{p[0]}/{p[1]}/{p[2]}" if p else None,
        number=p[0] if p else None, year=p[1] if p else None, code=p[2] if p else None,
        document_type=r["document_type"], document_title=r["document_title"],
        instance_level=r["instance_level"], case_category=r["case_category"],
        issued_date=r["issued_date"], court=r["court"],
        relation=r["relation"], outcome=r["outcome"], evidence=r["evidence"],
        in_decision_section=bool(c and c["in_decision"]),
        sources=sources, llm_missed=llm_missed, filled_from_rules=filled,
        checks=dict(number_in_text=bool(num_ok), evidence_in_text=ev_ok,
                    regex_also_found=bool(c and "markdown" in c["sources"]),
                    citations_case_also_found=bool(c and "citations_case" in c["sources"])),
        # verified: số hiệu có thật trong văn bản (hoặc trong citations_case) và câu trích không sai lệch
        verified=bool(num_ok and ev_ok is not False),
        _parsed=p,
    )


def match_in_dataset(out, parsed, index, exclude_doc):
    if not parsed:
        return dict(status="no_case_number")
    num, year, code, atoms = parsed
    stage = T.stage_of(atoms)
    lvl = out["instance_level"] if out["instance_level"] in T.LEVEL_RANK else T.STAGE_LEVEL.get(stage)
    ref = dict(number=num, year=year, atoms=atoms, stage=stage or {v: k for k, v in T.STAGE_LEVEL.items()}.get(lvl),
               cited_level=lvl, domain=T.domain_of(atoms), cited_date=out["issued_date"], cited_court=out["court"])
    m = T.match_ref(ref, index, exclude_doc=exclude_doc)
    res = dict(status=m["status"], doc_name=m["match_doc"], official_document_id=m["match_official_id"],
               instance_level=m["match_level"], court=m["match_court"], issued_date=m["match_date"],
               confidence=m["confidence"], reasons=m["match_reasons"], n_candidates=m["n_candidates"])
    if m["status"] in ("ambiguous", "candidates_rejected") and m["alt_candidates"]:
        res["candidates"] = json.loads(m["alt_candidates"])[:5]
    return res


# --------------------------------------------------------------------------------------
# Đọc dữ liệu
# --------------------------------------------------------------------------------------
def iter_rows(con, data, names, batch_size, cols_avail):
    want = ["doc_name", "official_document_id", "instance_level", "court", "court_level", "issued_date", "category",
            "web_url", "pdf_url", "markdown", "citations_case"]
    sel = ", ".join(c if c in cols_avail else f"NULL AS {c}" for c in want)
    con.register("want_tbl", pa.table({"doc_name": pa.array(sorted(names), pa.string())}))
    try:
        q = f"SELECT {sel} FROM {T._src(data)} WHERE doc_name IN (SELECT doc_name FROM want_tbl)"
        for rb in T._arrow_batches(con, q, batch_size):
            yield from rb.to_pylist()
    finally:
        try:
            con.unregister("want_tbl")
        except Exception:  # noqa: BLE001
            pass


def load_status(path: Path) -> dict:
    """doc_name → status, không giữ nội dung (để resume nhanh với file lớn)."""
    st = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                i = line.find('"status": "')
                d0 = line.find('"doc_name": "')
                if d0 < 0:
                    continue
                d1 = line.find('"', d0 + 13)
                doc = line[d0 + 13: d1]
                status = line[i + 11: line.find('"', i + 11)] if i >= 0 else "error"
                if st.get(doc) != "ok":
                    st[doc] = status
    return st


def load_records(path: Path):
    recs = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # dòng cuối bị ghi dở khi ngắt
                prev = recs.get(r["doc_name"])
                if prev is None or r.get("status") == "ok" or prev.get("status") != "ok":
                    recs[r["doc_name"]] = r
    return recs


# --------------------------------------------------------------------------------------
# Chạy
# --------------------------------------------------------------------------------------
async def run(args, con, roots, cols_avail, codenorm, index, rec_path: Path):
    llm = LLM(args)
    await llm.resolve_model()
    done = load_status(rec_path)
    skip = {d for d, st in done.items() if st == "ok" or not args.redo_errors}
    todo = [d for d in roots if d not in skip]
    log.info("Cần xử lý %d văn bản (%d đã có kết quả, bỏ qua)", len(todo), len(roots) - len(todo))
    if not todo:
        return llm.model
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    fout = open(rec_path, "a", encoding="utf-8")
    t0 = time.time()
    counters = Counter()

    async def worker(row):
        stats = Counter()
        t1 = time.time()
        rec = dict(doc_name=row["doc_name"], official_document_id=row.get("official_document_id"),
                   instance_level=row.get("instance_level"), court=row.get("court"),
                   court_level=row.get("court_level"), issued_date=str(row["issued_date"])[:10]
                   if row.get("issued_date") else None, category=row.get("category"),
                   web_url=row.get("web_url"), pdf_url=row.get("pdf_url"))
        try:
            text = clean_markdown(row.get("markdown") or "")
            if not text:
                raise ValueError("markdown rỗng")
            flat, self_keys, cands = rule_candidates(row, codenorm, index, args.use_cc)
            input_mode = getattr(args, "input_mode", "full")
            llm_text = condense_markdown(text) if input_mode == "condensed" else text
            parts = split_chunks(llm_text, args.max_chars, args.overlap)
            raw = {"referenced_judgments": [], "protests": []}
            decs = []
            for i, part in enumerate(parts, 1):
                o = await extract_text(llm, row, part, (i, len(parts)) if len(parts) > 1 else None, 0, stats,
                                       cands if args.hints else None)
                raw["referenced_judgments"] += o["referenced_judgments"]
                raw["protests"] += o["protests"]
                decs.append(o.get("decision"))
            rec["decision"] = merge_decisions(decs)
            rec["decision_rules"] = T.decision_flags(flat)
            stats["input_chars"] += len(llm_text)
            refs, dropped, rule_only = postprocess(row, raw, flat, self_keys, cands, codenorm, index,
                                                   not args.no_match, args.union)
            rec.update(status="ok", error=None, referenced_judgments=refs, protests=_dedupe_protests(raw["protests"]),
                       dropped=dropped)
            if rule_only:
                rec["rule_only_candidates"] = rule_only
            rec["n_citations_case"] = len(row.get("citations_case") or [])
            rec["llm"] = dict(model=llm.model, structured_mode=llm.mode, n_chunks=len(parts),
                              latency_s=round(time.time() - t1, 2), **stats)
            counters["ok"] += 1
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:500]}",
                       llm=dict(model=llm.model, latency_s=round(time.time() - t1, 2), **stats))
            counters["error"] += 1
            log.warning("Lỗi %s: %s", row["doc_name"], rec["error"])
        rec["prompt_version"] = PROMPT_VERSION
        async with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        n = counters["ok"] + counters["error"]
        if n % 20 == 0 or n == len(todo):
            el = time.time() - t0
            log.info("  %d/%d xong (%d lỗi) | %.2f văn bản/s | còn ~%.0f phút", n, len(todo), counters["error"],
                     n / el, (len(todo) - n) / max(n / el, 1e-9) / 60)

    async def guarded(row):
        try:
            await worker(row)
        finally:
            sem.release()

    tasks = set()
    try:
        for row in iter_rows(con, args.data, set(todo), args.batch_size, cols_avail):
            await sem.acquire()
            t = asyncio.create_task(guarded(row))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*list(tasks))
    finally:
        fout.close()
    return llm.model


def _dedupe_protests(prots):
    seen, out = set(), []
    for p in prots:
        k = (re.sub(r"\s+", "", p["number"]).lower(), p["issued_date"])
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def compile_json(rec_path: Path, out_path: Path, roots_order, meta):
    recs = load_records(rec_path)
    order = {d: i for i, d in enumerate(roots_order)}
    docs = sorted(recs.values(), key=lambda r: (order.get(r["doc_name"], 10 ** 12), r["doc_name"]))
    ok = [d for d in docs if d.get("status") == "ok"]
    refs = [r for d in ok for r in d["referenced_judgments"]]
    stats = dict(
        documents=len(docs), documents_ok=len(ok), documents_error=len(docs) - len(ok),
        documents_with_refs=sum(1 for d in ok if d["referenced_judgments"]),
        referenced_judgments=len(refs),
        by_instance_level=dict(Counter(r["instance_level"] for r in refs)),
        by_relation=dict(Counter(r["relation"] for r in refs)),
        by_outcome=dict(Counter(r["outcome"] for r in refs)),
        verified=sum(1 for r in refs if r.get("verified")),
        number_not_in_text=sum(1 for r in refs if not r["checks"]["number_in_text"]),
        evidence_not_verbatim=sum(1 for r in refs if r["checks"]["evidence_in_text"] is False),
        llm_only=sum(1 for r in refs if r.get("sources") == ["llm"]),
        llm_missed_added_from_rules=sum(1 for r in refs if r.get("llm_missed")),
        by_sources=dict(Counter("+".join(r.get("sources", [])) for r in refs)),
        in_decision_section=sum(1 for r in refs if r.get("in_decision_section")),
        filled_from_rules=dict(Counter(f for r in refs for f in r.get("filled_from_rules", []))),
        protests=sum(len(d["protests"]) for d in ok),
        dataset_match=dict(Counter((r.get("dataset_match") or {}).get("status") for r in refs)),
        dataset_match_confidence=dict(Counter((r.get("dataset_match") or {}).get("confidence") for r in refs
                                              if (r.get("dataset_match") or {}).get("status") == "matched")),
        prompt_tokens=sum(int((d.get("llm") or {}).get("prompt_tokens", 0)) for d in docs),
        completion_tokens=sum(int((d.get("llm") or {}).get("completion_tokens", 0)) for d in docs),
    )
    if meta.get("model") in (None, "auto"):
        used = Counter((d.get("llm") or {}).get("model") for d in docs if (d.get("llm") or {}).get("model"))
        meta["model"] = used.most_common(1)[0][0] if used else meta.get("model")
    payload = dict(meta=dict(meta, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             prompt_version=PROMPT_VERSION, stats=stats), documents=docs)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    return stats


def check_server(base_url, api_key, verify, served_name):
    """Gọi GET /models để chắc server đang chạy và có model SERVED_NAME. Trả về tên model sẽ dùng."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    ctx = None
    if base_url.startswith("https") and not verify:
        ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            models = [m.get("id") for m in json.loads(r.read()).get("data", [])]
    except ssl.SSLError as e:
        hint = ("\n  -> Nếu server chạy http thường, đổi BASE_URL thành http://..."
                "\n  -> Nếu là https chứng chỉ tự ký, đặt VERIFY_SSL = False")
        sys.exit(f"Lỗi SSL khi kết nối {base_url}: {e}{hint}")
    except Exception as e:  # noqa: BLE001
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            sys.exit(f"Lỗi SSL khi kết nối {base_url}: {e.reason}"
                     "\n  -> Nếu server chạy http thường, đổi BASE_URL thành http://..."
                     "\n  -> Nếu là https chứng chỉ tự ký, đặt VERIFY_SSL = False")
        sys.exit(f"Không kết nối được {base_url}: {e}\n  -> Kiểm tra server đã chạy và BASE_URL/cổng đúng chưa"
                 "\n  -> Nếu server chạy http thường, đổi BASE_URL thành http://...")
    print(f"Server OK, model đang phục vụ: {models}")
    if not served_name or served_name == "auto":
        if not models:
            sys.exit("Server không trả model nào ở /models -> đặt SERVED_NAME cụ thể")
        return models[0]
    if models and served_name not in models:
        print(f"CẢNH BÁO: không thấy '{served_name}' trong danh sách trên -> kiểm tra SERVED_NAME")
    return served_name


def resolve_data_path(p):
    return T.resolve_data_path(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # mặc định lấy từ phần CẤU HÌNH ở đầu file; chỉ cần truyền khi muốn ghi đè nhanh
    ap.add_argument("--data-path", "--data", dest="data", default=DATA_PATH)
    ap.add_argument("--no-load-all", dest="load_all", action="store_false", default=LOAD_ALL_FIRST)
    ap.add_argument("--load-to", default=LOAD_TO, help="file .duckdb để nạp dữ liệu ra đĩa thay vì RAM")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--limit", type=int, default=LIMIT, help="0 = chạy toàn bộ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--doc-names", default=None, help="danh sách doc_name (phân tách bởi dấu phẩy) hoặc file, mỗi dòng 1")
    ap.add_argument("--output-name", default=None, help="tên file JSON cuối (mặc định <levels>_references.json)")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=SERVED_NAME)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--no-verify-ssl", dest="verify_ssl", action="store_false", default=VERIFY_SSL)
    ap.add_argument("--extra-body", type=json.loads, default=EXTRA_BODY, help="JSON, ghi đè EXTRA_BODY")
    ap.add_argument("--concurrency", "--workers", dest="concurrency", type=int, default=WORKERS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=TIMEOUT)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--input-mode", default=LLM_INPUT, choices=["full", "condensed"])
    ap.add_argument("--overlap", type=int, default=1500)
    ap.add_argument("--structured", default=STRUCTURED, choices=LLM.MODES)
    ap.add_argument("--no-match", dest="no_match", action="store_true", default=not MATCH_DATASET)
    ap.add_argument("--no-citations-case", dest="use_cc", action="store_false", default=USE_CITATIONS_CASE)
    ap.add_argument("--no-hints", dest="hints", action="store_false", default=HINTS_TO_LLM)
    ap.add_argument("--no-union", dest="union", action="store_false", default=UNION_WITH_RULES)
    ap.add_argument("--index-dir", default=None, help="thư mục chứa doc_index.parquet (dùng chung với trace_case_chains)")
    ap.add_argument("--min-code-count", type=int, default=20)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--redo-errors", action="store_true", help="chạy lại các văn bản lần trước bị lỗi")
    ap.add_argument("--compile-only", action="store_true", help="chỉ dựng lại file JSON từ llm_records.jsonl")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    out_json = out / (args.output_name or f"{'_'.join(l.lower() for l in levels)}_references.json")
    rec_path = out / "llm_records.jsonl"
    data_path = args.data
    if not args.compile_only:  # kiểm tra server trước khi nạp dữ liệu (nạp có thể lâu)
        args.model = check_server(args.base_url, args.api_key, args.verify_ssl, args.model)
    con, args.data, cols_avail = T.open_data(data_path, args.load_all and not args.compile_only, args.load_to)
    meta_tbl = T.load_index(con, args.data, Path(args.index_dir) if args.index_dir else out, args.rebuild_index)
    codenorm = T.CodeNormalizer(T.learn_extra_code_chunks(meta_tbl, args.min_code_count))
    index = None if args.no_match else T.DocIndex(meta_tbl, codenorm)

    names = meta_tbl.column("doc_name").to_pylist()
    lv = meta_tbl.column("instance_level").to_pylist()
    if args.doc_names:
        p = Path(args.doc_names)
        wanted = [x.strip() for x in (p.read_text(encoding="utf-8").splitlines() if p.exists()
                                      else args.doc_names.split(",")) if x.strip()]
        roots = [d for d in wanted if d in set(names)]
    else:
        roots = [n for n, l in zip(names, lv) if l in levels]
    log.info("Văn bản cần đọc (%s): %d", ",".join(levels), len(roots))
    if args.limit and len(roots) > args.limit:
        random.Random(args.seed).shuffle(roots)
        roots = roots[: args.limit]
        log.info("  lấy mẫu %d văn bản", len(roots))

    model = args.model
    if not args.compile_only:
        model = asyncio.run(run(args, con, roots, cols_avail, codenorm, index, rec_path))
    stats = compile_json(rec_path, out_json, roots, dict(
        dataset="tmquan/cbba-toaan-gov-vn", data=str(data_path), levels=levels, model=model, base_url=args.base_url,
        temperature=args.temperature, dataset_matching=not args.no_match))
    log.info("Đã ghi %s\n%s", out_json, json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
