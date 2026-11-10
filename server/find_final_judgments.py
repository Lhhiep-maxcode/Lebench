#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_final_judgments.py
=======================
Tìm BẢN ÁN CUỐI CÙNG của mỗi vụ trong toàn bộ dataset tmquan/cbba-toaan-gov-vn:

  1. Bản án SƠ THẨM không được bản án phúc thẩm / giám đốc thẩm / tái thẩm nào tham chiếu
     → giữ bản sơ thẩm                                              (final_type = first_instance_unreferenced)
  2. Sơ thẩm → Phúc thẩm, không có gì cao hơn tham chiếu bản phúc thẩm
     → giữ bản phúc thẩm                                            (final_type = appellate_final)
  3. Giám đốc thẩm / Tái thẩm có kết quả HỦY bản án sơ thẩm / phúc thẩm → KHÔNG xét đến
     (coi như không có văn bản đó). GĐT/TT không hủy (giữ nguyên, sửa, ...) thì là bản án cuối
     của chuỗi                                                      (final_type = cassation_final / retrial_final)

Cách làm
--------
* Đọc MỌI văn bản cấp Appellate / Cassation / Retrial bằng DeepSeek đã host (REF_SOURCE = "llm"):
  LLM trích các bản án được tham chiếu + kết quả xử lý từng bản án (hủy / giữ nguyên / sửa / đình chỉ)
  + kết quả chung của phần QUYẾT ĐỊNH. Để không bỏ sót, prompt có gợi ý từ regex + cột citations_case
  và kết quả được gộp với chúng (xem extract_refs_llm.py). Mỗi bản án được tham chiếu được khớp với
  dataset (số + ngày + tòa) → cạnh "văn bản cấp trên → bản án cấp dưới".
  - Ghi dần vào llm_records.jsonl → bị ngắt thì chạy lại là chạy tiếp.
  - Văn bản LLM lỗi → tự dùng regex cho văn bản đó.
  - Cấp không nằm trong LLM_LEVELS (hoặc REF_SOURCE = "rules") → dùng regex + citations_case,
    chạy song song nhiều process, lưu edges_rules.parquet.
* Văn bản GĐT/TT bị coi là "hủy" nếu phần QUYẾT ĐỊNH của nó hủy bất kỳ bản án cấp dưới nào.
* Bản án X là bản án cuối nếu không có văn bản cấp trên nào (đang được xét) tham chiếu X.

Chính sách với bản án bị GĐT/TT hủy (ANNULLED_POLICY):
  "ignore" (mặc định, đúng như yêu cầu): bỏ qua GĐT/TT có kết quả hủy, coi như không tồn tại →
           chuỗi ST → PT phía dưới vẫn giữ bản PT. Cột annulled_by ghi GĐT/TT đã hủy nó để lọc.
  "void"  : bản án bị hủy coi như không còn hiệu lực → loại luôn nó và các tham chiếu của nó.
           VD GĐT "hủy bản án phúc thẩm, giữ nguyên bản án sơ thẩm" → giữ bản SƠ THẨM;
           GĐT hủy cả PT lẫn ST → vụ đó không có bản án cuối (đã giao xét xử lại).

Cần trace_case_chains.py và extract_refs_llm.py nằm CÙNG THƯ MỤC.

Cách dùng
---------
    pip install duckdb pyarrow rapidfuzz openai
    1. Sửa phần CẤU HÌNH bên dưới (ít nhất DATA_PATH, BASE_URL, SERVED_NAME).
    2. python find_final_judgments.py
       hoặc ghi đè nhanh:
       python find_final_judgments.py --data-path /data/cbba.parquet --workers 16
       python find_final_judgments.py --policy void            # dùng lại kết quả đã có, chỉ đổi chính sách
       python find_final_judgments.py --ref-source rules       # không dùng LLM, chỉ regex (nhanh)
       python find_final_judgments.py --rescan                 # quét regex lại từ đầu
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ============================== CẤU HÌNH – SỬA Ở ĐÂY ==============================
DATA_PATH = "/workspace/storage-shared/nlp/hieplh8/research/VT-LegalBench/src_data/cbba-toaan-gov-vn"        # thư mục (lấy MỌI file .parquet bên trong) | 1 file .parquet | glob
LOAD_ALL_FIRST = True                           # nạp hết mọi file vào DuckDB trước rồi mới chạy
LOAD_TO = None                                  # None = nạp vào RAM | "/local/disk/cbba.duckdb" = nạp ra đĩa, lần sau dùng lại
OUT_DIR = "out_final_judgments3-hieplh8-test"                 # thư mục kết quả

# Tìm tham chiếu
REF_SOURCE = "llm"               # "llm" = DeepSeek đã host | "rules" = chỉ regex + citations_case
LLM_LEVELS = ["Retrial", "Cassation", "Appellate"]  # cấp đọc bằng LLM; cấp khác (nếu có) dùng regex
LLM_INPUT = "condensed"          # "condensed": đầu văn bản + đoạn có số hiệu + toàn bộ phần QUYẾT ĐỊNH (nhanh)
                                 # "full": toàn văn (chậm hơn nhiều)

# Server model đã host sẵn
BASE_URL = "https://deepseek-v41-flash-runai-data-process.runai-inference.cyberspace.vn/v1"
SERVED_NAME = "DeepSeek-V4.1-Flash-1017"
API_KEY = "EMPTY"
VERIFY_SSL = True                # False nếu server https dùng chứng chỉ tự ký
EXTRA_BODY = {"chat_template_kwargs": {"thinking": False}}  # None = không gửi thêm
LLM_WORKERS = 32                 # số request song song tới server
MAX_TOKENS = 8192
TIMEOUT = 900                    # giây / request

WORKERS = 8                      # số process quét regex song song
MIN_CONFIDENCE = "medium"        # độ tin cậy tối thiểu của 1 tham chiếu để tính là "có tham chiếu": low|medium|high
ANNULLED_POLICY = "ignore"       # "ignore" | "void" (xem giải thích ở đầu file)
IGNORE_APPEAL_TERMINATION = True # QĐ đình chỉ xét xử phúc thẩm → bản án sơ thẩm vẫn là bản án cuối
ONLY_JUDGMENTS = True            # sơ thẩm/phúc thẩm: chỉ giữ "Bản án", bỏ "Quyết định" (QĐST công nhận thỏa thuận...)
USE_CITATIONS_CASE = True        # dùng thêm cột citations_case

REUSE_EDGES = True               # đã quét regex (edges_rules.parquet) thì dùng lại
LIMIT = None                     # chỉ đọc N văn bản cấp trên (CHẠY THỬ; kết quả sẽ không đầy đủ)
# ==================================================================================

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import trace_case_chains as T  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("Cần file trace_case_chains.py nằm cùng thư mục với script này.")
try:
    import duckdb  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("Thiếu thư viện: pip install duckdb pyarrow rapidfuzz")

log = logging.getLogger("find_final_judgments")

TIER = {"low": 1, "medium": 2, "high": 3}
HIGH_LEVELS = ("Cassation", "Retrial")
CITING_LEVELS = ("Appellate", "Cassation", "Retrial")
FINAL_TYPE = {"First-instance": "first_instance_unreferenced", "Appellate": "appellate_final",
              "Cassation": "cassation_final", "Retrial": "retrial_final"}

EDGE_SCHEMA = pa.schema([
    ("source_doc", pa.string()), ("source_level", pa.string()), ("target_doc", pa.string()),
    ("target_level", pa.string()), ("status", pa.string()), ("confidence", pa.string()),
    ("direction", pa.string()), ("outcome", pa.string()), ("outcome_via", pa.string()),
    ("in_decision", pa.bool_()), ("case_number", pa.string()), ("cited_date", pa.string()),
    ("cited_court", pa.string()), ("sources", pa.string()), ("alt_docs", pa.string()), ("edge_source", pa.string()),
])
FLAG_SCHEMA = pa.schema([
    ("doc_name", pa.string()), ("instance_level", pa.string()), ("decision_annuls", pa.bool_()),
    ("decision_terminates_appeal", pa.bool_()), ("decision_upholds", pa.bool_()), ("n_refs", pa.int32()),
    ("decision_source", pa.string()),
])


# --------------------------------------------------------------------------------------
# Quét văn bản cấp trên (chạy trong process con)
# --------------------------------------------------------------------------------------
_G: dict = {}


def scan_one(row):
    idx, cn, use_cc = _G["index"], _G["codenorm"], _G["use_cc"]
    flat = T.normalize_text(row.get("markdown") or "")
    dfl = T.decision_flags(flat)
    flags = dict(doc_name=row["doc_name"], instance_level=row.get("instance_level"), decision_annuls=dfl["annuls"],
                 decision_terminates_appeal=dfl["terminates"], decision_upholds=dfl["upholds"], n_refs=0,
                 decision_source="rules")
    try:
        refs = T.process_doc(row, idx, cn, 1, use_cc)
    except Exception:  # noqa: BLE001 — 1 văn bản lỗi không làm hỏng cả lượt quét
        flags["n_refs"] = -1
        return [], flags
    edges = []
    for r in refs:
        alt = None
        if r["status"] == "ambiguous" and r.get("alt_candidates"):
            alt = ",".join(a["doc"] for a in json.loads(r["alt_candidates"]) if a.get("tier") != "rejected")
        edges.append(dict(
            source_doc=r["source_doc"], source_level=r["source_level"], target_doc=r["match_doc"],
            target_level=r["match_level"] or r["cited_level"], status=r["status"], confidence=r["confidence"],
            direction=r["direction"], outcome=r["outcome"], outcome_via=r.get("outcome_via"),
            in_decision=bool(r["in_decision"]), case_number=f"{r['number']}/{r['year']}/{r['code']}",
            cited_date=r["cited_date"], cited_court=r["cited_court"], sources=r.get("sources"), alt_docs=alt,
            edge_source="rules"))
    flags["n_refs"] = len(refs)
    return edges, flags


def scan(con, data, names, index, codenorm, use_cc, out: Path, workers: int, batch: int, tag: str = "rules"):
    _G.update(index=index, codenorm=codenorm, use_cc=use_cc)
    e_tmp, f_tmp = out / f"edges_{tag}.parquet.tmp", out / f"flags_{tag}.parquet.tmp"
    ew, fw = pq.ParquetWriter(e_tmp, EDGE_SCHEMA), pq.ParquetWriter(f_tmp, FLAG_SCHEMA)
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None  # fork TRƯỚC khi mở truy vấn DuckDB
    t0, n, n_edges = time.time(), 0, 0
    try:
        buf = []

        def flush(rows):
            nonlocal n, n_edges
            if pool:
                res = pool.map(scan_one, rows, chunksize=max(1, len(rows) // (workers * 4)))
            else:
                res = [scan_one(r) for r in rows]
            edges = [e for es, _ in res for e in es]
            if edges:
                ew.write_table(pa.Table.from_pylist(edges, schema=EDGE_SCHEMA))
            fw.write_table(pa.Table.from_pylist([f for _, f in res], schema=FLAG_SCHEMA))
            n += len(rows)
            n_edges += len(edges)
            el = time.time() - t0
            log.info("  quét %d/%d văn bản | %d tham chiếu | %.0f văn bản/s | còn ~%.0f phút", n, len(names), n_edges,
                     n / el, (len(names) - n) / max(n / el, 1e-9) / 60)

        for row in T.iter_docs(con, data, set(names), batch, use_cc):
            buf.append(row)
            if len(buf) >= batch:
                flush(buf)
                buf = []
        if buf:
            flush(buf)
    finally:
        if pool:
            pool.close()
            pool.join()
        ew.close()
        fw.close()
    os.replace(e_tmp, out / f"edges_{tag}.parquet")
    os.replace(f_tmp, out / f"flags_{tag}.parquet")


# --------------------------------------------------------------------------------------
# Đọc bằng LLM (extract_refs_llm.py) → cạnh
# --------------------------------------------------------------------------------------
_LLM_OUTCOME = {"annulled": "huy", "partially_annulled": "huy", "upheld": "giu_nguyen", "amended": "sua",
                "terminated": "dinh_chi"}


def run_llm(args, con, docs, cols, codenorm, index, rec_path: Path):
    import asyncio
    import extract_refs_llm as E
    ns = argparse.Namespace(
        base_url=args.base_url, api_key=args.api_key, model=args.model, verify_ssl=args.verify_ssl,
        extra_body=args.extra_body, structured=E.STRUCTURED, temperature=0.0, max_tokens=args.max_tokens,
        timeout=args.timeout, retries=4, concurrency=args.llm_workers, max_chars=E.MAX_CHARS, overlap=1500,
        input_mode=args.llm_input, use_cc=args.use_cc, hints=True, union=True, no_match=False,
        redo_errors=True, data=args.data, batch_size=256)
    return asyncio.run(E.run(ns, con, docs, cols, codenorm, index, rec_path))


def llm_records_to_edges(rec_path: Path, index, wanted: set):
    """Đọc llm_records.jsonl (dạng stream) → (cạnh, cờ quyết định, danh sách văn bản lỗi)."""
    level = dict(zip(index.doc_name, index.level))
    by_doc, flags, status = {}, {}, {}
    with open(rec_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = r.get("doc_name")
            if doc not in wanted:
                continue
            if r.get("status") != "ok":
                status.setdefault(doc, "error")
                continue
            status[doc] = "ok"
            sl = level.get(doc) or r.get("instance_level")
            rs = T.LEVEL_RANK.get(sl)
            edges = []
            for ref in r.get("referenced_judgments", []):
                m = ref.get("dataset_match") or {}
                st = m.get("status") or "not_found"
                if not ref.get("verified"):
                    st = "unverified"  # số hiệu không có trong văn bản → không dùng
                tl = m.get("instance_level") or (ref.get("instance_level") if ref.get("instance_level") in T.LEVEL_RANK
                                                 else None)
                rt = T.LEVEL_RANK.get(tl)
                alt = None
                if st == "ambiguous" and m.get("candidates"):
                    alt = ",".join(c["doc"] for c in m["candidates"] if c.get("tier") != "rejected")
                via = "rules" if (ref.get("llm_missed") or "outcome" in (ref.get("filled_from_rules") or [])) else "llm"
                edges.append(dict(
                    source_doc=doc, source_level=sl, target_doc=m.get("doc_name"), target_level=tl, status=st,
                    confidence=m.get("confidence"),
                    direction=None if not (rs and rt) else "lower" if rt < rs else "same" if rt == rs else "higher",
                    outcome=_LLM_OUTCOME.get(ref.get("outcome")), outcome_via=via,
                    in_decision=bool(ref.get("in_decision_section")), case_number=ref.get("case_number"),
                    cited_date=ref.get("issued_date"), cited_court=ref.get("court"),
                    sources="+".join(ref.get("sources") or []), alt_docs=alt, edge_source="llm"))
            by_doc[doc] = edges
            d, dr = r.get("decision"), r.get("decision_rules") or {}
            if d:  # LLM đọc được phần QUYẾT ĐỊNH → tin LLM
                flags[doc] = dict(doc_name=doc, instance_level=sl, decision_annuls=bool(d["annuls_lower_judgment"]),
                                  decision_terminates_appeal=bool(d["terminates_proceeding"]),
                                  decision_upholds=bool(d["upholds_lower_judgment"]), n_refs=len(edges),
                                  decision_source="llm")
            else:
                flags[doc] = dict(doc_name=doc, instance_level=sl, decision_annuls=bool(dr.get("annuls")),
                                  decision_terminates_appeal=bool(dr.get("terminates")),
                                  decision_upholds=bool(dr.get("upholds")), n_refs=len(edges), decision_source="rules")
    edges = [e for es in by_doc.values() for e in es]
    errors = [d for d, s_ in status.items() if s_ != "ok"]
    missing = [d for d in wanted if d not in status]
    return edges, flags, errors + missing


# --------------------------------------------------------------------------------------
# Quyết định bản án cuối
# --------------------------------------------------------------------------------------
def decide(index, edges: list, flags: dict, cfg):
    min_t = TIER[cfg.min_confidence]
    level = dict(zip(index.doc_name, index.level))

    good, weak = [], defaultdict(set)
    for e in edges:
        if e["direction"] != "lower":
            continue
        if e["status"] == "matched" and e["target_doc"]:
            if TIER.get(e["confidence"], 0) >= min_t:
                good.append(e)
            else:
                weak[e["target_doc"]].add(e["source_doc"])
        elif e["status"] == "ambiguous" and e.get("alt_docs"):
            for d in e["alt_docs"].split(","):
                weak[d].add(e["source_doc"])
    out_edges = defaultdict(list)
    for e in good:
        out_edges[e["source_doc"]].append(e)

    # 1) GĐT/TT có kết quả hủy → không xét đến
    def annulling(src):
        if level.get(src) not in HIGH_LEVELS:
            return False
        f = flags.get(src, {})
        if f.get("decision_source") == "llm":  # LLM đọc phần QUYẾT ĐỊNH: tin kết quả chung + kết quả từng bản án của LLM
            return bool(f.get("decision_annuls")) or any(e["outcome"] == "huy" and e.get("outcome_via") == "llm"
                                                         for e in out_edges[src])
        return any(e["outcome"] == "huy" for e in out_edges[src]) or bool(f.get("decision_annuls"))

    ignored_src = {s for s in set(out_edges) | set(flags) if annulling(s)}

    # bản án nào bị chúng hủy
    annulled_by = defaultdict(list)
    for s in ignored_src:
        es = out_edges.get(s, [])
        tg = [e for e in es if e["outcome"] == "huy"]
        if not tg and es:  # chỉ biết "có hủy" ở cấp văn bản → giả định hủy bản án cấp cao nhất mà nó nêu
            top = max(T.LEVEL_RANK.get(e["target_level"], 0) for e in es)
            tg = [e for e in es if T.LEVEL_RANK.get(e["target_level"], 0) == top]
        for e in tg:
            annulled_by[e["target_doc"]].append(s)
    void = set(annulled_by) if cfg.policy == "void" else set()

    # 2) cạnh được xét
    def appeal_terminated(e):
        if not cfg.ignore_appeal_termination or level.get(e["source_doc"]) != "Appellate":
            return False
        if e["outcome"] == "dinh_chi":
            return True
        f = flags.get(e["source_doc"], {})
        return bool(f.get("decision_terminates_appeal")) and e["outcome"] in (None, "dinh_chi")

    citing, ignored_citing = defaultdict(list), defaultdict(list)
    considered_out = defaultdict(list)
    for e in good:
        s, t = e["source_doc"], e["target_doc"]
        if s in ignored_src:
            ignored_citing[t].append((s, f"annulling_{level.get(s)}"))
        elif s in void:
            ignored_citing[t].append((s, "source_annulled"))
        elif appeal_terminated(e):
            ignored_citing[t].append((s, "appeal_terminated"))
        else:
            citing[t].append(s)
            considered_out[s].append(e)

    meta = index.meta

    def lower_chain(doc):
        """Các bản án cấp dưới (theo cạnh được xét) của 1 bản án cuối."""
        by_level, seen, stack, min_conf = defaultdict(list), {doc}, [doc], 3
        outcome_on = {}
        while stack:
            d = stack.pop()
            for e in considered_out.get(d, []):
                t = e["target_doc"]
                min_conf = min(min_conf, TIER.get(e["confidence"], 0))
                if d == doc:
                    outcome_on[t] = e["outcome"]
                if t in seen or t in void:
                    continue
                seen.add(t)
                by_level[level.get(t)].append(t)
                stack.append(t)
        return by_level, outcome_on, min_conf

    finals, excluded = [], []
    stats = Counter()
    for i, doc in enumerate(index.doc_name):
        lvl = index.level[i]
        if lvl not in T.LEVEL_RANK:
            stats["skip_unknown_level"] += 1
            continue
        stats[f"total_{lvl}"] += 1
        if doc in ignored_src:
            excluded.append(dict(doc_name=doc, official_id=index.official_id[i], level=lvl,
                                 reason="annulling_higher", detail=",".join(e["target_doc"] for e in out_edges.get(doc, [])
                                                                             if e["outcome"] == "huy")))
            stats[f"excluded_annulling_{lvl}"] += 1
            continue
        if doc in void:
            excluded.append(dict(doc_name=doc, official_id=index.official_id[i], level=lvl, reason="annulled_by_higher",
                                 detail=",".join(annulled_by[doc])))
            stats[f"excluded_annulled_{lvl}"] += 1
            continue
        if citing.get(doc):
            stats[f"superseded_{lvl}"] += 1  # nằm trong chuỗi của bản án cao hơn
            continue
        atoms = index.key_of[i][2] if index.key_of[i] else ()
        if cfg.only_judgments and lvl in ("First-instance", "Appellate") and "QĐ" in atoms:
            stats[f"excluded_decision_{lvl}"] += 1
            excluded.append(dict(doc_name=doc, official_id=index.official_id[i], level=lvl,
                                 reason="decision_not_judgment", detail=None))
            continue
        by_level, outcome_on, min_conf = lower_chain(doc)
        st_docs, pt_docs, gdt_docs = by_level.get("First-instance", []), by_level.get("Appellate", []), \
            by_level.get("Cassation", [])

        def ids(ds):
            return " | ".join(str(index.official_id[index.by_name[x]]) for x in ds)

        chain = [x for x in (ids(st_docs), ids(pt_docs), ids(gdt_docs)) if x] + [str(index.official_id[i])]
        ig = ignored_citing.get(doc, [])
        m = meta(i)
        row = dict(
            final_doc=doc, final_id=m["official_id"], final_level=lvl, final_type=FINAL_TYPE[lvl],
            final_court=m["court"], final_date=m["date"], final_category=m["category"],
            st_docs=",".join(st_docs) or None, st_ids=ids(st_docs) or None,
            pt_docs=",".join(pt_docs) or None, pt_ids=ids(pt_docs) or None,
            gdt_docs=",".join(gdt_docs) or None, gdt_ids=ids(gdt_docs) or None,
            chain=" → ".join(chain), n_lower_in_dataset=len(st_docs) + len(pt_docs) + len(gdt_docs),
            outcome_on_lower=",".join(f"{index.official_id[index.by_name[t]]}:{o}" for t, o in outcome_on.items() if o)
            or None,
            lower_annulled_by_final=any(o == "huy" for o in outcome_on.values()),
            annulled_by_ignored_higher=",".join(annulled_by.get(doc, [])) or None,
            ignored_citing=",".join(f"{s}:{r}" for s, r in ig) or None,
            possibly_referenced_by=",".join(sorted(weak.get(doc, set()) - set(citing.get(doc, [])) - ignored_src
                                                   - void)) or None,
            min_edge_confidence={3: "high", 2: "medium", 1: "low", 0: None}.get(min_conf) if by_level else None,
        )
        finals.append(row)
        stats[f"final_{FINAL_TYPE[lvl]}"] += 1
        if row["possibly_referenced_by"]:
            stats[f"final_uncertain_{FINAL_TYPE[lvl]}"] += 1
        if row["annulled_by_ignored_higher"]:
            stats[f"final_annulled_by_ignored_higher_{FINAL_TYPE[lvl]}"] += 1
        if lvl == "Appellate" and not st_docs:
            stats["appellate_final_without_st_in_dataset"] += 1
    return finals, excluded, dict(sorted(stats.items()))


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", "--data", dest="data", default=DATA_PATH)
    ap.add_argument("--no-load-all", dest="load_all", action="store_false", default=LOAD_ALL_FIRST,
                    help="không nạp trước, đọc thẳng từ file parquet")
    ap.add_argument("--load-to", default=LOAD_TO, help="file .duckdb để nạp dữ liệu ra đĩa thay vì RAM")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--ref-source", default=REF_SOURCE, choices=["llm", "rules"])
    ap.add_argument("--llm-levels", default=",".join(LLM_LEVELS))
    ap.add_argument("--llm-input", default=LLM_INPUT, choices=["condensed", "full"])
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=SERVED_NAME)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--no-verify-ssl", dest="verify_ssl", action="store_false", default=VERIFY_SSL)
    ap.add_argument("--extra-body", type=json.loads, default=EXTRA_BODY)
    ap.add_argument("--llm-workers", type=int, default=LLM_WORKERS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--timeout", type=float, default=TIMEOUT)
    ap.add_argument("--workers", type=int, default=WORKERS, help="số process quét regex")
    ap.add_argument("--min-confidence", default=MIN_CONFIDENCE, choices=list(TIER))
    ap.add_argument("--policy", default=ANNULLED_POLICY, choices=["ignore", "void"])
    ap.add_argument("--keep-appeal-termination", dest="ignore_appeal_termination", action="store_false",
                    default=IGNORE_APPEAL_TERMINATION)
    ap.add_argument("--include-decisions", dest="only_judgments", action="store_false", default=ONLY_JUDGMENTS)
    ap.add_argument("--no-citations-case", dest="use_cc", action="store_false", default=USE_CITATIONS_CASE)
    ap.add_argument("--rescan", dest="reuse_edges", action="store_false", default=REUSE_EDGES)
    ap.add_argument("--limit", type=int, default=LIMIT, help="chỉ đọc N văn bản cấp trên (chạy thử)")
    ap.add_argument("--batch-size", type=int, default=4000)
    ap.add_argument("--min-code-count", type=int, default=20)
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = args.data
    if args.ref_source == "llm":  # kiểm tra server trước khi nạp dữ liệu (nạp có thể lâu)
        import extract_refs_llm as E
        args.model = E.check_server(args.base_url, args.api_key, args.verify_ssl, args.model)
    con, args.data, cols = T.open_data(data_path, args.load_all, args.load_to)
    args.use_cc = use_cc = args.use_cc and "citations_case" in cols

    meta_tbl = T.load_index(con, args.data, out, args.rebuild_index)
    codenorm = T.CodeNormalizer(T.learn_extra_code_chunks(meta_tbl, args.min_code_count))
    index = T.DocIndex(meta_tbl, codenorm)
    lv_count = Counter(index.level)
    log.info("Số văn bản theo cấp: %s", dict(lv_count))

    citing = [d for d, l in zip(index.doc_name, index.level) if l in CITING_LEVELS]
    if args.limit:
        log.warning("--limit %d: chỉ đọc một phần → kết quả KHÔNG đầy đủ, chỉ để chạy thử", args.limit)
        citing = citing[: args.limit]
    llm_levels = {x.strip() for x in args.llm_levels.split(",") if x.strip()} if args.ref_source == "llm" else set()
    lvl_of = dict(zip(index.doc_name, index.level))
    llm_docs = [d for d in citing if lvl_of[d] in llm_levels]
    rule_docs = [d for d in citing if lvl_of[d] not in llm_levels]
    log.info("Đọc bằng LLM: %d văn bản %s | bằng regex: %d văn bản", len(llm_docs),
             dict(Counter(lvl_of[d] for d in llm_docs)), len(rule_docs))

    edges, flags = [], {}

    # ---- 1) LLM ----
    llm_errors = []
    if llm_docs:
        import extract_refs_llm as E
        rec_path = out / "llm_records.jsonl"
        log.info("Gọi %s (%d request song song, đầu vào %s); kết quả ghi dần vào %s",
                 args.model, args.llm_workers, args.llm_input, rec_path)
        run_llm(args, con, llm_docs, cols, codenorm, index, rec_path)
        e_llm, f_llm, llm_errors = llm_records_to_edges(rec_path, index, set(llm_docs))
        edges += e_llm
        flags.update(f_llm)
        log.info("LLM: %d văn bản ok, %d lỗi/thiếu → dùng regex cho các văn bản đó", len(f_llm), len(llm_errors))
        if len(llm_docs) <= 100_000:
            E.compile_json(rec_path, out / "llm_references.json", llm_docs, dict(
                dataset="tmquan/cbba-toaan-gov-vn", data=str(data_path), levels=sorted(llm_levels), model=args.model,
                base_url=args.base_url, input_mode=args.llm_input, dataset_matching=True))

    # ---- 2) regex: cấp không dùng LLM (có cache) + văn bản LLM lỗi ----
    if rule_docs:
        meta_path = out / "edges_rules.meta.json"
        want_meta = dict(n_docs=len(rule_docs), levels=sorted({lvl_of[d] for d in rule_docs}), limit=args.limit,
                         use_cc=use_cc, data=T.data_signature(args.data))
        cached = (out / "edges_rules.parquet").exists() and (out / "flags_rules.parquet").exists() and \
            meta_path.exists() and json.loads(meta_path.read_text()) == want_meta
        if args.reuse_edges and cached:
            log.info("Dùng lại kết quả regex đã quét (thêm --rescan để quét lại)")
        else:
            log.info("Quét regex %d văn bản với %d process...", len(rule_docs), args.workers)
            scan(con, args.data, rule_docs, index, codenorm, use_cc, out, max(1, args.workers), args.batch_size, "rules")
            meta_path.write_text(json.dumps(want_meta))
        edges += pq.read_table(out / "edges_rules.parquet").to_pylist()
        flags.update({f["doc_name"]: f for f in pq.read_table(out / "flags_rules.parquet").to_pylist()})
    if llm_errors:
        scan(con, args.data, llm_errors, index, codenorm, use_cc, out, max(1, args.workers), args.batch_size,
             "llm_fallback")
        edges += pq.read_table(out / "edges_llm_fallback.parquet").to_pylist()
        flags.update({f["doc_name"]: f for f in pq.read_table(out / "flags_llm_fallback.parquet").to_pylist()})

    pq.write_table(pa.Table.from_pylist(edges, schema=EDGE_SCHEMA), out / "edges_all.parquet")

    # ---- 3) quyết định bản án cuối ----
    finals, excluded, stats = decide(index, edges, flags, args)
    if finals:
        pq.write_table(pa.Table.from_pylist(finals), out / "final_judgments.parquet")
        T._write_csv(out / "final_judgments.csv", finals, list(finals[0].keys()))
        with open(out / "final_judgments.jsonl", "w", encoding="utf-8") as f:
            for r in finals:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    T._write_csv(out / "excluded.csv", excluded, ["doc_name", "official_id", "level", "reason", "detail"])

    summary = dict(
        data=str(data_path), data_files=len(getattr(args.data, "files", []) or []), ref_source=args.ref_source, llm_levels=sorted(llm_levels), llm_input=args.llm_input,
        model=args.model if llm_docs else None, policy=args.policy, min_confidence=args.min_confidence,
        ignore_appeal_termination=args.ignore_appeal_termination, only_judgments=args.only_judgments,
        used_citations_case=use_cc, docs_by_level=dict(lv_count),
        docs_read_by_llm=len(llm_docs) - len(llm_errors), docs_read_by_rules=len(rule_docs) + len(llm_errors),
        llm_errors_fallback_to_rules=len(llm_errors),
        decision_source=dict(Counter(f.get("decision_source") for f in flags.values())),
        edges_total=len(edges), edges_by_source=dict(Counter(e["edge_source"] for e in edges)),
        edges_status=dict(Counter(e["status"] for e in edges)),
        edges_to_first_instance_status=dict(Counter(e["status"] for e in edges if e["target_level"] == "First-instance")),
        finals_by_type=dict(Counter(r["final_type"] for r in finals)),
        excluded_by_reason=dict(Counter(r["reason"] for r in excluded)),
        details=stats,
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Đã ghi %s/final_judgments.{parquet,csv,jsonl}, excluded.csv, summary.json\n%s", out,
             json.dumps({k: summary[k] for k in ("finals_by_type", "excluded_by_reason")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
