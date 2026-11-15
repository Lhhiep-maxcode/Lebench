# Legal Rule Retrieval Harness

End-to-end workflow for the supplied Neo4j JSONL exports:

```text
raw documents.jsonl + parts.jsonl
    → prepare → prepared/rules.jsonl
    → embed   → vector_index/index.faiss + mappings + manifest
    → run     → per-task Action → Observation trace + retrieval scores
```

There is **no SQLite database**. The prepared JSONL preserves the full provision text and document metadata. FAISS searches vector representations; the JSONL supplies exact rule IDs and full text. The agent sees facts, spotted issues, and tool descriptions, never `gold_rules`.

The interaction pattern follows the idea of [LegalAgentBench's ReAct agent](https://github.com/CSHaitao/LegalAgentBench/blob/main/agents.py) and [tool catalog](https://github.com/CSHaitao/LegalAgentBench/blob/main/src/generated_tools.py). Its [evaluation script](https://github.com/CSHaitao/LegalAgentBench/blob/main/src/evaluation/eval.py) scores answers and intermediate keywords; this harness scores exact provision IDs and records the tool trace.

## Install on the server

From the repository root:

```bash
python -m pip install -r requirements-legal-rule-retrieval.txt
```

For a local sentence-transformers embedding model instead of an embedding API, additionally install `sentence-transformers`.

## 1. Prepare the raw exports

The paths below can point to your **full exports**, not only the included 200-row samples:

```bash
python -m legal_rule_retrieval prepare \
  --documents /data/neo4j_documents.jsonl \
  --parts /data/neo4j_parts.jsonl \
  --output-dir /data/legal_rules/prepared
```

`prepared/rules.jsonl` has one provision per line. Each record contains the full `identifier` as `rule_id`, its immediate `parent_id`, resolved `document_id`, `name`, `part_type`, full `content`, and joined document metadata (`title`, code, field, issue/effective/expiration dates, validity status, URL). A part's parent can be another part; the preparer matches the document suffix of its full identifier. If a document is absent from the input export, its part stays searchable but document metadata is null. Re-run this phase when raw legal data changes.

## 2. Create embedding vectors

Use an OpenAI-compatible `/v1/embeddings` service with a **real embedding model**. It can run at a different URL from the chat model. vLLM [documents this endpoint for embedding models](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

```bash
export EMBEDDING_API_KEY=dummy
python -m legal_rule_retrieval embed \
  --rules-dir /data/legal_rules/prepared \
  --index-dir /data/legal_rules/vectors \
  --embedding-backend api \
  --embedding-base-url http://localhost:8001/v1 \
  --embedding-model YOUR_EMBEDDING_MODEL \
  --batch-size 32 --chunk-chars 1800 --overlap 200
```

Or embed locally with an installed sentence-transformers model:

```bash
python -m legal_rule_retrieval embed \
  --rules-dir /data/legal_rules/prepared \
  --index-dir /data/legal_rules/vectors \
  --embedding-backend local \
  --embedding-model /models/YOUR_EMBEDDING_MODEL
```

Long provisions are split into overlapping character chunks; each chunk receives an embedding. Search deduplicates chunk hits back to provision IDs and returns the matching chunk as a preview. `get_legal_rule` always returns the **full** provision text. The index stores normalized vectors and uses exact cosine similarity via FAISS `IndexFlatIP`. The vector files, chunk-to-rule mappings, and `manifest.json` are written under `--index-dir`. Re-embed after changing prepared rules, the embedding model, chunk settings, or the embedding model's prompt configuration. The manifest includes a SHA-256 hash of `rules.jsonl`, checked when running tasks to prevent index/data mismatch.

If your embedding model requires different prefixes for indexed passages and queries, set `--passage-prefix` and `--query-prefix` during `embed`. Both are stored in the manifest and reused at search time.

## 3. Run the agent and evaluate

```bash
export VLLM_API_KEY=dummy
python -m legal_rule_retrieval run \
  --rules-dir /data/legal_rules/prepared \
  --index-dir /data/legal_rules/vectors \
  --tasks /data/benchmark/tasks.jsonl \
  --output /data/benchmark/rule_results.jsonl \
  --workspace-root /data/benchmark/tool_files \
  --base-url http://localhost:8000/v1 \
  --model YOUR_CHAT_MODEL \
  --embedding-base-url http://localhost:8001/v1 \
  --max-steps 8
```

When the vector index was built with `--embedding-backend local`, the run command loads the same model from the index manifest and does not need `--embedding-base-url`. Optional chat settings: `--api-key-env`, `--timeout`, `--extra-body '{"max_tokens":1024}'`. The model uses one JSON action per turn over Chat Completions; native function calling is not required.

Each task line can be:

```json
{"id":"example-1","benchmark":{"facts":"Tình huống...","issues":["Vấn đề pháp lý..."],"gold_rules":["Điều 1, 173/2018/TT-BQP, 2018-12-31, Thông tư, Bộ Quốc phòng"]}}
```

Facts may instead be supplied by `benchmark_seed.facts_clean`; issues may be in `benchmark.spotted_issues`, `benchmark.gold_spotted_issues`, or `benchmark.gold_issues`. `gold_rules` can be omitted for inference without scoring. Gold IDs must be the full `identifier` values from the parts export, since `partId` is not globally unique. Keep gold **rule** IDs out of the visible facts/issues when making real benchmark tasks.

The tools are `search_legal_rules(query,top_k)`, `get_legal_rule(rule_id)`, `check_validity(rule_id,start,end)`, `submit_rules(rule_ids)`, `create_file(content,path)`, `read_file(path)`, and `add/subtract/multiply/divide(a,b)`. `submit_rules` accepts only IDs previously returned by search/get. Files are UTF-8, limited to 1 MB, and confined to a directory for that task under `--workspace-root`; `create_file` will not overwrite an existing file. Arithmetic uses decimal numbers and returns a decimal string. All calls and observations appear in the output trace.

`check_validity` uses **document-level** dates only. Missing expiration information is not proof of historical validity, and article amendments are unavailable in these exports. It reports `unknown` where the dates cannot establish the requested interval.

Per task, the evaluator reports Rule Recall, Precision, F1, retrieval calls, total/failed tool calls, and `tool_call_efficiency = F1 / max(1, work_calls)`. `work_calls` counts every actual tool call except the final `submit_rules`, including file, arithmetic, and validity calls. It also prints aggregate macro and micro scores. A task stops on `submit_rules` or `--max-steps`.

## Offline integration demo

```bash
python -m legal_rule_retrieval demo
```

This runs all three phases on the supplied sample exports and `legal_rule_retrieval/demo_tasks.jsonl`. It uses a deterministic word-hash vectorizer and a scripted agent to test plumbing **without any server**. The hash vectors are not semantic embeddings and demo F1 is not a model benchmark. Results go to `output/legal_rule_demo/`.

FAISS `IndexFlatIP` is exact and loads the vector index in memory. For a very large full corpus, replace it with a FAISS approximate index while keeping the prepared format, tool interface, and evaluation unchanged.
