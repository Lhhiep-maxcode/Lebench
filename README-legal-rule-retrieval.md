# Legal Rule Retrieval Harness

For the new **single-file** `neo4j_parts_final.jsonl` corpus and nested `laws[*].provisions` benchmark, use [README-legal-rule-retrieval-final.md](README-legal-rule-retrieval-final.md). The workflow below documents the older two-file corpus.

The old DeepSeek SDK instructions below are archived. The current `run-dsh` command is an alias for the direct Python `run-python` command and requires `--final-dir`; it does not start the SDK or MCP server. The older two-file `prepare`, `embed`, and `run` commands remain available.

End-to-end workflow for the supplied Neo4j JSONL exports:

```text
raw documents.jsonl + parts.jsonl
    → prepare → prepared/rules.jsonl
    → embed   → vector_index/index.faiss + mappings + manifest
    → run-dsh → DeepSeek Harness session + MCP tools → native event trace + scores
```

There is **no SQLite database**. The prepared JSONL preserves the full provision text and document metadata. FAISS searches vector representations; the JSONL supplies exact rule IDs and full text. The agent sees facts, spotted issues, and tool descriptions, never `gold_rules`.

The `run-dsh` path uses the cloned DeepSeek Harness's documented Python SDK, `sdk-minimal` profile, `llm-pi-ai` OpenAI Chat Completions route, MCP client plugin, and native `tool/call` / `tool/result` events. The clone is not edited. A generated per-task profile patch mounts our Python legal MCP server and disables the profile's shell, image, and workspace tools. Corpus preparation, embedding, tools, and scoring live in this repository, so they can be extended without changing the DeepSeek agent loop.

## Install on the server

From the repository root:

```bash
python -m pip install -r requirements-legal-rule-retrieval-dsh.txt
```

For a local sentence-transformers embedding model instead of an embedding API, additionally install `sentence-transformers`.

### Install without Internet on the internal server

Build the wheelhouse on a connected **Linux machine with the same Python version and CPU architecture** as the internal server, then copy the wheelhouse and this project to the server:

```bash
python -m pip download -r requirements-legal-rule-retrieval-dsh.txt -d wheelhouse
# On the internal server:
python -m pip install --no-index --find-links=/path/to/wheelhouse \
  -r requirements-legal-rule-retrieval-dsh.txt
```

For local sentence-transformers, add its wheels to the wheelhouse and copy the model directory too. `--embedding-backend local` requires an existing model directory and a sentence-transformers version supporting `local_files_only=True`; a Hub model ID is rejected. The DeepSeek SDK runtime wheel contains the `dsh` executable used by `run-dsh`; ordinary SDK startup does not download it.

Runtime network use is limited to the endpoint you supply: `embed --embedding-backend api` calls `--embedding-base-url`; `run-dsh` calls `--base-url` for chat and, for an API-built index, `--embedding-base-url` for queries. `prepare`, `demo`, and local embedding use no network. The Python HTTP clients ignore ambient proxies and refuse redirects. The DeepSeek child process has proxy environment variables cleared. Set both URLs to your internal services and enforce egress rules at the server/firewall if you need a hard guarantee that no other destination is reachable.

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

## 3. Run with DeepSeek Harness and evaluate

Set chat and embedding credentials in the environment. The chat endpoint must expose OpenAI-compatible `/v1/chat/completions` with tool calling; `llm-pi-ai` handles that protocol. The embedding endpoint is used by the Python MCP process. Use a virtual environment for the Python dependencies.

```bash
export VLLM_API_KEY=dummy
export EMBEDDING_API_KEY=dummy
python -m legal_rule_retrieval run-dsh \
  --rules-dir /data/legal_rules/prepared \
  --index-dir /data/legal_rules/vectors \
  --tasks /data/benchmark/tasks.jsonl \
  --output /data/benchmark/rule_results_dsh.jsonl \
  --workspace-root /data/benchmark/tool_files \
  --dsh-home /data/benchmark/dsh_home \
  --base-url http://localhost:8000/v1 \
  --model YOUR_CHAT_MODEL \
  --embedding-base-url http://localhost:8001/v1 \
  --max-steps 12 --max-tokens 4096 --context-window 131072
```

Use the same embedding model that created the index; `run-dsh` reads it from `manifest.json`. For a local embedding backend, omit `--embedding-base-url`. If you have built a matching `dsh` executable from the clone, pass `--dsh-bin /absolute/path/to/dsh`; otherwise the Python SDK uses its installed runtime. The clone in this workspace is `0.2.2-rc.3`; the available Python SDK wheel may be a different release, so use a matching build for strict runtime equivalence.

Each task gets a fresh DeepSeek Harness runtime, home, scratch workspace, and MCP server, so retrieved IDs and scratch files cannot leak into another task. The run writes one JSONL result per task, including the native tool-call trace, result content, submitted IDs, model finish reason, and metrics. DeepSeek Harness session logs remain in per-task directories under `--dsh-home`. On Linux, per-task directories are mode `0700` and benchmark result files are mode `0600`. `--max-steps` caps model steps; `--max-tokens` caps output tokens **per model request**; `--context-window` declares the chat model's context size to the adapter. A successful `submit_rules` writes a submission artifact before the final answer. If the model never submits, the scored prediction is empty. The evaluator does not parse IDs from free text.

On a runtime where native session persistence cannot load, use `--memory-sessions`. The per-task JSONL result still records tool calls and observations, but DeepSeek's internal session log is then kept only in memory during the task.

The generated profile patch exposes `mcp__legal_rules__search_legal_rules`, `get_legal_rule`, `check_validity`, `submit_rules`, `create_file`, `read_file`, and decimal `add/subtract/multiply/divide`. The MCP server code is in `legal_rule_retrieval/mcp_server.py`; the SDK and event adapter are in `legal_rule_retrieval/dsh_runner.py`. `submit_rules` accepts only exact IDs already returned by search or lookup and can succeed only once.

The DeepSeek profile disables shell, image, workspace, and the unused official DeepSeek model route. The runner rejects a model request if another tool or provider appears. File tools can access only the task's scratch directory; creation is limited to 10 files and 2 MB per task. Facts and retrieved rule text remain untrusted model input: a prompt injection may still spoil the selected IDs, but it cannot give the model a shell or a web tool through this profile. Run the benchmark under an ordinary OS account with access only to the intended corpus, index, output, and internal endpoints. Keep those directories and the installed Python/runtime packages trusted; this code does not sandbox native dependencies or prevent an operator from configuring an external URL.

### Older JSON-action runner

The earlier `run` command is retained for compatibility. It uses the custom one-JSON-action-per-turn loop rather than DeepSeek Harness. Use `run-dsh` for the new benchmark.

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

When the vector index was built with `--embedding-backend local`, the older `run` command loads the same model from the index manifest and does not need `--embedding-base-url`. Optional legacy chat settings: `--api-key-env`, `--timeout`, `--extra-body '{"max_tokens":1024}'`. The legacy model uses one JSON action per turn over Chat Completions.

Each task line can be:

```json
{"id":"example-1","benchmark":{"facts":"Tình huống...","issues":["Vấn đề pháp lý..."],"gold_rules":["Điều 1, 173/2018/TT-BQP, 2018-12-31, Thông tư, Bộ Quốc phòng"]}}
```

Facts may instead be supplied by `benchmark_seed.facts_clean`; issues may be in `benchmark.spotted_issues`, `benchmark.gold_spotted_issues`, or `benchmark.gold_issues`. `gold_rules` can be omitted for inference without scoring. Gold IDs must be the full `identifier` values from the parts export, since `partId` is not globally unique. Keep gold **rule** IDs out of the visible facts/issues when making real benchmark tasks.

The tools are `search_legal_rules(query,top_k)`, `get_legal_rule(rule_id)`, `check_validity(rule_id,start,end)`, `submit_rules(rule_ids)`, `create_file(content,path)`, `read_file(path)`, and `add/subtract/multiply/divide(a,b)`. `submit_rules` accepts only IDs previously returned by search/get. Files are UTF-8, limited to 1 MB, and confined to a directory for that task under `--workspace-root`; `create_file` will not overwrite an existing file. Arithmetic uses decimal numbers and returns a decimal string. All calls and observations appear in the output trace.

`check_validity` uses **document-level** dates only. Missing expiration information is not proof of historical validity, and article amendments are unavailable in these exports. It reports `unknown` where the dates cannot establish the requested interval.

Per task, the evaluator reports Rule Recall, Precision, F1, retrieval calls, total/failed tool calls, and `tool_call_efficiency = F1 / max(1, work_calls)`. `work_calls` counts every actual tool call except the final `submit_rules`, including file, arithmetic, and validity calls. It also prints aggregate macro and micro scores. The legacy `run` loop stops on `submit_rules`; DeepSeek Harness is instructed to stop after submission and the first successful submission is retained for scoring.

## Offline integration demo

```bash
python -m legal_rule_retrieval demo
```

This runs all three phases on the supplied sample exports and `legal_rule_retrieval/demo_tasks.jsonl`. It uses a deterministic word-hash vectorizer and a scripted agent to test plumbing **without any server**. The hash vectors are not semantic embeddings and demo F1 is not a model benchmark. Results go to `output/legal_rule_demo/`.

FAISS `IndexFlatIP` is exact and loads the vector index in memory. For a very large full corpus, replace it with a FAISS approximate index while keeping the prepared format, tool interface, and evaluation unchanged.
