# Legal rule retrieval benchmark

The agent's rule recall instructions are in
[`legal_rule_retrieval/skills/skill_rule_recall.md`](legal_rule_retrieval/skills/skill_rule_recall.md).

The benchmark reads `input/rule_recall.jsonl` and a **local snapshot** of
[`Hiepppp/vietnamese-legal-corpus`](https://huggingface.co/datasets/Hiepppp/vietnamese-legal-corpus).
It does not download data during a run. The corpus contains 170,824 documents,
their original fields in compressed JSONL shards, title vectors in `index.faiss`,
and the row order in `id_map.jsonl`.

## 1. Download the dataset once

If `input/vietnamese-legal-corpus` already contains `manifest.json`,
`index.faiss`, `id_map.jsonl`, and `data/train-*.jsonl.gz`, skip this step.
Otherwise, run:

```bash
python3 -m pip install huggingface_hub
hf download Hiepppp/vietnamese-legal-corpus --repo-type dataset \
  --local-dir input/vietnamese-legal-corpus
```

The runner checks for these files and prints this download command if any are
missing. The FAISS index is about 2.8 GB; allow additional RAM for loading it.
Install the benchmark dependencies with
`python3 -m pip install -r requirements-legal-rule-retrieval.txt`.

## 2. Run

Host the **same embedding model** used to build the corpus,
`nvidia/Nemotron-3-Embed-8B-BF16`, at an OpenAI-compatible `/v1/embeddings`
endpoint. Host a chat model with OpenAI-compatible tool calls separately.

```bash
python3 -u -m legal_rule_retrieval run-dsh \
  --corpus-dir input/vietnamese-legal-corpus \
  --tasks input/rule_recall.jsonl \
  --output output/rule_recall_results.jsonl \
  --conversation-output output/rule_recall_conversations.jsonl \
  --embedding-base-url http://127.0.0.1:8001/v1 \
  --base-url http://127.0.0.1:8000/v1 \
  --model YOUR_CHAT_MODEL \
  --max-steps 24 --max-tokens 4096 --verbosity 1
```

`run-python` is an alias for `run-dsh`. The default corpus and tasks paths are
the two paths above, so those flags may be omitted. The benchmark requires no
Hugging Face connection after the snapshot is downloaded. It loads the FAISS
index once and sends only `sample["input"]` to the model; `expected_answer` is
kept private for scoring.

The tool descriptions are defined once in `TOOL_SPECS` in
`legal_rule_retrieval/harness.py`. `python_runner.py` converts them to native
OpenAI function schemas; `mcp_server.py` uses the same descriptions.

`--verbosity 0` prints completed tasks; `1` shows each chat and tool step;
`2` also prints LLM text, arguments, and observations. Use `--debug` to include
chat error tracebacks. The optional conversation output is one JSONL line per
task with OpenAI-format `messages` and `tools`.

## Tools and scoring

- `search_legal_rules(query, top_k)` embeds `query: ` plus the query, searches
  the stored title vectors by cosine similarity, and returns **only `id` and
  `title`** for each hit.
- `get_legal_rule(rule_id)` returns the document's `full_content`.
- `check_validity(rule_id, start, end)` accepts ISO `YYYY-MM-DD` query dates
  and reads source dates from `ngay_co_hieu_luc` and `ngay_het_hieu_luc`
  (normally `DD/MM/YYYY`). The expiration day is treated as the first invalid
  day; a missing date may produce `unknown`.
- `check_relationship(rule_id)` returns the source `relationships` list.
- `create_file`, `read_file`, `append_file`, and the four arithmetic tools are
  available inside each task's private workspace.
- `submit_rules(rule_ids)` ends the task. Each string should be an article
  citation such as `Điều 114 Luật Tố tụng hành chính 2010`. The year is the
  **citation year used in the benchmark label**, which need not be the first
  effective year in `ngay_co_hieu_luc`.

Scoring compares article number, normalized document name, and citation year.
Points and clauses are collapsed to their parent article. Duplicate article
citations count once. Gold entries missing an article, document, or year are
ignored; if a task has no complete gold citations, it is marked unscored.
Invalid submitted citations count as false positives. Results report precision,
recall, F1, and tool-call counts.

The benchmark runs entirely through this repository's Python tool loop; it
does not install the DeepSeek Harness SDK or require its MCP runtime.
