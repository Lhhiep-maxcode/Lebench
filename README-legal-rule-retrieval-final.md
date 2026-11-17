# Legal rule retrieval with the final Neo4j parts export

This is the single-file workflow. It keeps the previous two-file workflow available for older benchmarks.

## Prepare the corpus

Install `requirements-legal-rule-retrieval-dsh.txt` first. Point the embedding API at a model hosted on your internal network:

```bash
python -m legal_rule_retrieval prepare-final \
  --parts /data/neo4j_parts_final.jsonl \
  --output-dir /data/legal_final \
  --embedding-backend api \
  --embedding-base-url http://localhost:8001/v1 \
  --embedding-model YOUR_EMBEDDING_MODEL \
  --batch-size 32
```

For an embedding model already downloaded to the server, use `--embedding-backend local --embedding-model /models/YOUR_MODEL` and omit the API URL. `--embedding-backend hash` is for plumbing tests only; it does not provide semantic search.

`/data/legal_final/rules.jsonl` contains every original field of every input record plus **only** `emb_vector`. The vector is calculated from `title` alone. Identical titles are embedded once and reused. The directory also contains `index.faiss` and `manifest.json`; the manifest records the embedding model and a hash of the prepared JSONL. Keep these files together and rerun preparation when the corpus or embedding model changes.

## Run the benchmark

The chat server must support OpenAI-compatible Chat Completions tool calls. Set both URLs to internal services:

```bash
python -m legal_rule_retrieval run-dsh \
  --final-dir /data/legal_final \
  --tasks /data/sample.jsonl \
  --output /data/results.jsonl \
  --conversation-output /data/conversations.jsonl \
  --base-url http://localhost:8000/v1 \
  --model YOUR_CHAT_MODEL \
  --embedding-base-url http://localhost:8001/v1 \
  --max-steps 12 --max-tokens 4096
```

For a local embedding backend, omit `--embedding-base-url`. The benchmark user prompt contains only `benchmark_seed.facts_clean` and `issues`. DeepSeek Harness supplies the registered MCP tools to the model in its native tool schema. The final action is `submit(identifiers=[...])`, using the exact `identifier` strings from the legal corpus. The earlier `submit_rules` tool remains available only in the older two-file workflow.

Each result line contains the document ID, selected identifiers, tool trace, precision, recall, F1, and tool-call counts. The optional conversation JSONL has one line per task with `messages` and `tools`. Messages use OpenAI roles (`system`, `user`, `assistant`, `tool`), assistant `tool_calls` and matching `tool_call_id`, and a `reasoning_content` field on every message. This conversation is reconstructed from DeepSeek Harness native events; a missing model reasoning block becomes an empty string. The `tools` array uses OpenAI function schema. The gold citations are never inserted into the model prompt or conversation.

Gold rules come from each `laws[*].provisions[*].citation_str`. The scorer collapses points and clauses into their parent article, and counts at most one prediction per article and document. A match requires the article number, document designation, and cited year to agree with the chosen corpus record. Each match contributes one true positive. The `submit` observation returns precision and recall immediately, with matched and unmatched IDs for debugging. Invalid or unknown IDs count as false positives.

`check_validity` accepts ISO `YYYY-MM-DD` strings in `effectiveDate` / `expirationDate` (or their snake_case equivalents). An absent expiration date is allowed. Re-run `prepare-final` after changing source dates. Search uses **only cosine similarity between the query embedding and each stored title embedding**. It does not inspect `content` or add an article-number boost. Provisions from the same document share a title vector and therefore tie in embedding score.

If a run reports `runtime_error`, the error is printed to stderr and stored in the result row's `error` field. Add `--debug` to include a Python traceback. The DeepSeek SDK's built-in initialization timeout is only 30 seconds; this runner now allows 300 seconds by default and gives its MCP process up to 270 seconds. Increase both through `--startup-timeout 600` if startup of your full corpus needs longer. A startup timeout occurs before any model request or legal tool call.

`--max-tokens` is the requested output limit per model call; it must be smaller than `--context-window` and supported by your chat server. The generated DeepSeek model profile now declares the same output limit. A large value such as `65536` may be rejected by a server with a smaller configured limit; the printed error will show whether that is happening.

The attached `sample.jsonl` has six valid rows and a seventh row cut off in the middle of a JSON string. The reader reports the bad line and stops. Re-export or replace that final row before running the full benchmark.
