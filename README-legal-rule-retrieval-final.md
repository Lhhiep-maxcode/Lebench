# Legal rule retrieval with the final Neo4j parts export

This is the single-file workflow. It keeps the previous two-file workflow available for older benchmarks.

## Prepare the corpus

Install `requirements-legal-rule-retrieval.txt`. The benchmark runner uses only this repository's Python code; it does not import the pip-installed DeepSeek Harness SDK or start an MCP subprocess. Point the embedding API at a model hosted on your internal network:

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
python -m legal_rule_retrieval run-python \
  --final-dir /data/legal_final \
  --tasks /data/sample.jsonl \
  --output /data/results.jsonl \
  --conversation-output /data/conversations.jsonl \
  --base-url http://localhost:8000/v1 \
  --model YOUR_CHAT_MODEL \
  --embedding-base-url http://localhost:8001/v1 \
  --max-steps 12 --max-tokens 4096
```

`run-dsh` remains an alias for `run-python`, so your existing command still works after updating the code. For a local embedding backend, omit `--embedding-base-url`. The user prompt contains only `benchmark_seed.facts_clean` and `issues`; the ten OpenAI function tools are sent in the request's `tools` field. The final action is `submit(identifiers=[...])`, using exact `identifier` strings from the corpus. The runner follows the DeepSeek Harness step → tool call → tool result pattern directly in Python and loads the legal index once per benchmark run.

For a quick debug run, add `--max-rules 10000`. This loads only the **first** 10,000 parts from `rules.jsonl` and builds a temporary FAISS index from their saved `emb_vector` values. It avoids reading the rest of the JSONL or loading the full FAISS index; no corpus rebuild is needed. This mode skips the full-file checksum, and retrieval cannot find rules outside the subset, so its benchmark scores are not representative. Omit the flag for the full run. A limit greater than or equal to the corpus size loads and verifies the full corpus normally.

At startup, the runner prints the resolved endpoints. Chat requests go to `--base-url` plus `/chat/completions`; query embeddings go to `--embedding-base-url` plus `/embeddings`. A dedicated embedding server is not expected to support `/chat/completions`. With `--debug`, the runner also prints its Python source path so you can confirm which installed checkout is running.

Each result line contains the document ID, selected identifiers, tool trace, precision, recall, F1, and tool-call counts. The optional conversation JSONL has one line per task with `messages` and `tools`. Messages use OpenAI roles (`system`, `user`, `assistant`, `tool`), assistant `tool_calls` and matching `tool_call_id`, and a `reasoning_content` field on every message. A missing model reasoning block becomes an empty string. The gold citations are never inserted into the model prompt or conversation.

Gold rules come from each `laws[*].provisions[*].citation_str`. The scorer collapses points and clauses into their parent article, and counts at most one prediction per article and document. A match requires the article number, document designation, and cited year to agree with the chosen corpus record. Each match contributes one true positive. The `submit` observation returns precision and recall immediately, with matched and unmatched IDs for debugging. Invalid or unknown IDs count as false positives.

`check_validity` accepts ISO `YYYY-MM-DD` strings in `effectiveDate` / `expirationDate` (or their snake_case equivalents). An absent expiration date is allowed. Re-run `prepare-final` after changing source dates. Search uses **only cosine similarity between the query embedding and each stored title embedding**. It does not inspect `content` or add an article-number boost. Provisions from the same document share a title vector and therefore tie in embedding score.

If a chat request fails, the task receives `llm_error`; the message is printed to stderr and stored in the result row's `error` field. Set `--verbosity 0` (default) for completed-task results only, `--verbosity 1` for live request/tool progress and elapsed times, or `--verbosity 2` to also print each LLM response (including `reasoning_content`), tool arguments, and tool observations as `TRACE` JSON lines. These lines are flushed immediately. `--debug` adds a traceback for chat failures and implies at least verbosity 1; use `--debug --verbosity 2` for both. Level 2 can print full case facts or legal text returned by tools, so keep the log private. `--request-timeout` controls the wait for each chat or embedding HTTP request (default 600 seconds). The older `--startup-timeout` flag is accepted as an alias; there is no separate runtime startup in this mode.

To watch and save the live log on Linux, add `-u` to Python and pipe both output streams to `tee`:

```bash
set -o pipefail
python3 -u -m legal_rule_retrieval run-python [your existing options] --verbosity 2 --debug 2>&1 | tee benchmark.log
```

Replace `[your existing options]` with the actual arguments from your command. In another terminal, `tail -f benchmark.log` follows the same log. Each chat request is logged when it starts; its response and subsequent tool interactions appear as they finish. Chat Completions currently uses `stream=false`, so model tokens are not shown while a request is still running.

`--max-tokens` is the requested output limit per model call; it must be smaller than `--context-window` and supported by your chat server. A large value such as `65536` may be rejected by a server with a smaller configured limit; the printed HTTP error will show whether that is happening.

Each task line must contain `benchmark_seed.facts_clean`, an `issues` list, and a `laws` list with nested `provisions`. Invalid JSONL lines fail with their line number.
