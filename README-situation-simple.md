# Situation extraction

Extract Vietnamese legal situations using a writer and an independent judge. If the judge rejects a draft, the writer receives feedback and rewrites it, up to `--max-retries` times. The judge sees the judgment and current draft, without previous drafts or feedback. Multiple judgments run concurrently.

## 1. Install

From the project root:

```bash
python -m pip install httpx tqdm
```

Keep `server/situation_prompts.py` and `server/extract_situations_rules.py` alongside the script. Edit prompts in `server/situation_prompts.py`.

## 2. Prepare input

Use JSONL: one JSON object per line, with a nonempty `doc_name` and the full judgment in `markdown`:

```json
{"doc_name": "judgment_001", "markdown": "Full Vietnamese judgment text..."}
```

Other fields are preserved. Optional `citations_law` metadata can guide the judge when `--use-law-hints` is enabled. Page markers such as `## Page 1` are removed before sending the judgment to the models.

## 3. Run

Replace the input path, server URL, and model name as needed:

```bash
python server/extract_situations_simple.py --input output/situation_with_judgments.jsonl --output output/situations_simple.jsonl --base-url http://localhost:8000/v1 --model YOUR_MODEL --concurrency 8 --max-retries 3
```

The endpoint must support OpenAI-compatible `/chat/completions` requests and JSON-object responses for the judge. Add `--api-key YOUR_KEY` if authentication is required.

For a small trial, add `--limit 5`.

## Main options

| Option | Meaning | Default |
|---|---|---|
| `--concurrency` | Judgments processed simultaneously. | `8` |
| `--max-retries` | Rewrites after the first draft; `3` allows up to four writer/judge rounds. Stops early on acceptance. This is not an HTTP retry setting. | `3` |
| `--max-tokens` | Maximum generated tokens per request, including reasoning tokens for models that use them. Input tokens are separate. | `8192` |
| `--timeout` | HTTP timeout in seconds. | `300` |
| `--limit` | Process only the first N nonblank input lines. | All |
| `--judge-model` | Model used by the judge. | Writer model |
| `--judge-base-url` | Server URL used by the judge. | Writer server |
| `--use-law-hints` | Give the judge optional `citations_law` hints. | Disabled |
| `--extra-body` | Additional request settings as a JSON object. | `{}` |
| `--no-progress` | Hide the tqdm progress bar. | Progress shown |

`--model` is an alias for `--writer-model`. You can also configure `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_API_KEY` as environment variables.

## Output

Each completed judgment is immediately written and flushed to the output JSONL. Results appear in completion order, which may differ from input order. Each result preserves the input fields and adds:

- `situation`: the latest generated situation.
- `situation_status`: `passed`, `needs_review` after all rounds were rejected, or `error` if processing failed.
- `situation_attempts`: number of writer attempts started.
- `situation_judge`: the latest judge verdict and feedback, when available.
- `situation_error`: error details, or `null`.

Invalid JSON lines or rows without a valid `doc_name` are reported and skipped. A `passed` result means the model judge accepted it; it is not a guarantee of correctness.

**Each run overwrites the output file. This version has no `--resume` option.** Use a different output path to preserve earlier results.

See all available flags:

```bash
python server/extract_situations_simple.py --help
```
