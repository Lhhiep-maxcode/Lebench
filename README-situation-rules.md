# Extract benchmark seeds with rules

Run from the project root. Install the progress-bar dependency first:

```bash
python -m pip install tqdm
```

Then run:

```bash
python server/extract_situations_rules.py --input output/situation_with_judgments.jsonl --output output/benchmark_seeds.jsonl --workers 8
```

For a small trial, add `--limit 20`. `--workers` controls the number of worker processes; the default is up to eight CPU cores. `--chunksize` controls rows per worker batch (default: 16).

A tqdm bar shows processed judgments, percentage, speed, ETA, and counts of kept, filtered, and invalid rows. All processed rows advance the bar, including filtered rows. Add `--no-progress` to hide it.

## Additional quality filters: category percentiles and strict prosecutor check

After extraction, apply both additional filters with:

```bash
python server/filter_benchmark_seeds.py --input output/benchmark_seeds_refined.jsonl --output output/benchmark_seeds_selected.jsonl --cutoff-percentile 10 --workers 8
```

The input must contain `benchmark_seed.facts` and `benchmark_seed.facts_clean`. This post-filter preserves the retained records and leaves all extraction/cleaning rules unchanged.

1. For each judgment, find the first prosecutor-related paragraph in the original `facts`. Measure the number of **characters from that paragraph's first non-whitespace character to the end of facts**. This includes the whole suffix, even when the Criminal cleaner restored part of it. The existing paragraph detector and prosecutor aliases are reused. No prosecutor paragraph means a length of zero.
2. Calculate the configured percentile of those lengths separately within each category, using linear interpolation. Keep lengths **at or below** their category threshold. `--cutoff-percentile 10` means the lower tenth percentile of character lengths; `20` is less restrictive, and `100` disables length rejection. It does not mean removing less than 10% of an individual judgment. Ties are retained, so more than 10% can pass, especially if many lengths are zero. A singleton category retains its one sample at this stage.
3. Reject the entire sample if its final `facts_clean` contains any supported prosecutor name/typo/abbreviation, including in a **Criminal final statement**. This additional strict gate has no final-statement exception. Empty clean facts are also rejected. The text itself is not rewritten.

Category names are grouped after trimming whitespace and ignoring case; missing categories use `uncategorized`. Percentiles use rows with nonempty original facts and nonempty clean facts, before the new strict prosecutor check. Invalid or empty rows do not affect thresholds. Each run calculates thresholds from that run's input; use the complete extracted input to obtain dataset-wide thresholds.

The command produces:

- `output/benchmark_seeds_selected.jsonl`: retained samples, in input order.
- `output/benchmark_seeds_selected.cutoff.csv`: every input judgment, category, original/clean facts lengths, prosecutor-paragraph offset, suffix length and ratio, category threshold, both filter results, and rejection reason. `remaining_ratio` is informational; selection uses character lengths.
- `output/benchmark_seeds_selected.cutoff_summary.json`: per-category population, minimum/median/maximum length, threshold, and retention/rejection counts.

Override report locations with `--report PATH` and `--summary PATH`. Both the analysis and selection stages show tqdm progress. Analysis runs with multiprocessing, keeps only lengths in memory, and stages the small per-judgment audit records in a temporary file. Samples are read again after all category thresholds are known.

## Filter results already extracted

Drop samples with empty/whitespace-only `benchmark_seed.facts_clean` or residual prosecutor mentions:

```bash
python server/extract_situations_rules.py --input output/benchmark_seeds.jsonl --output output/benchmark_seeds_filtered.jsonl --filter-only --workers 8
```

This mode preserves every field of retained rows and does not re-extract sections. Matching ignores case, accents, OCR spaces, and dots in abbreviations. `PROSECUTOR_NAMES` lists supported aliases, including `Viện kiểm sát`, `Vện kiểm sát`, `Việm kiểm sát`, `VKS`, `VKSND`, `VKSNDTC`, `V.K.S.`, `Kiểm sát viên`, and `KSV`. Abbreviations must have word boundaries so an unrelated identifier such as `AVKSNDX` is not matched. Extend this list for additional observed typos; this is not a general spelling corrector.

The extraction-stage filter checks only `facts_clean`. For `Criminal` (case-insensitive), a prosecutor mention inside a final-statement paragraph is allowed at this stage. Mentions in other paragraphs still reject the sample. Missing/non-string values are `invalid_input`; blank values are `empty_facts_clean`. The additional post-filter above rejects all residual prosecutor mentions, including those in final statements.

The same final filter is also applied automatically during normal extraction.

## Restore final statements in Criminal judgments

Re-run normal extraction from the original judgments to rebuild previously truncated `facts_clean`:

```bash
python server/extract_situations_rules.py --input output/situation_with_judgments.jsonl --output output/benchmark_seeds_refined.jsonl --workers 8
```

For `Criminal`, the cleaner keeps the text before the first prosecutor paragraph, removes the intervening block, and resumes at the first later paragraph containing `lời nói sau cùng`, `lời sau cùng`, `lời cuối cùng`, or `lời nói cuối cùng`. It retains that entire paragraph and every paragraph after it. The final-statement paragraph is retained even if it mentions a prosecutor. If no later final statement exists, only the prefix is kept. A standalone final-statement heading ending with a colon is grouped with its following body paragraph.

Other categories keep the original cutoff behavior. `--filter-only` cannot restore final statements already deleted from `facts_clean`; it only checks the existing value.

## Input and output

Input is JSONL, with a full judgment in each row's `markdown` field. Output preserves the original fields and adds:

```json
{
  "benchmark_seed": {
    "facts": "Original factual section...",
    "facts_clean": "Clean factual text, including retained final statements for Criminal...",
    "application": "Court reasoning...",
    "conclusion": "Court decision...",
    "judgment_date": "2023-10-13"
  }
}
```

The input file is preserved. The output file is overwritten on each run. Retained rows stay in input order, including when multiple workers are used.

## Extraction rules

- Heading aliases are defined in `HEADINGS`. `XÉT THẤY` is **application**, never a facts fallback.
- Keep only documents with exactly one facts, one application, and one decision heading, in that order, with nonempty section bodies. A `QUYẾT ĐỊNH` document title before the facts section is ignored.
- Facts end at the application heading. Application ends at the decision heading. Conclusion ends at the recipient/signature block (`Nơi nhận`, `TM. HỘI ĐỒNG...`, or `THẨM PHÁN`) or the document end.
- Matching tolerates capitalization, Vietnamese accent differences, and whitespace/line wrapping. It does not guess misspelled headings or rewrite OCR words. `## Page n` markers are removed from the extracted text; the original `markdown` is preserved.
- For categories other than `Criminal`, `facts_clean` discards the first paragraph containing a prosecutor opinion/recommendation and everything after it. Names are combined with opinion cues such as `phát biểu`, `ý kiến`, `quan điểm`, or `đề nghị`; a bare mention is not sufficient to trigger this cutoff. The final filter still rejects residual prosecutor mentions.
- For `Criminal`, the cutoff starts at the first prosecutor mention, including an indictment paragraph, then resumes at a later final-statement paragraph as described above.
- Paragraphs are separated by blank lines or newlines following sentence/list punctuation. Other newlines are treated as PDF line wrapping. Paragraphs are copied intact; only the removed block and joining whitespace change. Rows whose clean facts become empty are filtered out.
- `judgment_date` uses ISO `YYYY-MM-DD`. It is parsed from a caption date before the facts, or an explicit current hearing date. Invalid, missing, or conflicting dates become `null`. The script does not copy potentially incorrect `issued_date` metadata or use a filing date as the judgment date.

The final summary reports retained rows and filtering reasons: `missing_headings`, `duplicate_headings`, `invalid_order`, `empty_section`, `empty_facts_clean`, or `prosecutor_in_facts_clean`. Invalid JSON and missing/empty `markdown` (or missing/non-string `benchmark_seed.facts_clean` in filter-only mode) are reported as `invalid_input`; processing continues, with exit code 1 if any occurred.

These are text rules, so unusual layouts still require inspection. Review a sample of `facts_clean` and dates before building the benchmark.
