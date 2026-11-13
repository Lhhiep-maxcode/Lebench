"""Post-filter extracted judgments by category cutoff-length percentile and prosecutor mentions."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

if __package__:
    from .extract_situations_rules import PROSECUTOR, fold, paragraph_spans
else:
    from extract_situations_rules import PROSECUTOR, fold, paragraph_spans


REPORT_FIELDS = (
    'line_number', 'id', 'doc_name', 'category', 'facts_chars', 'facts_clean_chars',
    'first_prosecutor_paragraph_start', 'remaining_chars', 'remaining_ratio',
    'prosecutor_in_facts_clean', 'cutoff_percentile', 'category_threshold_chars',
    'cutoff_pass', 'kept', 'reason', 'error',
)


def category_key(value):
    return value.strip().casefold() if isinstance(value, str) and value.strip() else 'uncategorized'


def cutoff_metrics(facts):
    """Measure the whole original suffix, not just the text removed by the cleaner."""
    start = next((left for left, right in paragraph_spans(facts)
                  if PROSECUTOR.search(fold(facts[left:right]))), None)
    if start is not None:
        while start < len(facts) and facts[start].isspace():
            start += 1
    remaining = len(facts) - start if start is not None else 0
    return {
        'facts_chars': len(facts),
        'first_prosecutor_paragraph_start': start,
        'remaining_chars': remaining,
        'remaining_ratio': remaining / len(facts) if facts else 0.0,
    }


def percentile(sorted_values, percent):
    """Linear interpolation between adjacent values; input must already be sorted."""
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def analyze_line(item):
    number, line = item
    audit = dict.fromkeys(REPORT_FIELDS)
    audit.update(line_number=number, category='uncategorized', reason='invalid_input', error='')
    try:
        row = json.loads(line)
    except ValueError as exc:
        audit['error'] = str(exc)
        return audit
    if not isinstance(row, dict):
        audit['error'] = 'Expected a JSON object'
        return audit
    audit.update(id=row.get('id'), doc_name=row.get('doc_name'), category=category_key(row.get('category')))
    seed = row.get('benchmark_seed')
    if not isinstance(seed, dict):
        audit['error'] = 'benchmark_seed is missing or not an object'
        return audit
    facts, facts_clean = seed.get('facts'), seed.get('facts_clean')
    if not isinstance(facts, str) or not facts.strip() or not isinstance(facts_clean, str):
        audit['error'] = 'Expected nonempty benchmark_seed.facts and a facts_clean string'
        return audit
    audit.update(cutoff_metrics(facts), facts_clean_chars=len(facts_clean),
                 prosecutor_in_facts_clean=bool(PROSECUTOR.search(fold(facts_clean))),
                 reason='candidate' if facts_clean.strip() else 'empty_facts_clean')
    return audit


def select_reason(audit, threshold):
    if audit['reason'] != 'candidate':
        return audit['reason']
    if audit['remaining_chars'] > threshold:
        return 'cutoff_above_percentile'
    # Additional strict gate: no exception for Criminal final statements here.
    if audit['prosecutor_in_facts_clean']:
        return 'prosecutor_in_facts_clean'
    return 'kept'


def run(args):
    counts, category_counts = Counter(), defaultdict(Counter)
    lengths = defaultdict(list)
    input_stat = args.input.stat()
    with args.input.open(encoding='utf-8-sig') as source:
        total = sum(1 for line in source if line.strip())
        source.seek(0)
        # Stage only small audit records on disk. Original judgments are read again
        # after thresholds are known, so their full contents never accumulate in RAM.
        with tempfile.TemporaryFile(mode='w+t', encoding='utf-8', dir=args.output.parent) as audits:
            with mp.Pool(args.workers) as pool, \
                    tqdm(total=total, desc='Analyze cutoff', unit='doc', disable=args.no_progress) as progress:
                lines = ((number, line) for number, line in enumerate(source, 1) if line.strip())
                for audit in pool.imap(analyze_line, lines, chunksize=args.chunksize):
                    if audit['reason'] == 'candidate':
                        lengths[audit['category']].append(audit['remaining_chars'])
                    if audit['reason'] == 'invalid_input':
                        tqdm.write(f"Line {audit['line_number']}: {audit['error']}", file=sys.stderr)
                    audits.write(json.dumps(audit, ensure_ascii=False) + '\n')
                    progress.update(1)

            ordered = {category: sorted(values) for category, values in lengths.items()}
            thresholds = {category: percentile(values, args.cutoff_percentile)
                          for category, values in ordered.items()}
            current_stat = args.input.stat()
            if (input_stat.st_size, input_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
                raise ValueError('Input changed during analysis; re-run using a completed input file')
            source.seek(0)
            audits.seek(0)
            with args.output.open('w', encoding='utf-8') as output, \
                    args.report.open('w', encoding='utf-8-sig', newline='') as report, \
                    tqdm(total=total, desc='Select samples', unit='doc', disable=args.no_progress) as progress:
                writer = csv.DictWriter(report, fieldnames=REPORT_FIELDS)
                writer.writeheader()
                lines = ((number, line) for number, line in enumerate(source, 1) if line.strip())
                for (number, line), audit_json in zip(lines, audits):
                    audit = json.loads(audit_json)
                    if number != audit['line_number']:
                        raise ValueError('Input line order changed during processing')
                    threshold = thresholds.get(audit['category'])
                    reason = select_reason(audit, threshold)
                    audit.update(cutoff_percentile=args.cutoff_percentile, category_threshold_chars=threshold,
                                 cutoff_pass=(audit['remaining_chars'] <= threshold)
                                 if audit['reason'] == 'candidate' else None,
                                 kept=reason == 'kept', reason=reason)
                    if reason == 'kept':
                        output.write(line.rstrip('\r\n') + '\n')
                    writer.writerow(audit)
                    counts[reason] += 1
                    category_counts[audit['category']][reason] += 1
                    progress.set_postfix(kept=counts['kept'], filtered=sum(counts.values()) - counts['kept'],
                                         refresh=False)
                    progress.update(1)

    categories = {}
    for category in sorted(category_counts):
        values = ordered.get(category, [])
        categories[category] = {
            'input_rows': sum(category_counts[category].values()),
            'percentile_population': len(values),
            'threshold_remaining_chars': thresholds.get(category),
            'min_remaining_chars': values[0] if values else None,
            'median_remaining_chars': percentile(values, 50),
            'max_remaining_chars': values[-1] if values else None,
            'counts': dict(category_counts[category]),
        }
    summary = {
        'cutoff_percentile': args.cutoff_percentile,
        'length_unit': 'characters',
        'selection_rule': 'remaining_chars <= category percentile threshold AND no prosecutor mention in facts_clean',
        'percentile_population': 'Valid nonempty facts and facts_clean, before the strict prosecutor-mention filter',
        'no_prosecutor_paragraph_remaining_chars': 0,
        'counts': dict(counts),
        'categories': categories,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('output/benchmark_seeds_refined.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('output/benchmark_seeds_selected.jsonl'))
    parser.add_argument('--cutoff-percentile', type=float, default=10,
                        help='Keep cutoff lengths at or below this percentile within each category (0..100)')
    parser.add_argument('--report', type=Path, help='Per-judgment CSV; default: OUTPUT.cutoff.csv')
    parser.add_argument('--summary', type=Path, help='Category summary JSON; default: OUTPUT.cutoff_summary.json')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--chunksize', type=int, default=16)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.cutoff_percentile <= 100 or args.workers < 1 or args.chunksize < 1:
        parser.error('cutoff-percentile must be between 0 and 100; workers and chunksize must be positive')
    args.report = args.report or args.output.with_suffix('.cutoff.csv')
    args.summary = args.summary or args.output.with_suffix('.cutoff_summary.json')
    paths = [args.input, args.output, args.report, args.summary]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error('Input, output, report and summary must be different files')
    try:
        for path in (args.output, args.report, args.summary):
            path.parent.mkdir(parents=True, exist_ok=True)
        counts = run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return int(bool(counts.get('invalid_input')))


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
