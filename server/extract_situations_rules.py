"""Extract facts, application and conclusion from judgments using heading rules."""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import date
from itertools import islice
from pathlib import Path

from tqdm import tqdm


HEADINGS = {
    'NỘI DUNG VỤ ÁN': 'facts',
    'NỘI DUNG CỦA VỤ ÁN': 'facts',
    'NỘI DUNG VỤ VIỆC': 'facts',
    'NỘI DUNG VIỆC DÂN SỰ': 'facts',
    'NỘI DUNG VIỆC HÔN NHÂN VÀ GIA ĐÌNH': 'facts',
    'NHỮNG VẤN ĐỀ CỤ THỂ YÊU CẦU TÒA ÁN GIẢI QUYẾT': 'facts',
    'NHỮNG VẤN ĐỀ YÊU CẦU TÒA ÁN GIẢI QUYẾT': 'facts',
    'NHẬN THẤY': 'facts',
    'XÉT THẤY': 'application',
    'NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ': 'application',
    'NHẬN ĐỊNH CỦA TÒA ÁN': 'application',
    'NHẬN ĐỊNH NHƯ SAU': 'application',
    'NHẬN ĐỊNH': 'application',
    'QUYẾT ĐỊNH': 'decision',
}
PAGE_MARKER = re.compile(
    r'(?i)#{2,6}[ \t]*Page[ \t]*(?:\{[ \t]*)?\d+(?:[ \t]*/[ \t]*\d+)*(?:[ \t]*\})?(?![\w/])'
)
# A wrapped PDF line is not necessarily a new paragraph. Also split at a new
# line following sentence/list punctuation, since PDF text often has no blank lines.
PARAGRAPH_BREAK = re.compile(r'\r?\n[ \t\r]*\n|[.!?;:]["”\')\]]*[ \t\r]*\n[ \t]*')
# Explicit aliases keep OCR matching limited to prosecutor names, not arbitrary words.
PROSECUTOR_NAMES = (
    'Viện kiểm sát', 'Vện kiểm sát', 'Việm kiểm sát', 'Vịn kiểm sát',
    'Viện kiểm xát', 'Viện kiểm soát', 'Vlen kiểm sát', 'Viện klem sát',
    'V1ện k1ểm sát', 'Viện KS', 'Viện KSND',
    'Kiểm sát viên', 'Kiểm sát vên', 'VKS', 'VKSND', 'VKSNDTC', 'VKSNDCC', 'VKSTC', 'KSV',
)
OPINION = re.compile(
    r'phatbieu|ykien|quandiem|chorang|'
    r'denghi|kiennghi|dexuat|luantoi|ketluan|giunguyenquandiem|giunguyenquyetdinhtruyto'
)
OPINION_PREFIX = re.compile(
    r'(?:ykien(?:phatbieu)?|quandiem|phatbieu)(?:cua)?(?:nguoi|vi)?(?:daidien)?$'
)
FOOTER = re.compile(
    r'(?m)^[ \t#*]*(?:noi\s+nhan\b|t\.?\s*m\.?\s*hoi\s+dong\b|'
    r'tham\s+phan(?:\s*[-–]\s*chu\s+toa[^\n]*)?[ \t:*#]*$)'
)
DATE_PATTERN = re.compile(
    r'\bn\s*g\s*a\s*y\s*:?\s*(\d{1,2})\s*(?:t\s*h\s*a\s*n\s*g|[/\.–—-])\s*'
    r'(\d{1,2})\s*(?:n\s*a\s*m|[/\.–—-])\s*(\d(?:[ \t]*\d){3})(?!\d)'
)


def fold(text):
    text = unicodedata.normalize('NFKD', text.lower().replace('đ', 'd'))
    return ''.join(char for char in text if not unicodedata.combining(char))


def phrase_pattern(names):
    """Allow case/accent differences, OCR spaces and dotted abbreviations."""
    keys = {re.sub(r'\s+', '', fold(name)) for name in names}
    return re.compile('|'.join(
        r'\b' + r'[\s.]*'.join(re.escape(char) for char in key) + r'\b'
        for key in sorted(keys, key=lambda key: (-len(key), key))
    ))


PROSECUTOR = phrase_pattern(PROSECUTOR_NAMES)
FINAL_STATEMENT = phrase_pattern(('lời nói sau cùng', 'lời sau cùng', 'lời cuối cùng', 'lời nói cuối cùng'))


def is_criminal(category):
    return isinstance(category, str) and category.strip().casefold() == 'criminal'


def paragraph_spans(text):
    """Keep original offsets, including wrapped final-statement headings and bodies."""
    spans, start = [], 0
    for boundary in PARAGRAPH_BREAK.finditer(text):
        end = boundary.start() + (1 if text[boundary.start()] in '.!?;:' else 0)
        if text[start:end].strip():
            spans.append((start, end))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    merged = []
    for start, end in spans:
        if merged:
            previous = fold(text[merged[-1][0]:merged[-1][1]])
            marker = FINAL_STATEMENT.search(previous)
            # "Lời nói sau cùng:\nBị cáo ..." is one statement, not two paragraphs.
            if marker and not previous[marker.end():].strip(' \t\r\n:*#.;'):
                merged[-1] = (merged[-1][0], end)
                continue
        merged.append((start, end))
    return merged


def shadow_with_positions(text):
    """Normalize only for matching; offsets still point to the original text."""
    letters, positions = [], []
    for index, char in enumerate(text):
        folded = fold(char)
        letters.extend(folded)
        positions.extend([index] * len(folded))
    return ''.join(letters), positions


def strip_page_markers(text):
    # Also imported by extract_situations_simple.py.
    return PAGE_MARKER.sub('\n', text)


HEADING_NAMES = sorted(HEADINGS, key=len, reverse=True)
HEADING_PATTERN = re.compile(r'\b(?:' + '|'.join(
    f'(?P<h{i}>' + r'\s*'.join(re.escape(word) for word in fold(name).split()) + ')'
    for i, name in enumerate(HEADING_NAMES)
) + r')\b')


def find_headings(text):
    """Match the listed aliases, tolerating case, accents and whitespace only."""
    shadow, positions = shadow_with_positions(text)
    headings = []
    for match in HEADING_PATTERN.finditer(shadow):
        start = positions[match.start()]
        end = positions[match.end() - 1] + 1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        name = HEADING_NAMES[int(match.lastgroup[1:])]
        prefix = text[text.rfind('\n', 0, start) + 1:start]
        line_end = text.find('\n', end)
        suffix = text[end:line_end if line_end >= 0 else len(text)]
        at_line_start = not re.sub(r'[\s#*\d\[\]().:–-]', '', prefix)
        standalone = not suffix.strip(' \t\r:*#.;')
        colon = suffix.lstrip(' \t*').startswith(':')
        upper = text[start:end].isupper()
        if HEADINGS[name] == 'decision':
            valid = ((at_line_start and standalone)
                     or (colon and (upper or at_line_start))
                     or (upper and bool(re.match(r'\s*(?:1|I)[.)]', suffix))))
        else:
            valid = (at_line_start and (standalone or colon)) or upper
        if valid:
            headings.append({'start': start, 'end': end, 'heading': name, 'role': HEADINGS[name]})
    return headings


def is_prosecutor_opinion(paragraph):
    # Removing whitespace also recognizes OCR splits such as "phát bi ểu ý ki ến".
    shadow = fold(paragraph)
    for actor in PROSECUTOR.finditer(shadow):
        before = re.sub(r'\s+', '', shadow[max(0, actor.start() - 180):actor.start()])[-70:]
        after = re.sub(r'\s+', '', shadow[actor.end():actor.end() + 650])[:250]
        if OPINION_PREFIX.search(before) or OPINION.search(after):
            return True
    return False


def clean_facts(facts, category=None):
    """Remove prosecutor material; Criminal cases can resume at the final statement."""
    criminal = is_criminal(category)
    paragraphs = paragraph_spans(facts)
    for index, (start, end) in enumerate(paragraphs):
        paragraph = facts[start:end]
        detected = PROSECUTOR.search(fold(paragraph)) if criminal else is_prosecutor_opinion(paragraph)
        if not detected:
            continue
        prefix = facts[:start].rstrip()
        if criminal:
            resume = next((left for left, right in paragraphs[index:]
                           if FINAL_STATEMENT.search(fold(facts[left:right]))), None)
            if resume is not None:
                if resume == start:
                    return facts
                tail = facts[resume:].lstrip()
                return '\n\n'.join(part for part in (prefix, tail) if part)
        return prefix
    return facts


def filter_extracted(row):
    """Reject empty clean facts or remaining prosecutor-name variants."""
    seed = row.get('benchmark_seed') if isinstance(row, dict) else None
    facts_clean = seed.get('facts_clean') if isinstance(seed, dict) else None
    if not isinstance(facts_clean, str):
        return None, 'invalid_input'
    if not facts_clean.strip():
        return None, 'empty_facts_clean'
    criminal = is_criminal(row.get('category'))
    for start, end in paragraph_spans(facts_clean):
        paragraph = fold(facts_clean[start:end])
        if PROSECUTOR.search(paragraph) and not (criminal and FINAL_STATEMENT.search(paragraph)):
            return None, 'prosecutor_in_facts_clean'
    return row, 'kept'


def extract_judgment_date(header):
    """Prefer a caption date, then an explicit current hearing date; never use a case-filing date."""
    shadow = fold(header)
    intro = re.search(r'\bnhan\s+danh\b|\bthanh\s+phan\b|\bhoi\s+dong\s+xet\s+xu\b', shadow)
    caption_end = intro.start() if intro else len(shadow)
    caption_dates, hearing_dates = set(), set()
    for match in DATE_PATTERN.finditer(shadow):
        day, month, year = (int(re.sub(r'\s+', '', value)) for value in match.groups())
        try:
            value = date(year, month, day).isoformat()
        except ValueError:
            continue
        prefix = shadow[shadow.rfind('\n', 0, match.start()) + 1:match.start()]
        line_end = shadow.find('\n', match.end())
        suffix = shadow[match.end():line_end if line_end >= 0 else len(shadow)]
        caption_prefix = not prefix.strip(' \t\r#*') or re.search(r'[^\n\d,]{2,100},[ \t]*$', prefix)
        caption_suffix = not suffix.strip(' \t\r.,;:*#') or re.match(
            r'\s*(?:quyet\s+dinh|nhan\s+danh|v/v)\b', suffix)
        if match.start() < caption_end and caption_prefix and caption_suffix:
            caption_dates.add(value)
        following = shadow[match.end():match.end() + 300]
        hearing = re.search(r'\bxet\s*x\s*u\b|\bmo\s*phien\s*hop\b', following)
        if (re.match(r'\s*,?\s*(?:tai|toa\s*an)\b', following) and hearing
                and not re.search(r'\b(?:noi\s*cu\s*tru|tien\s*an|thu\s*ly)\b', following[:hearing.start()])):
            hearing_dates.add(value)
    if caption_dates:
        return next(iter(caption_dates)) if len(caption_dates) == 1 else None
    return next(iter(hearing_dates)) if len(hearing_dates) == 1 else None


def trim_section(text):
    return text.lstrip(' \t\r\n:*#').rstrip()


def extract_document(row):
    """Return (enriched row, 'kept'), or (None, reason) for a filtered row."""
    if not isinstance(row, dict) or not isinstance(row.get('markdown'), str) or not row['markdown'].strip():
        return None, 'invalid_input'
    text = strip_page_markers(row['markdown'])
    headings = find_headings(text)
    first_facts = next((h['start'] for h in headings if h['role'] == 'facts'), None)
    # A decision title before the factual section is a document title, not its conclusion.
    headings = [h for h in headings if not (
        h['role'] == 'decision' and first_facts is not None and h['start'] < first_facts)]
    counts = Counter(h['role'] for h in headings)
    if any(counts[role] == 0 for role in ('facts', 'application', 'decision')):
        return None, 'missing_headings'
    if any(counts[role] != 1 for role in ('facts', 'application', 'decision')):
        return None, 'duplicate_headings'
    if [h['role'] for h in headings] != ['facts', 'application', 'decision']:
        return None, 'invalid_order'
    facts_heading, application_heading, decision_heading = headings
    facts = trim_section(text[facts_heading['end']:application_heading['start']])
    application = trim_section(text[application_heading['end']:decision_heading['start']])
    conclusion = text[decision_heading['end']:]
    shadow, positions = shadow_with_positions(conclusion)
    footer = FOOTER.search(shadow)
    if footer:
        conclusion = conclusion[:positions[footer.start()]]
    conclusion = trim_section(conclusion)
    if not all((facts, application, conclusion)):
        return None, 'empty_section'
    seed = {
        'facts': facts,
        'facts_clean': clean_facts(facts, row.get('category')),
        'application': application,
        'conclusion': conclusion,
        'judgment_date': extract_judgment_date(text[:facts_heading['start']]),
    }
    return filter_extracted({**row, 'benchmark_seed': seed})


def process_line(item):
    """Top-level worker function so multiprocessing also works on Windows."""
    number, line, filter_only = item
    try:
        row = json.loads(line)
    except ValueError as exc:
        return 'invalid_input', number, str(exc)
    result, status = filter_extracted(row) if filter_only else extract_document(row)
    if status == 'invalid_input':
        field = 'a benchmark_seed.facts_clean string' if filter_only else 'nonempty markdown'
        return status, number, f'Expected an object with {field}'
    return status, number, json.dumps(result, ensure_ascii=False) if result is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('output/situation_with_judgments.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('output/benchmark_seeds.jsonl'))
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--chunksize', type=int, default=16, help='Rows sent to a worker per batch')
    parser.add_argument('--limit', type=int, help='Maximum number of nonblank input lines to process')
    parser.add_argument('--no-progress', action='store_true', help='Hide the tqdm progress bar')
    parser.add_argument('--filter-only', action='store_true',
                        help='Filter existing benchmark_seed.facts_clean without re-extracting sections')
    args = parser.parse_args()
    if args.workers < 1 or args.chunksize < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers, chunksize and limit must be positive')
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    try:
        with args.input.open(encoding='utf-8-sig') as source:
            total = sum(1 for line in source if line.strip())
            if args.limit is not None:
                total = min(total, args.limit)
            source.seek(0)
            with mp.Pool(args.workers) as pool, args.output.open('w', encoding='utf-8') as output, \
                    tqdm(total=total, desc='Judgments', unit='doc', disable=args.no_progress) as progress:
                lines = ((number, line, args.filter_only) for number, line in enumerate(source, 1) if line.strip())
                selected = islice(lines, args.limit) if args.limit is not None else lines
                for status, number, payload in pool.imap(process_line, selected, chunksize=args.chunksize):
                    counts[status] += 1
                    if status == 'kept':
                        output.write(payload + '\n')
                    elif status == 'invalid_input':
                        tqdm.write(f'Line {number}: {payload}', file=sys.stderr)
                    progress.set_postfix(kept=counts['kept'],
                                         filtered=sum(counts.values()) - counts['kept'] - counts['invalid_input'],
                                         errors=counts['invalid_input'], refresh=False)
                    progress.update(1)
    except OSError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    return int(bool(counts['invalid_input']))


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
