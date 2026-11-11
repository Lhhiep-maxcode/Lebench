"""Extract judgment sections by headings, with optional, logged OCR word repairs."""

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path


# Edit this small predefined vocabulary to support another judgment template.
HEADINGS = {
    'NỘI DUNG VỤ ÁN': 'content',
    'NỘI DUNG CỦA VỤ ÁN': 'content',
    'NỘI DUNG VỤ VIỆC': 'content',
    'NỘI DUNG VIỆC DÂN SỰ': 'content',
    'NỘI DUNG VIỆC HÔN NHÂN VÀ GIA ĐÌNH': 'content',
    'NHỮNG VẤN ĐỀ CỤ THỂ YÊU CẦU TÒA ÁN GIẢI QUYẾT': 'content',
    'NHỮNG VẤN ĐỀ YÊU CẦU TÒA ÁN GIẢI QUYẾT': 'content',
    'NHẬN THẤY': 'facts',
    'XÉT THẤY': 'consideration',
    'NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ': 'reasoning',
    'NHẬN ĐỊNH CỦA TÒA ÁN': 'reasoning',
    'NHẬN ĐỊNH NHƯ SAU': 'reasoning',
    'NHẬN ĐỊNH': 'reasoning',
    'QUYẾT ĐỊNH': 'decision',
}
PAGE_MARKER = re.compile(r'(?i)#{2,6}[ \t]*Page[ \t]*(?:\{[ \t]*)?\d+(?:[ \t]*/[ \t]*\d+)*(?:[ \t]*\})?(?![\w/])')
WORD = re.compile(r'(?<!\w)[^\W\d_]+(?!\w)', re.UNICODE)
# A permissive syllable shape protects valid words absent from the vocabulary.
VIETNAMESE_SYLLABLE = re.compile(r'(?:ngh|ng|nh|ph|th|tr|ch|kh|gh|gi|qu|[bcdghklmnpqrstvx])?[aeiouy]+(?:ch|nh|ng|[cmnpt])?\Z')


def fold(text):
    text = unicodedata.normalize('NFKD', text.lower().replace('đ', 'd'))
    return ''.join(c for c in text if not unicodedata.combining(c))


def compact(text):
    return re.sub('[^a-z]', '', fold(text))


def load_vocabulary(path):
    # Viet74K includes multiword entries; the OCR corrector operates on syllables.
    return {word for line in path.read_text(encoding='utf-8-sig').splitlines()
            for word in unicodedata.normalize('NFC', line.partition('#')[0]).lower().split()
            if WORD.fullmatch(word)}


def strip_page_markers(text):
    # Use a newline, not an empty string, so words on adjacent pages do not join.
    return PAGE_MARKER.sub('\n', text)


def find_headings(text):
    # Search an accent-insensitive shadow; retain positions in the ORIGINAL text.
    shadow, positions = [], []
    for i, char in enumerate(text):
        for letter in fold(char):   # fold("NHẬN ĐỊNH CỦA TÒA ÁN") -> "nhan dinh cua toa an"
            shadow.append(letter)
            positions.append(i)     # positions[i] = index of the original char in text
    shadow = ''.join(shadow)
    names = list(HEADINGS)
    patterns = [r'\s*'.join(re.escape(w) for w in fold(name).split()) for name in names] # regex for each heading
    pattern = r'\b(?:'+ '|'.join(f'(?P<h{i}>{p})' for i, p in enumerate(patterns))+r')\b' # Combine all heading patterns
    found = []
    for match in re.finditer(pattern, shadow):
        start, end = positions[match.start()], positions[match.end()-1]+1
        while end < len(text) and unicodedata.combining(text[end]):
            end += 1
        name = names[int(match.lastgroup[1:])]
        prefix = text[text.rfind('\n', 0, start)+1:start]
        line_end = text.find('\n', end)
        suffix = text[end:line_end if line_end >= 0 else len(text)]
        at_line_start = not re.sub(r'[\s#*\d\[\]().:–-]', '', prefix)
        standalone = not suffix.strip(' \t\r:*#.;')
        colon = suffix.lstrip(' \t*').startswith(':')
        upper = text[start:end].isupper()
        # Avoid treating prose such as "quyết định số ..." as a section boundary.
        if HEADINGS[name] == 'decision':
            valid = ((at_line_start and standalone) or (colon and (upper or at_line_start))
                     or (upper and bool(re.match(r'\s*1[.)]', suffix))))
        else:
            valid = (at_line_start and (colon or standalone or HEADINGS[name] in ('facts', 'consideration'))) or upper
        if valid:
            found.append({'start': start, 'end': end, 'heading': name, 'fuzzy': False})

    # Fuzzy matching is restricted to short uppercase labels, never arbitrary prose.
    # {
    #     "noidungvuan": "NỘI DUNG VỤ ÁN",
    #     "nhandinhcuatoaan": "NHẬN ĐỊNH CỦA TÒA ÁN",
    #     "quyetdinh": "QUYẾT ĐỊNH",
    #     ...
    # }
    lookup = {compact(name): name for name in names}    # compact() lowercases, removes accents, and keeps only letters a–z.
    excluded = {'nhandanh', 'nhanthan', 'quydinh'}  # Real words, not OCR errors.
    offset = 0
    for line in text.splitlines(keepends=True):
        label = line.split(':', 1)[0].strip(' \t\r\n#*0123456789[].)-–')
        key = compact(label)
        if (label.isupper() and 5 <= len(key) <= 55 and key not in excluded
                and not key.startswith(('ve', 'toaan')) and key not in lookup):
            matches = difflib.get_close_matches(key, lookup, n=2, cutoff=0.80)
            if matches:
                score = lambda k: difflib.SequenceMatcher(None, key, k).ratio()
                if len(matches) == 1 or score(matches[0])-score(matches[1]) >= 0.08:
                    start = offset+line.index(label)
                    if not any(h['start'] <= start < h['end'] for h in found):
                        found.append({'start': start, 'end': start+len(label),
                                      'heading': lookup[matches[0]], 'fuzzy': True})
        offset += len(line)
    return sorted(found, key=lambda h: h['start'])


def extract_section(text):
    headings = find_headings(text)
    first_reasoning = next((h['start'] for h in headings if HEADINGS[h['heading']] == 'reasoning'), len(text))
    # A later "Xét thấy" inside reasoning is not a missing factual section.
    eligible = [h for h in headings if h['start'] < first_reasoning]
    start = next((h for role in ('content', 'facts', 'consideration') for h in eligible
                  if HEADINGS[h['heading']] == role), None)
    if start is None:
        return None, {'warnings': ['missing_start_heading']}
    ends = [h for h in headings if h['start'] > start['end'] and
            (HEADINGS[h['heading']] in ('reasoning', 'decision') or
             (HEADINGS[h['heading']] == 'consideration' and HEADINGS[start['heading']] != 'consideration'))]
    if not ends:
        return None, {'start_heading': start, 'warnings': ['missing_end_heading']}
    end = ends[0]
    left, right = start['end'], end['start']
    while left < right and text[left] in ' \t\r\n:*#':
        left += 1
    while right > left and text[right-1].isspace():
        right -= 1
    warnings = []
    if start['fuzzy'] or end['fuzzy']:
        warnings.append('fuzzy_heading_match')
    if HEADINGS[start['heading']] == 'consideration':
        warnings.append('xet_thay_may_include_legal_reasoning')
    if any(HEADINGS[h['heading']] in ('content', 'facts') and left < h['start'] < right for h in headings):
        warnings.append('multiple_content_headings')
    if left == right:
        warnings.append('empty_section')
    return text[left:right], {'start_heading': start, 'end_heading': end, 'start': left, 'end': right,
                              'offset_basis': 'markdown_without_page_markers', 'warnings': warnings}


def repair_ocr(text, vocabulary):
    """Only unique, one-edit lowercase matches; every replacement is recorded for review."""
    protected = {'không', 'có', 'chưa', 'đã', 'phải', 'được', 'miễn',
                 'gram', 'gam', 'kilogram', 'kilogam', 'kg', 'mg', 'ml', 'cm', 'mm', 'km', 'usd', 'vnd', 'eur'}
    candidates_by_shape = {}
    for word in vocabulary:
        if word and word not in protected:
            candidates_by_shape.setdefault((word[0], len(word)), []).append(word)

    @lru_cache(maxsize=20000)
    def nearest(word):
        if (word in vocabulary or word in protected or len(word) < 4
                or VIETNAMESE_SYLLABLE.fullmatch(fold(word))):
            return word
        candidates = [v for size in (len(word)-1, len(word), len(word)+1)
                      for v in candidates_by_shape.get((word[0], size), ())]
        matches = difflib.get_close_matches(word, candidates, n=2, cutoff=0.75)
        if not matches:
            return word
        best = matches[0]
        ratio = lambda v: difflib.SequenceMatcher(None, word, v).ratio()
        changes = [op for op in difflib.SequenceMatcher(None, word, best).get_opcodes() if op[0] != 'equal']
        edits = sum(max(b-a, d-c) for tag, a, b, c, d in changes)
        tag, a, b, c, d = changes[0]
        visual_confusion = (word[a:b], best[c:d]) in {('ƣ', 'ư'), ('µ', 'à'), ('¹', 'ạ'), ('l', 'i'), ('i', 'l')}
        repeated_letter = tag == 'delete' and b-a == 1 and (
            (a > 0 and word[a-1] == word[a]) or (b < len(word) and word[b] == word[a]))
        # Tone-only substitutions can change valid words; leave them unchanged.
        if (edits != 1 or not (visual_confusion or repeated_letter) or fold(word) == fold(best)
                or (len(matches) > 1 and ratio(best)-ratio(matches[1]) < 0.12)):
            return word
        return best

    tokens = list(WORD.finditer(text))
    pieces, corrections, cursor, i = [], [], 0, 0
    while i < len(tokens):
        token = tokens[i]
        start, end, replacement, consumed = token.start(), token.end(), token.group(), 1
        neighbors = text[start-1:start] if start else ''
        neighbors += text[end:end+1]
        if replacement.islower() and not any(c in '/_-@' or unicodedata.combining(c) for c in neighbors):
            # Join obvious broken syllables such as "th ẩm" when present in the vocabulary.
            for size in (3, 2):
                chunk = tokens[i:i+size]
                if len(chunk) != size or not all(t.group().islower() for t in chunk):
                    continue
                gaps = [text[a.end():b.start()] for a, b in zip(chunk, chunk[1:])]
                joined = ''.join(t.group() for t in chunk)
                if (all(g.isspace() and len(g) <= 4 for g in gaps) and joined in vocabulary
                        and joined not in protected and any(len(t.group()) <= 2 and t.group() not in vocabulary for t in chunk)):
                    end, replacement, consumed = chunk[-1].end(), joined, size
                    break
            # Do not guess a unit or another token immediately following a quantity.
            if consumed == 1 and not re.search(r'\d[ \t]*$', text[max(0, start-20):start]):
                replacement = nearest(replacement)
        if replacement != text[start:end]:
            corrections.append({'start': start, 'end': end, 'before': text[start:end], 'after': replacement})
        pieces.extend((text[cursor:start], replacement))
        cursor, i = end, i+consumed
    return ''.join(pieces)+text[cursor:], corrections


def extract_document(row, vocabulary=None):
    text = row.get('markdown')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('markdown is missing or empty')
    cleaned = strip_page_markers(text)
    raw, metadata = extract_section(cleaned)
    situation, corrections = raw, []
    if raw and vocabulary:
        situation, corrections = repair_ocr(raw, vocabulary)
        if corrections:
            metadata['warnings'].append('ocr_replacements_require_review')
    if raw and re.search(r'vien\s+kiem\s+sat[\s\S]{0,350}?de\s+nghi', fold(raw)):
        metadata['warnings'].append('may_include_prosecutor_recommendations')
    if raw and re.search('[ƣƢ©®µ¶¸¹º»¼½¾¿×ØÞß]', raw):
        metadata['warnings'].append('ocr_or_legacy_font_artifacts')
    metadata.update(method='heading_rules', ocr_corrections=corrections)
    status = 'missing_section' if not raw else ('needs_review' if metadata['warnings'] else 'extracted')
    return {**row, 'markdown': cleaned, 'markdown_chars': len(cleaned), 'situation_raw': raw, 'situation': situation,
            'situation_status': status, 'situation_extraction': metadata,
            'situation_judge': None, 'situation_attempts': 0, 'situation_error': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='output/situation_with_judgments.jsonl')
    parser.add_argument('--output', default='output/situation_rules.jsonl', help='New JSONL file; overwritten on each run')
    parser.add_argument('--fix-ocr', action='store_true', help='Apply conservative dictionary repairs; edited extracts require review')
    parser.add_argument('--vocab', type=Path, default=Path(__file__).with_name('Viet74K.txt'), help='OCR vocabulary (default: server/Viet74K.txt)')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    source, destination = Path(args.input).resolve(), Path(args.output).resolve()
    if source == destination:
        parser.error('Input and output must be different files')
    if args.limit is not None and args.limit < 1:
        parser.error('limit must be positive')
    vocabulary = None
    if args.fix_ocr:
        vocabulary = load_vocabulary(args.vocab)
    counts = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding='utf-8-sig') as stream, destination.open('w', encoding='utf-8') as output:
        selected = 0
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if args.limit is not None and selected >= args.limit:
                break
            selected += 1
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected a JSON object')
                result = extract_document(row, vocabulary)
                if vocabulary is not None:
                    result['situation_extraction'].update(ocr_vocabulary=args.vocab.name,
                                                          ocr_vocabulary_size=len(vocabulary))
            except ValueError as exc:
                counts['invalid_input'] += 1
                print(f'Line {number}: {exc}', file=sys.stderr)
                continue
            counts[result['situation_status']] += 1
            output.write(json.dumps(result, ensure_ascii=False)+'\n')
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    return int(bool(counts['invalid_input']))


if __name__ == '__main__':
    raise SystemExit(main())
