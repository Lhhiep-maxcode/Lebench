"""Article-level scoring for free-text Vietnamese legal citations."""

import re
import unicodedata


def _fold(value):
    value = unicodedata.normalize('NFKD', value.casefold().replace('đ', 'd'))
    return ''.join(char for char in value if not unicodedata.combining(char))


def citation_key(value):
    """Return (article, document, year), or None if a required field is absent."""
    if not isinstance(value, str):
        return None
    text = _fold(value)
    article = re.search(r'\bdieu\s+(\d+[a-z]?)\b', text)
    if article is None:
        return None
    tail = text[article.end():]
    year = re.search(r'\b(?:19|20)\d{2}\b', tail)
    if year is None:
        return None
    document = tail[:year.start()] + ' ' + tail[year.end():]
    document = re.sub(r'^\s*(?:cua|tai|theo)\b', '', document)
    document = re.sub(r'\b(?:so|nam)\b', ' ', document)
    document = re.sub(r'[^a-z0-9]+', ' ', document).strip()
    if not document:
        return None
    return article.group(1), document, year.group(0)


def score_submission(rule_ids, expected_answer):
    """Count each distinct article/document/year match once; ignore incomplete gold."""
    if (not isinstance(rule_ids, list) or len(rule_ids) > 100 or
            any(not isinstance(item, str) for item in rule_ids)):
        raise ValueError('rule_ids must be a list of at most 100 citation strings')
    if not isinstance(expected_answer, list) or any(not isinstance(x, str) for x in expected_answer):
        raise ValueError('expected_answer must be a list of citation strings')
    submitted = list(dict.fromkeys(x.strip() for x in rule_ids if x.strip()))
    gold = {key for item in expected_answer if (key := citation_key(item)) is not None}
    ignored = sum(citation_key(item) is None for item in expected_answer)
    predicted = {citation_key(item) or ('invalid', item) for item in submitted}
    correct = len(gold & predicted)
    scored = bool(gold)
    precision = correct / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = correct / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'submitted_rule_ids': submitted, 'scored': scored,
            'ignored_expected_count': ignored, 'true_positives': correct,
            'predicted_count': len(predicted), 'gold_count': len(gold),
            'precision': precision if scored else None,
            'recall': recall if scored else None, 'f1': f1 if scored else None}
