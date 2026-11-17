"""Prepare the final Neo4j parts export and score article-level citations."""

import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from .corpus import rows
from .vectors import file_sha256


def prepare_final(source, output_dir, embedder, backend, model=None, batch_size=32):
    """Preserve each input object, append emb_vector(title), and index it."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / 'rules.jsonl'
    if Path(source).resolve() == destination.resolve():
        raise ValueError('Output must differ from input')
    (output_dir / 'manifest.json').unlink(missing_ok=True)
    temporary = output_dir / 'rules.jsonl.tmp'
    index = None
    identifiers, cache, batch = set(), {}, []
    count = 0

    def flush(stream):
        nonlocal index, count
        if not batch:
            return
        titles = list(dict.fromkeys(row['title'] for row in batch if row['title'] not in cache))
        if titles:
            encoded = np.asarray(embedder.embed(titles), dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(titles):
                raise ValueError('Embedding batch has an invalid shape')
            for title, vector in zip(titles, encoded):
                cache[title] = vector
        vectors = np.asarray([cache[row['title']] for row in batch], dtype=np.float32)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError('Embedding contains nonfinite or zero vectors')
        vectors = np.ascontiguousarray(vectors)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != index.d:
            raise ValueError('Embedding dimension changed')
        normalized = vectors.copy()
        faiss.normalize_L2(normalized)
        index.add(normalized)
        for row, vector in zip(batch, vectors):
            stream.write(json.dumps({**row, 'emb_vector': vector.tolist()}, ensure_ascii=False) + '\n')
            count += 1
        batch.clear()

    try:
        with temporary.open('w', encoding='utf-8') as output:
            for row in tqdm(rows(source), desc='Embedding legal titles', unit='part'):
                if (not isinstance(row.get('identifier'), str) or not row['identifier'].strip()
                        or not isinstance(row.get('title'), str) or not row['title'].strip()):
                    raise ValueError('Every part needs a nonempty identifier and title')
                if 'emb_vector' in row:
                    raise ValueError('Input already contains emb_vector')
                if row['identifier'] in identifiers:
                    raise ValueError('Duplicate identifier: ' + row['identifier'])
                identifiers.add(row['identifier'])
                batch.append(row)
                if len(batch) >= batch_size:
                    flush(output)
            flush(output)
        if index is None:
            raise ValueError('No legal parts in source')
        index_tmp = output_dir / 'index.faiss.tmp'
        faiss.write_index(index, str(index_tmp))
        temporary.replace(destination)
        index_tmp.replace(output_dir / 'index.faiss')
        manifest = {'version': 2, 'backend': backend, 'embedding_model': model,
                    'embedding_field': 'title', 'dimension': index.d, 'rules': count,
                    'rules_sha256': file_sha256(destination)}
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


class FinalRuleDatabase:
    def __init__(self, directory, embedder, max_rules=None):
        if max_rules is not None and (type(max_rules) is not int or max_rules < 1):
            raise ValueError('max_rules must be a positive integer')
        directory = Path(directory)
        self.path = directory / 'rules.jsonl'
        self.source = self.path.open('rb')
        try:
            self.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
            if self.manifest.get('version') != 2:
                raise ValueError('Final corpus changed or has an unsupported manifest')
            partial = max_rules is not None and max_rules < self.manifest['rules']
            limit = max_rules if partial else self.manifest['rules']
            self.offsets, self.id_to_row = [], {}
            digest = hashlib.sha256() if not partial else None
            vectors = []
            if partial:
                self.index = faiss.IndexFlatIP(self.manifest['dimension'])
                print(f'Debug subset: loading first {limit} of {self.manifest["rules"]} legal parts; '
                      'retrieval is limited to these parts and the full-file checksum is skipped.',
                      flush=True)
            else:
                print(f'Loading {self.manifest["rules"]} legal parts from {self.path}...', flush=True)

            def add_vectors():
                if not vectors:
                    return
                matrix = np.asarray(vectors, dtype=np.float32)
                if (matrix.shape != (len(vectors), self.index.d) or
                        not np.isfinite(matrix).all() or
                        np.any(np.linalg.norm(matrix, axis=1) == 0)):
                    raise ValueError('Invalid stored embedding in debug subset')
                matrix = np.ascontiguousarray(matrix)
                faiss.normalize_L2(matrix)
                self.index.add(matrix)
                vectors.clear()

            with tqdm(total=limit, desc='Loading legal parts', unit='part') as progress:
                while not partial or len(self.offsets) < limit:
                    line = self.source.readline()
                    if not line:
                        break
                    if digest is not None:
                        digest.update(line)
                    offset = self.source.tell() - len(line)
                    row = json.loads(line)
                    identifier = row['identifier']
                    if identifier in self.id_to_row:
                        raise ValueError('Duplicate identifier: ' + identifier)
                    self.id_to_row[identifier] = len(self.offsets)
                    self.offsets.append(offset)
                    if partial:
                        vectors.append(row['emb_vector'])
                        if len(vectors) == 512:
                            add_vectors()
                    progress.update(1)
            if partial:
                add_vectors()
            else:
                if digest.hexdigest() != self.manifest['rules_sha256']:
                    raise ValueError('Final corpus changed or has an unsupported manifest')
                print('Loading FAISS index...', flush=True)
                self.index = faiss.read_index(str(directory / 'index.faiss'))
            if (len(self.offsets) != limit or len(self.offsets) != self.index.ntotal or
                    self.index.d != self.manifest['dimension']):
                raise ValueError('Final corpus and vector index do not align')
            self.embedder = embedder
            print(f'Loaded {len(self.offsets)} legal parts with {self.index.d}-dimensional embeddings.')
        except Exception:
            self.source.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.source.close()

    def by_row(self, position):
        self.source.seek(self.offsets[position])
        return json.loads(self.source.readline())

    def get(self, rule_id):
        position = self.id_to_row.get(rule_id)
        if position is None:
            return None
        row = self.by_row(position)
        row.pop('emb_vector', None)  # Do not spend model context on raw floats.
        return {'rule_id': row['identifier'], **row}

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError('query must be nonempty text of at most 4096 characters')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('top_k must be between 1 and 20')
        vector = np.asarray(self.embedder.embed([query]), dtype=np.float32)
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError('Invalid query embedding')
        faiss.normalize_L2(vector)
        scores, positions = self.index.search(np.ascontiguousarray(vector), min(top_k, self.index.ntotal))
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            row = self.by_row(int(position))
            results.append({'rule_id': row['identifier'], 'title': row['title'],
                            'part_type': row.get('partType'),
                            'effective_date': row.get('effectiveDate'),
                            'expiration_date': row.get('expirationDate'),
                            'score': float(score),
                            'preview': str(row.get('content') or '')[:320]})
        return results

    def check_validity(self, rule_id, start, end):
        row = self.get(rule_id)
        if row is None:
            raise ValueError('Unknown rule_id')
        try:
            left, right = date.fromisoformat(start), date.fromisoformat(end)
        except (TypeError, ValueError) as exc:
            raise ValueError('start and end must be ISO YYYY-MM-DD dates') from exc
        if left > right:
            raise ValueError('start must be on or before end')
        raw_effective = row.get('effectiveDate', row.get('effective_date'))
        raw_expiration = row.get('expirationDate', row.get('expiration_date'))
        def parsed(value):
            try:
                return date.fromisoformat(value) if isinstance(value, str) else None
            except ValueError:
                return None
        effective, expiration = parsed(raw_effective), parsed(raw_expiration)
        if ((raw_effective is not None and effective is None) or
                (raw_expiration is not None and expiration is None)):
            status = 'unknown'
            note = 'Source date is not ISO-formatted; its integer encoding is unknown.'
        elif effective and right < effective or expiration and left >= expiration:
            status, note = 'invalid', 'Document-level dates only.'
        elif effective and left < effective <= right or expiration and left < expiration <= right:
            status, note = 'partially_valid', 'Document-level dates only.'
        elif effective and effective <= left and (expiration is None or right < expiration):
            status, note = 'valid', 'Document-level dates only; missing expiration is treated as open-ended.'
        else:
            status, note = 'unknown', 'Effective date is unavailable.'
        return {'rule_id': rule_id, 'start': start, 'end': end, 'status': status,
                'effective_date': raw_effective, 'expiration_date': raw_expiration, 'note': note}


def fold(value):
    value = unicodedata.normalize('NFKD', str(value).casefold().replace('đ', 'd'))
    return re.sub(r'\s+', ' ', ''.join(c if c.isalnum() else ' ' for c in value
                                      if not unicodedata.combining(c))).strip()


def article(value):
    match = re.search(r'\bdieu\s+(\d+[a-z]?)\b', fold(value))
    return match.group(1) if match else None


def citation_info(citation, fallback_year=None):
    raw = fold(citation)
    match = re.search(r'\bdieu\s+(\d+[a-z]?)\b', raw)
    if not match:
        return None
    document = raw[match.end():].strip()
    document = re.sub(r'^(cua|tai|theo)\s+', '', document)
    years = re.findall(r'\b(?:19|20)\d{2}\b', document)
    year = years[0] if years else (str(fallback_year) if fallback_year else None)
    document = re.sub(r'\b(?:nam\s+)?(?:19|20)\d{2}\b', '', document)
    document = re.sub(r'\bso\b', '', document)
    document = re.sub(r'\s+', ' ', document).strip()
    if not document or not year:
        return None
    return match.group(1), document, year


def gold_citations(laws):
    if not isinstance(laws, list):
        raise ValueError('laws must be a list')
    gold, unscorable = [], []
    for law in laws:
        if not isinstance(law, dict):
            raise ValueError('Each laws item must be an object')
        for provision in law.get('provisions') or []:
            if not isinstance(provision, dict):
                raise ValueError('Each provision must be an object')
            citation = provision.get('citation_str')
            info = citation_info(citation, law.get('year')) if isinstance(citation, str) else None
            if info is None:
                unscorable.append(citation)
            elif info not in [item[0] for item in gold]:
                gold.append((info, citation))
    return gold, unscorable


def score_submission(identifiers, laws, database):
    """One-to-one matching of article + document designation + cited year."""
    if not isinstance(identifiers, list) or len(identifiers) > 100 or any(not isinstance(x, str) for x in identifiers):
        raise ValueError('identifiers must be a list of at most 100 strings')
    gold, unscorable = gold_citations(laws)
    predicted, unmatched, id_to_key = [], [], {}
    for identifier in dict.fromkeys(identifiers):
        row = database.get(identifier)
        if row is None:
            unmatched.append(identifier)
            continue
        number = article(identifier)
        title = fold(row['title'])
        year_match = re.search(r'\b(?:19|20)\d{2}\b', title) or re.search(r'\b(?:19|20)\d{2}\b', identifier)
        year = year_match.group() if year_match else None
        title = re.sub(r'\b(?:nam\s+)?(?:19|20)\d{2}\b', '', title)
        title = re.sub(r'\bso\b', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        key = (number, title, year)
        id_to_key[identifier] = key
        if key not in [item[0] for item in predicted]:
            predicted.append((key, identifier))
    matched_gold, matched_keys = set(), set()
    for (number, title, year), identifier in predicted:
        for i, ((g_number, g_document, g_year), citation) in enumerate(gold):
            if (i not in matched_gold and number == g_number and year == g_year
                    and (title.startswith(g_document + ' ') or title == g_document
                         or (' ' + g_document + ' ') in (' ' + title + ' '))):
                matched_gold.add(i)
                matched_keys.add((number, title, year))
                break
    tp = len(matched_gold)
    predicted_count = len(predicted) + len(unmatched)
    gold_count = len(gold)
    precision = tp / predicted_count if predicted_count else (1.0 if not gold_count else 0.0)
    recall = tp / gold_count if gold_count else (1.0 if not predicted_count else 0.0)
    return {'submitted_rule_ids': list(dict.fromkeys(identifiers)),
            'true_positives': tp, 'predicted_count': predicted_count, 'gold_count': gold_count,
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'matched_rule_ids': [x for x in identifiers if id_to_key.get(x) in matched_keys],
            'unmatched_rule_ids': [x for x in identifiers if id_to_key.get(x) not in matched_keys],
            'unscorable_gold_count': len(unscorable)}
