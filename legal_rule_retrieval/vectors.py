"""Embedding providers and an exact cosine-similarity FAISS index."""

import hashlib
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from .corpus import RuleStore, rows


class APIEmbedder:
    """OpenAI-compatible /v1/embeddings endpoint (can be separate from chat)."""

    def __init__(self, base_url, model, api_key='dummy', timeout=120):
        if not base_url:
            raise ValueError('Embedding API requires --embedding-base-url')
        self.url = base_url.rstrip('/') + '/embeddings'
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts):
        payload = {'model': self.model, 'input': texts, 'encoding_format': 'float'}
        request = urllib.request.Request(
            self.url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Embedding HTTP {exc.code}: {exc.read(500).decode("utf-8", errors="replace")}') from exc
        ordered = sorted(result['data'], key=lambda item: item['index'])
        if len(ordered) != len(texts) or [item['index'] for item in ordered] != list(range(len(texts))):
            raise ValueError('Embedding server returned incomplete or misordered batch')
        return np.asarray([item['embedding'] for item in ordered], dtype=np.float32)


class LocalEmbedder:
    """Optional sentence-transformers backend, loaded only when selected."""

    def __init__(self, model):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError('Install sentence-transformers for --embedding-backend local') from exc
        self.model = SentenceTransformer(model)

    def embed(self, texts):
        return np.asarray(self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
                          dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words vectors for offline plumbing tests, not real retrieval."""

    def embed(self, texts):
        matrix = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            folded = unicodedata.normalize('NFKD', text.casefold().replace('đ', 'd'))
            folded = ''.join(char for char in folded if not unicodedata.combining(char))
            for word in re.findall(r'[a-z0-9]+', folded):
                digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
                matrix[row, int.from_bytes(digest, 'little') % 256] += 1
        return matrix


def make_embedder(backend, model=None, base_url=None, api_key='dummy', timeout=120):
    if backend == 'api':
        if not model:
            raise ValueError('Embedding API requires --embedding-model')
        return APIEmbedder(base_url, model, api_key, timeout)
    if backend == 'local':
        if not model:
            raise ValueError('Local embeddings require --embedding-model')
        return LocalEmbedder(model)
    if backend == 'hash':
        return HashEmbedder()
    raise ValueError(f'Unknown embedding backend: {backend}')


def chunks(text, size, overlap):
    start = 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            yield start, piece
        if start + size >= len(text):
            break
        start += size - overlap


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_vectors(rules_dir, index_dir, embedder, backend, model=None,
                  batch_size=32, chunk_chars=1800, overlap=200,
                  passage_prefix='', query_prefix=''):
    """Embed every rule chunk and save index plus aligned row/chunk mappings."""
    if batch_size < 1 or chunk_chars < 100 or not 0 <= overlap < chunk_chars:
        raise ValueError('Invalid batch_size, chunk_chars, or overlap')
    rules_path = Path(rules_dir) / 'rules.jsonl'
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    # A manifest is published last; an interrupted rebuild must not look complete.
    (index_dir / 'manifest.json').unlink(missing_ok=True)
    index = None
    rule_rows, chunk_starts = [], []
    texts, pending_rows, pending_starts = [], [], []
    rule_count = 0

    def flush():
        nonlocal index
        if not texts:
            return
        vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError('Embedding batch has the wrong shape')
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError('Embedding batch contains nonfinite or zero vectors')
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)
        if index is None:
            index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != index.d:
            raise ValueError('Embedding dimension changed between batches')
        index.add(vectors)
        rule_rows.extend(pending_rows)
        chunk_starts.extend(pending_starts)
        texts.clear(); pending_rows.clear(); pending_starts.clear()

    with tqdm(desc='Embedding provisions', unit='rule',
              disable=not sys.stderr.isatty()) as progress:
        for rule_count, rule in enumerate(rows(rules_path), 1):
            header = ' | '.join(str(rule.get(key) or '') for key in
                                ('document_code', 'title', 'name', 'rule_id'))
            for start, piece in chunks(rule['content'], chunk_chars, overlap):
                texts.append(passage_prefix + header + '\n' + piece)
                pending_rows.append(rule_count - 1)
                pending_starts.append(start)
                if len(texts) >= batch_size:
                    flush()
            progress.update(1)
    flush()
    if index is None:
        raise ValueError('Prepared corpus contains no embeddable rules')
    temporary = {name: index_dir / (name + '.tmp') for name in
                 ('index.faiss', 'rule_rows.npy', 'chunk_starts.npy', 'manifest.json')}
    faiss.write_index(index, str(temporary['index.faiss']))
    for name, data in (('rule_rows.npy', rule_rows), ('chunk_starts.npy', chunk_starts)):
        with temporary[name].open('wb') as output:
            np.save(output, np.asarray(data, dtype=np.int64))
    manifest = {'version': 1, 'backend': backend, 'embedding_model': model,
                'dimension': index.d, 'rules': rule_count, 'vectors': index.ntotal,
                'chunk_chars': chunk_chars, 'overlap': overlap,
                'passage_prefix': passage_prefix, 'query_prefix': query_prefix,
                'rules_sha256': file_sha256(rules_path)}
    temporary['manifest.json'].write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name in ('index.faiss', 'rule_rows.npy', 'chunk_starts.npy', 'manifest.json'):
        temporary[name].replace(index_dir / name)
    return manifest


class VectorRuleDatabase:
    """Same get/validity interface as the corpus, plus embedding search."""

    def __init__(self, rules_dir, index_dir, embedder):
        self.store = RuleStore(Path(rules_dir) / 'rules.jsonl')
        try:
            index_dir = Path(index_dir)
            self.manifest = json.loads((index_dir / 'manifest.json').read_text(encoding='utf-8'))
            if self.manifest.get('version') != 1:
                raise ValueError('Unsupported vector index manifest version')
            if self.manifest['rules_sha256'] != file_sha256(self.store.path):
                raise ValueError('Prepared rules changed since embedding; rebuild vectors')
            self.index = faiss.read_index(str(index_dir / 'index.faiss'))
            self.rule_rows = np.load(index_dir / 'rule_rows.npy')
            self.chunk_starts = np.load(index_dir / 'chunk_starts.npy')
            self.embedder = embedder
            if (len(self.store) != self.manifest['rules'] or
                    self.index.ntotal != self.manifest['vectors'] or
                    self.index.ntotal != len(self.rule_rows) or
                    self.index.ntotal != len(self.chunk_starts)):
                raise ValueError('Index, mappings, and prepared rules do not align')
        except Exception:
            self.store.close()
            raise

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get(self, rule_id):
        return self.store.get(rule_id)

    def check_validity(self, rule_id, start, end):
        return self.store.check_validity(rule_id, start, end)

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('query must be nonempty text')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('top_k must be an integer between 1 and 20')
        vector = np.asarray(self.embedder.embed([self.manifest.get('query_prefix', '') + query]),
                            dtype=np.float32)
        if vector.shape != (1, self.index.d) or not np.isfinite(vector).all():
            raise ValueError('Query embedding dimension differs from index')
        if np.linalg.norm(vector) == 0:
            raise ValueError('Query embedding is zero')
        vector = np.ascontiguousarray(vector)
        faiss.normalize_L2(vector)
        count = min(self.index.ntotal, max(20, top_k * 4))
        results, seen = [], set()
        while True:
            scores, positions = self.index.search(vector, count)
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                row_number = int(self.rule_rows[position])
                if row_number in seen:
                    continue
                seen.add(row_number)
                rule = self.store.by_row(row_number)
                start = int(self.chunk_starts[position])
                results.append({'rule_id': rule['rule_id'], 'name': rule['name'],
                                'part_type': rule['part_type'],
                                'document_title': rule['title'],
                                'document_code': rule['document_code'],
                                'field': rule.get('field'),
                                'effective_date': rule['effective_date'],
                                'expiration_date': rule['expiration_date'],
                                'score': float(score),
                                'preview': rule['content'][start:start + 320]})
                if len(results) == top_k:
                    return results
            if count == self.index.ntotal:
                return results
            count = min(self.index.ntotal, count * 2)
