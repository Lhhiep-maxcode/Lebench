"""Read the published document corpus and search its title-only FAISS index."""

import gzip
import itertools
import json
from datetime import date, datetime
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


DOWNLOAD_COMMAND = ('hf download Hiepppp/vietnamese-legal-corpus --repo-type dataset '
                    '--local-dir input/vietnamese-legal-corpus')


def require_corpus(directory):
    """Use a local snapshot; never download dataset files during a benchmark run."""
    directory = Path(directory)
    required = [directory / name for name in ('manifest.json', 'index.faiss', 'id_map.jsonl')]
    if any(not path.is_file() for path in required) or not list((directory / 'data').glob('train-*.jsonl.gz')):
        raise FileNotFoundError(
            f'Dataset snapshot missing or incomplete at {directory}. Download it first:\n'
            f'  {DOWNLOAD_COMMAND}')
    return directory


def _date(value):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError('Invalid source date')
    for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError('Invalid source date')


class HFDocumentDatabase:
    def __init__(self, directory, embedder):
        self.directory = require_corpus(directory)
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('version') != 4 or self.manifest.get('vector_storage') != 'faiss_only':
            raise ValueError('Expected a version-4, FAISS-only document corpus')
        self.embedder = embedder
        self.shard_rows = self.manifest['shard_rows']
        self.ids, self.titles, self.positions = [], [], {}
        with (self.directory / 'id_map.jsonl').open(encoding='utf-8') as stream:
            for position, line in enumerate(tqdm(stream, total=self.manifest['rows'],
                                                  desc='Loading document IDs', unit='doc')):
                item = json.loads(line)
                identifier = str(item['id'])
                if identifier in self.positions:
                    raise ValueError(f'Duplicate document id: {identifier}')
                self.positions[identifier] = position
                self.ids.append(identifier)
                self.titles.append(item.get('title'))
        if len(self.ids) != self.manifest['rows']:
            raise ValueError('ID map row count differs from manifest')
        shard_count = (len(self.ids) + self.shard_rows - 1) // self.shard_rows
        if any(not (self.directory / 'data' / f'train-{i:05d}.jsonl.gz').is_file()
               for i in range(shard_count)):
            raise ValueError('Dataset shards are missing')
        self.index = faiss.read_index(str(self.directory / 'index.faiss'))
        if self.index.ntotal != len(self.ids) or self.index.d != self.manifest['dimension']:
            raise ValueError('FAISS index, ID map, and manifest do not align')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self.index

    def _row(self, rule_id):
        position = self.positions.get(str(rule_id))
        if position is None:
            raise ValueError('Unknown rule_id')
        shard, offset = divmod(position, self.shard_rows)
        path = self.directory / 'data' / f'train-{shard:05d}.jsonl.gz'
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            line = next(itertools.islice(stream, offset, offset + 1), None)
        if line is None:
            raise ValueError(f'Data shard is shorter than ID map: {path}')
        row = json.loads(line)
        if str(row.get('id')) != self.ids[position]:
            raise ValueError(f'Data shard and ID map disagree at position {position}')
        return row

    def search(self, query, top_k=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError('query must be nonempty text of at most 4096 characters')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('top_k must be between 1 and 20')
        prefix = self.manifest.get('query_prefix', 'query: ')
        vector = np.asarray(self.embedder.embed([prefix + query]), dtype=np.float32)
        if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] < self.index.d:
            raise ValueError('Invalid query embedding shape')
        vector = np.ascontiguousarray(vector[:, :self.index.d])
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError('Invalid query embedding values')
        faiss.normalize_L2(vector)
        _, positions = self.index.search(vector, min(top_k, self.index.ntotal))
        return [{'id': self.ids[int(pos)], 'title': self.titles[int(pos)]}
                for pos in positions[0] if pos >= 0]

    def get(self, rule_id):
        row = self._row(rule_id)
        return {'rule_id': str(row['id']), 'full_content': row.get('full_content')}

    def check_relationship(self, rule_id):
        row = self._row(rule_id)
        return {'rule_id': str(row['id']), 'relationships': row.get('relationships')}

    def check_validity(self, rule_id, start, end):
        row = self._row(rule_id)
        try:
            left, right = date.fromisoformat(start), date.fromisoformat(end)
        except (TypeError, ValueError) as exc:
            raise ValueError('start and end must be ISO YYYY-MM-DD dates') from exc
        if left > right:
            raise ValueError('start must be on or before end')
        raw_effective, raw_expiration = row.get('ngay_co_hieu_luc'), row.get('ngay_het_hieu_luc')
        try:
            effective, expiration = _date(raw_effective), _date(raw_expiration)
        except ValueError:
            effective = expiration = None
            status = 'unknown'
        else:
            if effective and expiration and expiration < effective:
                status = 'unknown'
            elif effective and right < effective or expiration and left >= expiration:
                status = 'invalid'
            elif effective and left < effective <= right or expiration and left < expiration <= right:
                status = 'partially_valid'
            elif effective and effective <= left and (expiration is None or right < expiration):
                status = 'valid'
            else:
                status = 'unknown'
        return {'rule_id': str(row['id']), 'start': start, 'end': end, 'status': status,
                'ngay_co_hieu_luc': raw_effective, 'ngay_het_hieu_luc': raw_expiration}
