"""Embed document titles from a Hugging Face dataset and publish retrieval files.

Data shards preserve the source fields without vectors. Each completed shard
has a temporary FAISS checkpoint; these are merged into one index and removed
after preparation, allowing interrupted runs to resume without re-embedding.
"""

import gzip
import itertools
import json
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from .vectors import file_sha256


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _shards(data_dir):
    paths = sorted(data_dir.glob('train-*.jsonl.gz'))
    for number, path in enumerate(paths):
        if path.name != f'train-{number:05d}.jsonl.gz':
            raise ValueError('Completed shard numbers must be contiguous from zero')
    return paths


def _paired_shards(output_dir):
    """Return committed data/index pairs and discard an interrupted last pair."""
    output_dir = Path(output_dir)
    data_paths = _shards(output_dir / 'data')
    checkpoint_dir = output_dir / 'checkpoints'
    index_paths = sorted(checkpoint_dir.glob('train-*.faiss'))
    for number, path in enumerate(index_paths):
        if path.name != f'train-{number:05d}.faiss':
            raise ValueError('Completed FAISS shard numbers must be contiguous from zero')
    committed = min(len(data_paths), len(index_paths))
    for path in data_paths[committed:] + index_paths[committed:]:
        path.unlink()
    for temporary in (output_dir / 'data').glob('train-*.jsonl.gz.tmp'):
        temporary.unlink()
    for temporary in checkpoint_dir.glob('train-*.faiss.tmp'):
        temporary.unlink()
    return list(zip(data_paths[:committed], index_paths[:committed]))


def _clear_checkpoints(output_dir):
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    for path in checkpoint_dir.glob('train-*.faiss'):
        path.unlink()
    if checkpoint_dir.is_dir():
        checkpoint_dir.rmdir()


def _rows_in_shard(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def _embedding_title(row):
    """Use the title when present; preserve the source row when it is missing."""
    title = row.get('title')
    if isinstance(title, str) and title.strip():
        return title.strip()
    metadata = [row[key].strip() for key in ('loai_van_ban', 'so_ky_hieu')
                if isinstance(row.get(key), str) and row[key].strip()]
    if metadata:
        return ' '.join(metadata)
    content = row.get('full_content')
    if isinstance(content, str) and content.strip():
        return ' '.join(content.split())[:250]
    return f"Văn bản pháp luật {row['id']}"


def prepare_document_rows(dataset, output_dir, embedder, *, source_repo,
                          source_revision, split='train', embedding_model,
                          batch_size=8, shard_rows=1000, vector_dim=4096,
                          max_rows=None):
    """Prepare a streamed source dataset using an OpenAI embedding client."""
    if batch_size < 1 or shard_rows < 1 or vector_dim < 1:
        raise ValueError('batch_size, shard_rows, and vector_dim must be positive')
    if max_rows is not None and max_rows < 1:
        raise ValueError('max_rows must be positive')
    output_dir = Path(output_dir)
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {'format_version': 4, 'vector_storage': 'faiss_only',
              'source_repo': source_repo, 'source_revision': source_revision,
              'split': split, 'embedding_model': embedding_model,
              'embedding_endpoint': 'v1/embeddings', 'document_prefix': 'passage: ',
              'query_prefix': 'query: ', 'dimension': vector_dim,
              'shard_rows': shard_rows, 'max_rows': max_rows}
    config_path = output_dir / 'prepare_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Preparation settings changed; use a new output directory')
    elif _shards(data_dir):
        raise ValueError('Data shards exist without prepare_config.json')
    else:
        _write_json(config_path, config)

    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (output_dir / 'index.faiss').is_file() and (output_dir / 'id_map.jsonl').is_file():
            _clear_checkpoints(output_dir)
            return manifest
        manifest_path.unlink()

    pairs = _paired_shards(output_dir)
    seen, completed = set(), 0
    for path, index_path in pairs:
        part = faiss.read_index(str(index_path))
        if part.d != vector_dim:
            raise ValueError(f'Wrong FAISS shard dimension: {index_path}')
        shard_count = 0
        for row in _rows_in_shard(path):
            if 'emb_vector' in row:
                raise ValueError(f'Unexpected embedding in data shard: {path}')
            identifier = str(row['id'])
            if identifier in seen:
                raise ValueError(f'Duplicate source id in completed shards: {identifier}')
            seen.add(identifier)
            completed += 1
            shard_count += 1
        if part.ntotal != shard_count:
            raise ValueError(f'Data and FAISS shard have different row counts: {path}')
    if max_rows is not None and completed > max_rows:
        raise ValueError('Completed shards exceed max_rows')
    print(f'Resuming after {completed} embedded documents in {len(pairs)} shard(s).', flush=True)

    iterator = dataset.skip(completed) if hasattr(dataset, 'skip') else itertools.islice(dataset, completed, None)
    shard_number, shard_count, total = len(pairs), 0, completed
    shard_stream, temporary_path = None, None
    pending, shard_vectors = [], []

    def finish_shard():
        nonlocal shard_number, shard_count, shard_stream, temporary_path
        if shard_stream is None:
            return
        shard_stream.close()
        shard_stream = None
        part = faiss.IndexFlatIP(vector_dim)
        part.add(np.ascontiguousarray(np.vstack(shard_vectors), dtype=np.float32))
        checkpoint_tmp = checkpoint_dir / f'train-{shard_number:05d}.faiss.tmp'
        faiss.write_index(part, str(checkpoint_tmp))
        temporary_path.replace(data_dir / f'train-{shard_number:05d}.jsonl.gz')
        checkpoint_tmp.replace(checkpoint_dir / f'train-{shard_number:05d}.faiss')
        shard_number += 1
        shard_vectors.clear()
        shard_count, shard_stream, temporary_path = 0, None, None

    def flush_batch():
        nonlocal shard_stream, temporary_path, shard_count, total
        if not pending:
            return
        vectors = np.asarray(embedder.embed(['passage: ' + _embedding_title(row)
                                             for row in pending]),
                             dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(pending) or vectors.shape[1] < vector_dim:
            raise ValueError('Embedding response has an invalid shape')
        vectors = np.ascontiguousarray(vectors[:, :vector_dim])
        if not np.isfinite(vectors).all() or np.any(np.sum(vectors * vectors, axis=1) == 0):
            raise ValueError('Embedding response contains a nonfinite or zero vector')
        faiss.normalize_L2(vectors)
        for row, vector in zip(pending, vectors):
            if shard_stream is None:
                temporary_path = data_dir / f'train-{shard_number:05d}.jsonl.gz.tmp'
                shard_stream = gzip.open(temporary_path, 'wt', encoding='utf-8')
            shard_stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            shard_vectors.append(vector.copy())
            shard_count += 1
            total += 1
            if shard_count == shard_rows:
                finish_shard()
        pending.clear()

    try:
        with tqdm(desc='Embedding document titles', unit='doc', initial=completed) as progress:
            for row in iterator:
                if max_rows is not None and total + len(pending) >= max_rows:
                    break
                if not isinstance(row, dict):
                    raise ValueError('Each dataset row must be an object')
                if not isinstance(row.get('id'), (str, int)) or not str(row['id']).strip():
                    raise ValueError('Each row needs a nonempty id')
                if 'emb_vector' in row:
                    raise ValueError('Source row already contains emb_vector')
                identifier = str(row['id'])
                if identifier in seen:
                    raise ValueError(f'Duplicate source id: {identifier}')
                seen.add(identifier)
                pending.append(row)
                if len(pending) >= batch_size:
                    flush_batch()
                    progress.update(batch_size)
            remaining = len(pending)
            flush_batch()
            progress.update(remaining)
        finish_shard()
    finally:
        if shard_stream is not None:
            shard_stream.close()

    if total == 0:
        raise ValueError('Source dataset contains no documents')
    manifest = build_document_index(output_dir, config, total)
    return manifest


def build_document_index(output_dir, config, expected_rows):
    """Merge resumable FAISS shards in the same order as the data rows."""
    output_dir = Path(output_dir)
    index = faiss.IndexFlatIP(config['dimension'])
    index_tmp = output_dir / 'index.faiss.tmp'
    id_tmp = output_dir / 'id_map.jsonl.tmp'
    count, fallback_title_rows = 0, 0

    with id_tmp.open('w', encoding='utf-8') as ids, \
            tqdm(total=expected_rows, desc='Building FAISS index', unit='doc') as progress:
        for path, index_path in _paired_shards(output_dir):
            part = faiss.read_index(str(index_path))
            if part.d != index.d or part.ntotal == 0:
                raise ValueError(f'Invalid FAISS shard: {index_path}')
            index.add(np.ascontiguousarray(part.reconstruct_n(0, part.ntotal),
                                           dtype=np.float32))
            shard_count = 0
            for row in _rows_in_shard(path):
                ids.write(json.dumps({'id': row['id'], 'title': row.get('title')},
                                     ensure_ascii=False) + '\n')
                if not isinstance(row.get('title'), str) or not row['title'].strip():
                    fallback_title_rows += 1
                count += 1
                shard_count += 1
                progress.update(1)
            if shard_count != part.ntotal:
                raise ValueError(f'Data and FAISS shard have different row counts: {path}')
    if count != expected_rows or index.ntotal != expected_rows:
        raise ValueError('Data shards and FAISS index do not align')
    faiss.write_index(index, str(index_tmp))
    index_tmp.replace(output_dir / 'index.faiss')
    id_tmp.replace(output_dir / 'id_map.jsonl')
    manifest = {'version': 4, **config, 'rows': count,
                'fallback_title_rows': fallback_title_rows,
                'index_file': 'index.faiss', 'id_map_file': 'id_map.jsonl',
                'index_sha256': file_sha256(output_dir / 'index.faiss')}
    card = ('---\nconfigs:\n- config_name: default\n  data_files:\n'
            '  - split: train\n    path: "data/train-*.jsonl.gz"\n---\n\n'
            '# Vietnamese legal corpus with title embeddings\n\n'
            f'Source: {config["source_repo"]} at revision {config["source_revision"]}.\n\n'
            'Every data row preserves the source fields without embedding floats. '
            'Title vectors are stored only in `index.faiss`. '
            '`index.faiss` and `id_map.jsonl` use the same row order. '
            'Documents were embedded with the `passage: ` prefix; embed queries '
            'with `query: ` using the same model and vector dimension.\n')
    (output_dir / 'README.md').write_text(card, encoding='utf-8')
    _write_json(output_dir / 'manifest.json', manifest)
    _clear_checkpoints(output_dir)
    return manifest


def upload_document_corpus(output_dir, repo_id, *, token=None, private=False):
    """Upload only completed dataset/index artifacts to a dataset repository."""
    from huggingface_hub import HfApi

    output_dir = Path(output_dir)
    if not (output_dir / 'manifest.json').is_file():
        raise ValueError('Preparation must finish before upload')
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type='dataset', private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id, repo_type='dataset', folder_path=str(output_dir),
        allow_patterns=['README.md', 'manifest.json', 'index.faiss',
                        'id_map.jsonl', 'data/*.jsonl.gz'],
        commit_message='Add title embeddings and FAISS index')


def search_document_index(output_dir, embedder, query, top_k=5):
    """Small retrieval smoke test for the prepared document-level index."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be nonempty text')
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError('top_k must be an integer between 1 and 100')
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 4:
        raise ValueError('Expected a prepare-hf-documents corpus')
    index = faiss.read_index(str(output_dir / 'index.faiss'))
    if index.ntotal != manifest['rows'] or index.d != manifest['dimension']:
        raise ValueError('FAISS index does not align with manifest')
    vector = np.asarray(embedder.embed(['query: ' + query]), dtype=np.float32)
    if vector.ndim != 2 or vector.shape[0] != 1 or vector.shape[1] < index.d:
        raise ValueError('Query embedding has an invalid shape')
    vector = np.ascontiguousarray(vector[:, :index.d])
    if not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError('Query embedding is nonfinite or zero')
    faiss.normalize_L2(vector)
    scores, positions = index.search(vector, min(top_k, index.ntotal))
    wanted = set(int(position) for position in positions[0] if position >= 0)
    records = {}
    with (output_dir / 'id_map.jsonl').open(encoding='utf-8') as stream:
        for position, line in enumerate(stream):
            if position in wanted:
                records[position] = json.loads(line)
    if len(records) != len(wanted):
        raise ValueError('ID map does not align with FAISS index')
    return [{'id': records[int(position)]['id'], 'title': records[int(position)]['title'],
             'score': float(score)}
            for score, position in zip(scores[0], positions[0]) if position >= 0]
