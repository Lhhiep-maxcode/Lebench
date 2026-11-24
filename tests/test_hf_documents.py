"""Document-level Hugging Face corpus preparation and resume behavior."""

import gzip
import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from legal_rule_retrieval.hf_documents import (
    prepare_document_rows, search_document_index, upload_document_corpus,
)


class FakeEmbedder:
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def embed(self, texts):
        self.calls.append(texts)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError('interrupted embedding')
        return np.asarray([[len(text), len(text) % 7 + 1, 3, 4]
                           for text in texts], dtype=np.float32)


class HuggingFaceDocumentTests(unittest.TestCase):
    def test_resume_preserves_fields_and_aligns_index(self):
        rows = [
            {'id': str(i), 'title': f'Luật số {i}', 'so_ky_hieu': f'{i}/2020/QH',
             'full_content': f'Nội dung {i}', 'relationships': [{'target_id': '9',
                                                                'relationship_type': 'Căn cứ'}]}
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            options = {'source_repo': 'example/legal', 'source_revision': 'abc123',
                       'embedding_model': 'model', 'batch_size': 2,
                       'shard_rows': 2, 'vector_dim': 2}
            with self.assertRaisesRegex(RuntimeError, 'interrupted embedding'):
                prepare_document_rows(rows, output, FakeEmbedder(fail_after=1), **options)
            self.assertEqual(len(list((output / 'data').glob('train-*.jsonl.gz'))), 1)
            self.assertEqual(len(list((output / 'checkpoints').glob('train-*.faiss'))), 1)
            # Simulate a crash after a data shard is renamed but before its
            # matching FAISS checkpoint is committed.
            with gzip.open(output / 'data' / 'train-00001.jsonl.gz', 'wt',
                           encoding='utf-8') as stream:
                stream.write(json.dumps({'id': 'orphan', 'title': 'unfinished'}) + '\n')

            resumed = FakeEmbedder()
            manifest = prepare_document_rows(rows, output, resumed, **options)
            self.assertEqual(resumed.calls, [['passage: Luật số 2']])
            self.assertEqual((manifest['version'], manifest['rows'], manifest['dimension']),
                             (4, 3, 2))
            index = faiss.read_index(str(output / 'index.faiss'))
            self.assertEqual(index.ntotal, 3)
            self.assertTrue(np.allclose(np.linalg.norm(index.reconstruct_n(0, 3), axis=1), 1))
            self.assertFalse((output / 'checkpoints').exists())
            data = []
            for path in sorted((output / 'data').glob('train-*.jsonl.gz')):
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    data.extend(json.loads(line) for line in stream)
            self.assertEqual(data, rows)
            ids = [json.loads(line)['id'] for line in
                   (output / 'id_map.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(ids, ['0', '1', '2'])
            query_embedder = FakeEmbedder()
            hits = search_document_index(output, query_embedder, 'đất đai', top_k=2)
            self.assertEqual(query_embedder.calls, [['query: đất đai']])
            self.assertEqual(len(hits), 2)
            self.assertTrue(all(hit['id'] in ids for hit in hits))
            try:
                from datasets import load_dataset
            except ImportError:
                pass
            else:
                loaded = load_dataset('json', data_files=str(output / 'data' / '*.jsonl.gz'),
                                      split='train')
                self.assertEqual(len(loaded), 3)
                self.assertEqual(loaded[0]['relationships'], rows[0]['relationships'])
                self.assertNotIn('emb_vector', loaded.column_names)
                del loaded
                gc.collect()

            reused = FakeEmbedder()
            self.assertEqual(prepare_document_rows(rows, output, reused, **options), manifest)
            self.assertEqual(reused.calls, [])

            with patch('huggingface_hub.HfApi') as api_class:
                upload_document_corpus(output, 'example/embedded-legal', token='test-token')
                api_class.assert_called_once_with(token='test-token')
                api_class.return_value.create_repo.assert_called_once_with(
                    'example/embedded-legal', repo_type='dataset', private=False,
                    exist_ok=True)
                upload_args = api_class.return_value.upload_folder.call_args.kwargs
                self.assertEqual(upload_args['repo_type'], 'dataset')
                self.assertIn('index.faiss', upload_args['allow_patterns'])
                self.assertIn('data/*.jsonl.gz', upload_args['allow_patterns'])
                self.assertFalse(any('checkpoints' in pattern for pattern in
                                     upload_args['allow_patterns']))

    def test_index_merge_can_resume_without_reembedding(self):
        rows = [{'id': str(i), 'title': f'Law {i}', 'full_content': f'Content {i}'}
                for i in range(3)]
        options = {'source_repo': 'example/legal', 'source_revision': 'abc123',
                   'embedding_model': 'model', 'batch_size': 2,
                   'shard_rows': 2, 'vector_dim': 2}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch('legal_rule_retrieval.hf_documents.build_document_index',
                       side_effect=RuntimeError('merge interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'merge interrupted'):
                    prepare_document_rows(rows, output, FakeEmbedder(), **options)
            self.assertEqual(len(list((output / 'checkpoints').glob('train-*.faiss'))), 2)
            resumed = FakeEmbedder()
            manifest = prepare_document_rows(rows, output, resumed, **options)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(manifest['rows'], 3)
            self.assertFalse((output / 'checkpoints').exists())


if __name__ == '__main__':
    unittest.main()
