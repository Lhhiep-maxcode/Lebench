"""Document-corpus retrieval, citation scoring, and tool loop smoke tests."""

import asyncio
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

from legal_rule_retrieval.citation_scoring import citation_key, score_submission
from legal_rule_retrieval.harness import TOOL_SPECS, ToolSession
from legal_rule_retrieval.hf_database import HFDocumentDatabase, require_corpus
from legal_rule_retrieval.python_runner import TOOLS, run_task


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.extend(texts)
        return np.asarray([[1, 0, 0, 0] for _ in texts], dtype=np.float32)


class FakeChat:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []
        self.url = 'http://localhost/v1/chat/completions'

    def complete(self, messages):
        self.messages.append(messages.copy())
        name, arguments = next(self.actions)
        return {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'call_' + str(len(self.messages)), 'type': 'function',
             'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


def make_corpus(directory):
    rows = [
        {'id': '10', 'title': 'Luật Tố tụng hành chính 2010',
         'full_content': 'Điều 114. Thời hiệu...',
         'ngay_co_hieu_luc': '01/07/2011', 'ngay_het_hieu_luc': '01/07/2016',
         'relationships': [{'target_id': '11', 'target_title': 'Nghị quyết',
                            'relationship_type': 'Căn cứ'}]},
        {'id': '11', 'title': 'Nghị quyết 326/2016/UBTVQH14',
         'full_content': 'Điều 26. Án phí...',
         'ngay_co_hieu_luc': '01/01/2017', 'ngay_het_hieu_luc': None,
         'relationships': []},
    ]
    (directory / 'data').mkdir()
    with gzip.open(directory / 'data' / 'train-00000.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (directory / 'id_map.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({'id': row['id'], 'title': row['title']},
                                    ensure_ascii=False) + '\n')
    index = faiss.IndexFlatIP(4)
    index.add(np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    faiss.write_index(index, str(directory / 'index.faiss'))
    (directory / 'manifest.json').write_text(json.dumps({
        'version': 4, 'vector_storage': 'faiss_only', 'rows': 2,
        'shard_rows': 1000, 'dimension': 4,
        'embedding_model': 'fake', 'query_prefix': 'query: '}), encoding='utf-8')


class HFBenchmarkTests(unittest.TestCase):
    def test_document_mcp_tools(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            self.skipTest('Optional MCP client is not installed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / 'corpus'
            corpus.mkdir()
            make_corpus(corpus)
            gold = root / 'gold.json'
            gold.write_text(json.dumps(['Điều 114 Luật Tố tụng hành chính 2010'],
                                       ensure_ascii=False), encoding='utf-8')
            submission = root / 'submission.json'

            async def check():
                params = StdioServerParameters(
                    command=sys.executable,
                    args=['-m', 'legal_rule_retrieval.mcp_server',
                          '--corpus-dir', str(corpus), '--gold-path', str(gold),
                          '--workspace-root', str(root / 'work'), '--task-id', 'case',
                          '--submission-path', str(submission),
                          '--embedding-base-url', 'http://127.0.0.1:9/v1'],
                    cwd=str(Path(__file__).resolve().parent.parent))
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        advertised = {tool.name: tool for tool in (await session.list_tools()).tools}
                        self.assertEqual(set(advertised), {spec['name'] for spec in TOOL_SPECS})
                        for spec in TOOL_SPECS:
                            self.assertEqual(advertised[spec['name']].description,
                                             spec['description'])
                        result = await session.call_tool('get_legal_rule', {'rule_id': '10'})
                        self.assertFalse(result.isError)
                        self.assertIn('Điều 114', str(result.content))
                        result = await session.call_tool('check_relationship', {'rule_id': '10'})
                        self.assertFalse(result.isError)
                        result = await session.call_tool('submit_rules', {'rule_ids': [
                            'Điều 114 Luật Tố tụng hành chính 2010']})
                        self.assertFalse(result.isError)

            asyncio.run(check())
            self.assertEqual(json.loads(submission.read_text(encoding='utf-8'))['recall'], 1.0)

    def test_missing_corpus_requests_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'hf download'):
                require_corpus(Path(tmp) / 'absent')

    def test_active_tool_schemas_match_harness_specs(self):
        self.assertEqual([spec['name'] for spec in TOOL_SPECS],
                         [tool['function']['name'] for tool in TOOLS])
        self.assertEqual([spec['description'] for spec in TOOL_SPECS],
                         [tool['function']['description'] for tool in TOOLS])
        self.assertEqual([list(spec['arguments']) for spec in TOOL_SPECS],
                         [list(tool['function']['parameters']['properties']) for tool in TOOLS])
        self.assertIn('citation', next(spec['description'] for spec in TOOL_SPECS
                                       if spec['name'] == 'submit_rules'))

    def test_database_tools_and_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_corpus(directory)
            embedder = FakeEmbedder()
            with HFDocumentDatabase(directory, embedder) as db:
                hits = db.search('tố tụng hành chính', 1)
                self.assertEqual(hits, [{'id': '10', 'title': 'Luật Tố tụng hành chính 2010'}])
                self.assertEqual(embedder.calls, ['query: tố tụng hành chính'])
                self.assertEqual(db.get('10')['full_content'], 'Điều 114. Thời hiệu...')
                self.assertEqual(db.check_relationship('10')['relationships'][0]['target_id'], '11')
                self.assertEqual(db.check_validity('10', '2012-01-01', '2013-01-01')['status'], 'valid')
                self.assertEqual(db.check_validity('10', '2010-01-01', '2010-12-31')['status'], 'invalid')
                self.assertEqual(db.check_validity('10', '2016-06-01', '2016-08-01')['status'],
                                 'partially_valid')
                self.assertEqual(db.check_validity('10', '2016-07-01', '2017-01-01')['status'],
                                 'invalid')
                session = ToolSession(db, directory / 'work', 'test')
                self.assertEqual(session.dispatch('search_legal_rules',
                                 {'query': 'test', 'top_k': 1})['results'], hits)
                self.assertEqual(session.dispatch('get_legal_rule',
                                 {'rule_id': '10'})['full_content'], 'Điều 114. Thời hiệu...')
                self.assertEqual(session.dispatch('check_relationship',
                                 {'rule_id': '10'})['relationships'][0]['target_id'], '11')
                session.dispatch('create_file', {'path': 'notes.txt', 'content': 'A'})
                session.dispatch('append_file', {'path': 'notes.txt', 'content': 'B'})
                self.assertEqual(session.dispatch('read_file', {'path': 'notes.txt'})['content'], 'AB')
                with self.assertRaisesRegex(ValueError, 'does not exist'):
                    session.dispatch('append_file', {'path': 'missing.txt', 'content': 'B'})

    def test_article_scoring_and_gold_privacy(self):
        gold = ['khoản 2 Điều 114 Luật Tố tụng hành chính 2010',
                'Điều 26 Nghị quyết 326/2016/UBTVQH14',
                'Nghị quyết 326/2016/UBTVQH14']
        predicted = ['Điều 114 Luật Tố tụng hành chính 2010',
                     'điều 114 luật tố tụng hành chính 2010',
                     'Điều 26 Nghị quyết số 326/2016/UBTVQH14',
                     'Điều 9 Luật khác 2020']
        score = score_submission(predicted, gold)
        self.assertEqual((score['true_positives'], score['predicted_count'],
                          score['gold_count'], score['ignored_expected_count']), (2, 3, 2, 1))
        self.assertAlmostEqual(score['precision'], 2 / 3)
        self.assertEqual(score['recall'], 1.0)
        self.assertIsNone(citation_key('Nghị quyết 326/2016/UBTVQH14'))
        self.assertIsNone(citation_key('Điều 114 Luật Tố tụng hành chính'))
        self.assertIsNone(citation_key('Điều 114 2010'))
        self.assertFalse(score_submission([], ['Nghị quyết 326/2016/UBTVQH14'])['scored'])

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_corpus(directory)
            client = FakeChat([
                ('search_legal_rules', {'query': 'Luật Tố tụng hành chính', 'top_k': 1}),
                ('get_legal_rule', {'rule_id': '10'}),
                ('submit_rules', {'rule_ids': ['Điều 114 Luật Tố tụng hành chính 2010']}),
            ])
            with HFDocumentDatabase(directory, FakeEmbedder()) as db:
                result = run_task({'id': 'task', 'input': 'Find relevant law',
                                   'expected_answer': gold}, db, client,
                                  workspace_root=directory / 'work')
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['metrics']['true_positives'], 1)
            self.assertEqual(result['metrics']['gold_count'], 2)
            self.assertEqual(result['conversation']['tools'], TOOLS)
            self.assertEqual(result['conversation']['messages'][1]['content'], 'Find relevant law')
            self.assertNotIn('expected_answer', json.dumps(client.messages))


if __name__ == '__main__':
    unittest.main()
