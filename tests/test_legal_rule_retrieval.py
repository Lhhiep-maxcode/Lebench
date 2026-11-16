import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from legal_rule_retrieval.__main__ import main
from legal_rule_retrieval.corpus import prepare
from legal_rule_retrieval.files import private_text_output
from legal_rule_retrieval.harness import OpenAICompatibleClient, ToolSession, aggregate, parse_action, run_task
from legal_rule_retrieval.vectors import APIEmbedder, HashEmbedder, LocalEmbedder, VectorRuleDatabase, build_vectors


DOC_ID = '10/2020/ND, 2020-01-01, Nghị định, Chính phủ'
ARTICLE_ID = 'Điều 3, ' + DOC_ID
CLAUSE_ID = 'Khoản 2, ' + ARTICLE_ID
OTHER_ID = 'Điều 4, ' + DOC_ID


class ScriptedClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return next(self.actions)


class RecordingEmbedder:
    def __init__(self):
        self.inputs = []

    def embed(self, texts):
        self.inputs.extend(texts)
        return HashEmbedder().embed(texts)


class RuleHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        documents = base / 'documents.jsonl'
        parts = base / 'parts.jsonl'
        documents.write_text(json.dumps({
            'identifier': DOC_ID, 'lawId': 10, 'documentCode': '10/2020/ND',
            'title': 'Nghị định về du lịch', 'effectiveDate': '2020-02-01',
            'expirationDate': '2022-01-01', 'validityStatus': 'Hết hiệu lực'}, ensure_ascii=False) + '\n',
            encoding='utf-8')
        parts.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in [
            {'identifier': ARTICLE_ID, 'parent': DOC_ID, 'partId': '3',
             'name': 'Điều 3', 'partType': 'Điều', 'content': 'Điều 3. Điều kiện cấp phép kinh doanh du lịch.'},
            {'identifier': CLAUSE_ID, 'parent': ARTICLE_ID, 'partId': '2',
             'name': 'Khoản 2', 'partType': 'Khoản', 'content': '2. Phải có vốn kinh doanh và trụ sở.'},
            {'identifier': OTHER_ID, 'parent': DOC_ID, 'partId': '4',
             'name': 'Điều 4', 'partType': 'Điều', 'content': 'Điều 4. Thủ tục nộp đơn.'},
            {'identifier': 'Điều 1, missing-document', 'parent': 'missing-document',
             'partId': '1', 'name': 'Điều 1', 'content': 'Quy định khác.'},
        ]) + '\n', encoding='utf-8')
        self.rules_dir, self.index_dir = base / 'prepared', base / 'vectors'
        self.stats = prepare(documents, parts, self.rules_dir)
        self.manifest = build_vectors(self.rules_dir, self.index_dir, HashEmbedder(),
                                      'hash', batch_size=2, chunk_chars=100, overlap=20)
        self.database = VectorRuleDatabase(self.rules_dir, self.index_dir, HashEmbedder())
        self.addCleanup(self.database.close)
        self.workspace = base / 'workspaces'

    def test_preparation_nested_ids_and_vector_retrieval(self):
        self.assertEqual(self.stats['rules'], 4)
        self.assertEqual(self.stats['rules_without_document'], 1)
        self.assertEqual(self.manifest['vectors'], 4)
        self.assertEqual(self.database.get(CLAUSE_ID)['document_id'], DOC_ID)
        self.assertIsNone(self.database.get('Điều 1, missing-document')['document_id'])
        self.assertEqual(self.database.get(CLAUSE_ID)['content'], '2. Phải có vốn kinh doanh và trụ sở.')
        self.assertIn(CLAUSE_ID, [hit['rule_id'] for hit in self.database.search('vốn kinh doanh', 4)])
        with self.assertRaisesRegex(ValueError, '4096'):
            self.database.search('x' * 4097)
        self.assertIsNone(self.database.get('Khoản 2'))

    def test_integrity_rejects_modified_prepared_data(self):
        self.database.close()
        corpus = self.rules_dir / 'rules.jsonl'
        data = corpus.read_text(encoding='utf-8').splitlines()
        first = json.loads(data[0]); first['content'] += ' changed'
        data[0] = json.dumps(first, ensure_ascii=False)
        corpus.write_text('\n'.join(data) + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed since embedding'):
            VectorRuleDatabase(self.rules_dir, self.index_dir, HashEmbedder())

    def test_embedding_prefixes_are_reused_for_queries(self):
        encoder = RecordingEmbedder()
        index_dir = Path(self.tmp.name) / 'prefix_index'
        manifest = build_vectors(self.rules_dir, index_dir, encoder, 'hash',
                                 passage_prefix='passage: ', query_prefix='query: ')
        self.assertEqual(manifest['query_prefix'], 'query: ')
        self.assertTrue(encoder.inputs[0].startswith('passage: '))
        with VectorRuleDatabase(self.rules_dir, index_dir, encoder) as database:
            database.search('du lịch', 2)
        self.assertEqual(encoder.inputs[-1], 'query: du lịch')

    def test_validity_is_conservative(self):
        self.assertEqual(self.database.check_validity(ARTICLE_ID, '2020-03-01', '2021-12-31')['status'], 'valid')
        self.assertEqual(self.database.check_validity(ARTICLE_ID, '2019-01-01', '2019-12-31')['status'], 'invalid')
        self.assertEqual(self.database.check_validity(ARTICLE_ID, '2022-01-01', '2023-01-01')['status'], 'invalid')
        self.assertEqual(self.database.check_validity(ARTICLE_ID, '2021-12-01', '2022-03-01')['status'], 'partially_valid')
        self.assertEqual(self.database.check_validity('Điều 1, missing-document', '2020-01-01', '2020-02-01')['status'], 'unknown')

    def test_file_and_decimal_math_tools(self):
        session = ToolSession(self.database, self.workspace, 'task/1')
        self.assertEqual(session.dispatch('create_file', {'path': 'notes/facts.txt', 'content': 'văn bản'})['characters_written'], 7)
        self.assertEqual(session.dispatch('read_file', {'path': 'notes/facts.txt'})['content'], 'văn bản')
        with self.assertRaises(ValueError):
            session.dispatch('read_file', {'path': '../outside.txt'})
        outside = Path(self.tmp.name) / 'outside.txt'
        outside.write_text('private', encoding='utf-8')
        try:
            (session.workspace / 'link.txt').symlink_to(outside)
        except (OSError, NotImplementedError):
            pass  # Windows may disallow symlink creation for this account.
        else:
            with self.assertRaises(ValueError):
                session.dispatch('read_file', {'path': 'link.txt'})
        with self.assertRaises(ValueError):
            session.dispatch('create_file', {'path': 'notes/facts.txt', 'content': 'overwrite'})
        for action, expected in [('add', '0.3'), ('subtract', '-0.1'),
                                 ('multiply', '0.02'), ('divide', '0.5')]:
            self.assertEqual(session.dispatch(action, {'a': '0.1', 'b': '0.2'})['result'], expected)
        with self.assertRaisesRegex(ValueError, 'Division by zero'):
            session.dispatch('divide', {'a': 1, 'b': 0})
        for number in range(9):
            session.dispatch('create_file', {'path': f'note-{number}.txt', 'content': 'a'})
        with self.assertRaisesRegex(ValueError, 'quota'):
            session.dispatch('create_file', {'path': 'too-many.txt', 'content': 'a'})

    def test_offline_model_and_http_url_guards(self):
        with self.assertRaisesRegex(ValueError, 'existing directory'):
            LocalEmbedder('this-model-would-require-a-download')
        model_dir = Path(self.tmp.name) / 'local-model'
        model_dir.mkdir()
        calls = []
        fake_module = types.ModuleType('sentence_transformers')
        class FakeSentenceTransformer:
            def __init__(self, path, local_files_only=False):
                calls.append((path, local_files_only))
        fake_module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {'sentence_transformers': fake_module}):
            LocalEmbedder(model_dir)
        self.assertEqual(calls, [(str(model_dir.resolve()), True)])
        with self.assertRaisesRegex(ValueError, 'HTTP'):
            APIEmbedder('file:///tmp/rules', 'embed-model')
        with self.assertRaisesRegex(ValueError, 'without credentials'):
            OpenAICompatibleClient('http://user:secret@localhost:8000/v1', 'chat-model')

    def test_embedding_client_does_not_follow_redirects(self):
        class Redirect(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header('Location', 'https://example.invalid/collect')
                self.end_headers()

            def log_message(self, *_):
                pass

        with HTTPServer(('127.0.0.1', 0), Redirect) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = APIEmbedder(f'http://127.0.0.1:{server.server_port}/v1', 'embed-model')
                with patch.dict(os.environ, {'HTTP_PROXY': 'http://127.0.0.1:1',
                                            'http_proxy': 'http://127.0.0.1:1',
                                            'NO_PROXY': '', 'no_proxy': ''}):
                    with self.assertRaisesRegex(RuntimeError, 'Embedding HTTP 302'):
                        client.embed(['sample'])
            finally:
                server.shutdown()
                thread.join()

    def test_benchmark_output_is_private(self):
        path = Path(self.tmp.name) / 'trace.jsonl'
        with private_text_output(path) as stream:
            stream.write('private case facts\n')
        if os.name == 'posix':
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            link = Path(self.tmp.name) / 'trace-link.jsonl'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                private_text_output(link)

    def test_multi_turn_trace_and_gold_not_in_prompt(self):
        task = {'id': 't1', 'benchmark': {'facts': 'Một người xin phép kinh doanh du lịch.',
                'issues': ['Điều kiện cấp phép kinh doanh du lịch'], 'gold_rules': [ARTICLE_ID]}}
        actions = [
            {'action': 'search_legal_rules', 'arguments': {'query': 'điều kiện cấp phép kinh doanh du lịch', 'top_k': 4}},
            {'action': 'get_legal_rule', 'arguments': {'rule_id': ARTICLE_ID}},
            {'action': 'submit_rules', 'arguments': {'rule_ids': [ARTICLE_ID, ARTICLE_ID]}},
        ]
        client = ScriptedClient([json.dumps(x, ensure_ascii=False) for x in actions])
        result = run_task(task, self.database, client, workspace_root=self.workspace)
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual(result['submitted_rule_ids'], [ARTICLE_ID])
        self.assertEqual(result['metrics']['f1'], 1.0)
        self.assertEqual(result['metrics']['retrieval_calls'], 2)
        self.assertEqual(result['metrics']['tool_call_efficiency'], 0.5)
        self.assertEqual([x['action'] for x in result['trace']],
                         ['search_legal_rules', 'get_legal_rule', 'submit_rules'])
        self.assertNotIn('gold_rules', json.dumps(client.messages, ensure_ascii=False))
        self.assertIn('observation', client.messages[1][-1]['content'])
        self.assertEqual(aggregate([result])['micro_f1'], 1.0)

    def test_bad_action_observation_and_step_limit(self):
        task = {'id': 't2', 'benchmark': {'facts': 'F', 'issues': ['I'], 'gold_rules': [ARTICLE_ID]}}
        bad = ScriptedClient(['not json', json.dumps({'action': 'submit_rules',
                                                       'arguments': {'rule_ids': [ARTICLE_ID]}}, ensure_ascii=False)])
        result = run_task(task, self.database, bad, max_steps=2, workspace_root=self.workspace)
        self.assertEqual(result['status'], 'max_steps')
        self.assertEqual(result['metrics']['failed_tool_calls'], 1)
        self.assertIn('error', result['trace'][0]['observation'])
        self.assertIn('error', result['trace'][1]['observation'])
        self.assertEqual(result['metrics']['recall'], 0.0)

    def test_embedding_api_request_and_ordering(self):
        response = {'data': [{'index': 1, 'embedding': [0, 1]},
                             {'index': 0, 'embedding': [1, 0]}]}
        client = APIEmbedder('http://localhost:8001/v1', 'embed-model', api_key='secret')
        with patch('legal_rule_retrieval.vectors.urlopen_direct', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            vectors = client.embed(['a', 'b'])
        self.assertEqual(vectors.tolist(), [[1, 0], [0, 1]])
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:8001/v1/embeddings')
        self.assertEqual(json.loads(request.data)['input'], ['a', 'b'])

    def test_chat_api_request_and_fenced_action(self):
        client = OpenAICompatibleClient('http://localhost:8000/v1', 'local-model',
                                        api_key='secret', extra_body={'max_tokens': 512})
        response = {'choices': [{'message': {'content': '```json\n{"action":"submit_rules","arguments":{"rule_ids":[]}}\n```'}}]}
        with patch('legal_rule_retrieval.harness.urlopen_direct', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            raw = client.complete([{'role': 'user', 'content': 'Find a rule'}])
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, 'http://localhost:8000/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer secret')
        body = json.loads(request.data)
        self.assertEqual(body['model'], 'local-model')
        self.assertEqual(body['max_tokens'], 512)
        self.assertEqual(parse_action(raw), ('submit_rules', {'rule_ids': []}))

    def test_run_cli_from_prepared_vectors(self):
        task_path = Path(self.tmp.name) / 'tasks.jsonl'
        output_path = Path(self.tmp.name) / 'results.jsonl'
        task_path.write_text(json.dumps({'id': 'cli-1', 'benchmark': {
            'facts': 'Kinh doanh du lịch', 'issues': ['Điều kiện cấp phép'],
            'gold_rules': [ARTICLE_ID]}}, ensure_ascii=False) + '\n', encoding='utf-8')
        actions = [
            {'action': 'search_legal_rules', 'arguments': {'query': 'điều kiện cấp phép', 'top_k': 4}},
            {'action': 'get_legal_rule', 'arguments': {'rule_id': ARTICLE_ID}},
            {'action': 'submit_rules', 'arguments': {'rule_ids': [ARTICLE_ID]}},
        ]
        responses = [io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(action, ensure_ascii=False)}}]}).encode())
                     for action in actions]
        with patch('legal_rule_retrieval.harness.urlopen_direct', side_effect=responses):
            code = main(['run', '--rules-dir', str(self.rules_dir), '--index-dir', str(self.index_dir),
                         '--tasks', str(task_path), '--output', str(output_path),
                         '--workspace-root', str(self.workspace), '--base-url', 'http://localhost:8000/v1',
                         '--model', 'fake-chat-model'])
        self.assertEqual(code, 0)
        result = json.loads(output_path.read_text(encoding='utf-8'))
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual(result['metrics']['f1'], 1.0)


if __name__ == '__main__':
    unittest.main()
