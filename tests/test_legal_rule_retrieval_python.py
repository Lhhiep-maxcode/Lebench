"""Direct Python legal tool loop without DeepSeek SDK or MCP startup."""

import json
import io
import contextlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from legal_rule_retrieval.final_data import FinalRuleDatabase, prepare_final
from legal_rule_retrieval.python_runner import ChatClient, TOOLS, run_benchmark, run_task
from legal_rule_retrieval.vectors import HashEmbedder


class ScriptedClient:
    def __init__(self, rule_id):
        self.rule_id = rule_id
        self.calls = 0
        self.url = 'http://localhost/v1/chat/completions'

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            name, arguments = 'search_legal_rules', {'query': 'Bộ luật Lao động 2020', 'top_k': 5}
        else:
            assert messages[-1]['role'] == 'tool'
            name, arguments = 'submit', {'identifiers': [self.rule_id]}
        return {'role': 'assistant', 'content': '', 'reasoning_content': 'checking' if self.calls == 1 else '',
                'tool_calls': [{'id': f'call_{self.calls}', 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}


class DirectRunnerTests(unittest.TestCase):
    def test_partial_load_uses_only_prefix_and_builds_its_own_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'parts.jsonl'
            rules = [{'identifier': f'Rule {number}', 'title': f'Law {number}',
                      'content': f'Article {number}'} for number in range(3)]
            source.write_text(''.join(json.dumps(row) + '\n' for row in rules), encoding='utf-8')
            prepared = root / 'prepared'
            prepare_final(source, prepared, HashEmbedder(), 'hash')

            with patch('legal_rule_retrieval.final_data.faiss.read_index',
                       side_effect=AssertionError('full index must not be read')):
                with FinalRuleDatabase(prepared, HashEmbedder(), max_rules=2) as database:
                    self.assertEqual(database.index.ntotal, 2)
                    self.assertLess(database.source.tell(), (prepared / 'rules.jsonl').stat().st_size)
                    self.assertIsNone(database.get('Rule 2'))
                    with patch('legal_rule_retrieval.final_data.np.linalg.norm',
                               side_effect=AssertionError('single-query search must avoid BLAS')):
                        self.assertEqual({hit['rule_id'] for hit in database.search('Law 1', top_k=3)},
                                         {'Rule 0', 'Rule 1'})
            with self.assertRaisesRegex(ValueError, 'max_rules must be a positive integer'):
                FinalRuleDatabase(prepared, HashEmbedder(), max_rules=0)

    def test_benchmark_caps_faiss_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rules = [{'identifier': f'Rule {i}', 'title': f'Law {i}', 'content': 'Text'}
                     for i in range(2)]
            source = root / 'parts.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rules), encoding='utf-8')
            prepared = root / 'prepared'
            prepare_final(source, prepared, HashEmbedder(), 'hash')
            tasks = root / 'tasks.jsonl'
            tasks.write_text(json.dumps({'id': 'case', 'benchmark_seed': {'facts_clean': 'Facts'},
                                         'issues': ['Issue'], 'laws': []}) + '\n', encoding='utf-8')
            with patch('legal_rule_retrieval.python_runner.ChatClient',
                       return_value=ScriptedClient('Rule 0')), \
                    patch('legal_rule_retrieval.python_runner.faiss.omp_set_num_threads') as set_threads:
                summary = run_benchmark(tasks, root / 'results.jsonl', final_dir=prepared,
                                        workspace_root=root / 'work', base_url='http://localhost/v1',
                                        model='mock', max_rules=1, search_threads=1)
            set_threads.assert_called_once_with(1)
            self.assertEqual(summary['statuses']['submitted'], 1)

    def test_chat_client_sends_native_tool_schemas(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                received.append(json.loads(body))
                payload = json.dumps({'choices': [{'message': {
                    'role': 'assistant', 'content': '', 'tool_calls': []}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ChatClient(f'http://127.0.0.1:{server.server_port}/v1', 'mock',
                                max_tokens=512)
            client.complete([{'role': 'user', 'content': 'Find a law', 'reasoning_content': ''}])
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        self.assertEqual(received[0]['max_tokens'], 512)
        self.assertEqual(len(received[0]['tools']), 10)
        self.assertNotIn('reasoning_content', received[0]['messages'][0])

    def test_search_observation_submit_and_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rule = {'identifier': 'Điều 5, 01/2020/BL-LD, 2020-01-01, Bộ luật',
                    'title': 'Bộ luật Lao động 2020', 'content': 'Điều 5. Quyền của người lao động',
                    'effectiveDate': '2020-01-01', 'expirationDate': None}
            source = root / 'parts.jsonl'
            source.write_text(json.dumps(rule, ensure_ascii=False) + '\n', encoding='utf-8')
            prepared = root / 'prepared'
            prepare_final(source, prepared, HashEmbedder(), 'hash')
            sample = {'doc_name': 'case-1', 'benchmark_seed': {'facts_clean': 'Người lao động có tranh chấp.'},
                      'issues': ['Quyền của người lao động'],
                      'laws': [{'year': 2020, 'provisions': [
                          {'citation_str': 'Điều 5 Bộ luật Lao động 2020'}]}]}
            log = io.StringIO()
            with contextlib.redirect_stderr(log), \
                    FinalRuleDatabase(prepared, HashEmbedder()) as database:
                result = run_task(sample, database, ScriptedClient(rule['identifier']),
                                  workspace_root=root / 'work', debug=True, verbosity=2)
            self.assertEqual(result['status'], 'submitted')
            self.assertEqual(result['metrics']['precision'], 1.0)
            self.assertEqual(result['metrics']['recall'], 1.0)
            self.assertEqual(result['metrics']['work_calls'], 1)
            self.assertEqual([step['action'] for step in result['trace']], ['search_legal_rules', 'submit'])
            conversation = result['conversation']
            self.assertEqual(len(conversation['tools']), len(TOOLS))
            self.assertEqual([message['role'] for message in conversation['messages']],
                             ['system', 'user', 'assistant', 'tool', 'assistant', 'tool'])
            self.assertTrue(all('reasoning_content' in message for message in conversation['messages']))
            self.assertNotIn('laws', conversation['messages'][1]['content'])
            self.assertIn('step 1: sending chat request', log.getvalue())
            self.assertIn('tool submit finished', log.getvalue())
            events = [json.loads(line.removeprefix('TRACE '))
                      for line in log.getvalue().splitlines() if line.startswith('TRACE ')]
            self.assertEqual([event['event'] for event in events],
                             ['task_start', 'llm_request', 'llm_response',
                              'tool_call', 'tool_result', 'llm_request',
                              'llm_response', 'tool_call', 'tool_result', 'task_end'])
            self.assertEqual(events[2]['reasoning_content'], 'checking')
            self.assertEqual(events[3]['arguments']['top_k'], 5)
            self.assertIn('results', events[4]['observation'])
            self.assertEqual(events[8]['observation']['precision'], 1.0)
            self.assertEqual(events[-1]['status'], 'submitted')
            self.assertTrue(all(event['timestamp'].endswith('+00:00') for event in events))
            self.assertIn('Loading legal parts', log.getvalue())

            rules_path = prepared / 'rules.jsonl'
            changed = json.loads(rules_path.read_text(encoding='utf-8'))
            changed['content'] += ' changed'
            rules_path.write_text(json.dumps(changed, ensure_ascii=False) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Final corpus changed'):
                FinalRuleDatabase(prepared, HashEmbedder())


if __name__ == '__main__':
    unittest.main()
