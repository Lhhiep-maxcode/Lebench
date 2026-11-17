"""Direct Python legal tool loop without DeepSeek SDK or MCP startup."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from legal_rule_retrieval.final_data import FinalRuleDatabase, prepare_final
from legal_rule_retrieval.python_runner import ChatClient, TOOLS, run_task
from legal_rule_retrieval.vectors import HashEmbedder


class ScriptedClient:
    def __init__(self, rule_id):
        self.rule_id = rule_id
        self.calls = 0

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
            with FinalRuleDatabase(prepared, HashEmbedder()) as database:
                result = run_task(sample, database, ScriptedClient(rule['identifier']),
                                  workspace_root=root / 'work')
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


if __name__ == '__main__':
    unittest.main()
