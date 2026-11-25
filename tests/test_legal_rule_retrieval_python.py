"""Native tool-call request and corpus-level runner tests."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from legal_rule_retrieval.python_runner import ChatClient, run_benchmark
from test_hf_benchmark import FakeChat, FakeEmbedder, make_corpus


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
        names = {item['function']['name'] for item in received[0]['tools']}
        self.assertIn('check_relationship', names)
        self.assertIn('append_file', names)
        self.assertIn('submit_rules', names)
        self.assertEqual(len(names), 12)
        self.assertNotIn('reasoning_content', received[0]['messages'][0])

    def test_benchmark_loads_document_corpus_once_and_limits_faiss_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / 'corpus'
            corpus.mkdir()
            make_corpus(corpus)
            tasks = root / 'tasks.jsonl'
            tasks.write_text(json.dumps({'id': 'case', 'input': 'Find relevant law',
                                         'expected_answer': [
                                             'Điều 114 Luật Tố tụng hành chính 2010']},
                                        ensure_ascii=False) + '\n', encoding='utf-8')
            scripted = FakeChat([
                ('search_legal_rules', {'query': 'tố tụng hành chính', 'top_k': 1}),
                ('submit_rules', {'rule_ids': ['Điều 114 Luật Tố tụng hành chính 2010']}),
            ])
            with patch('legal_rule_retrieval.python_runner.ChatClient', return_value=scripted), \
                    patch('legal_rule_retrieval.python_runner.APIEmbedder',
                          return_value=FakeEmbedder()), \
                    patch('legal_rule_retrieval.python_runner.faiss.omp_set_num_threads') as set_threads:
                summary = run_benchmark(tasks, root / 'results.jsonl', corpus_dir=corpus,
                                        workspace_root=root / 'work', base_url='http://localhost/v1',
                                        model='mock', search_threads=1)
            set_threads.assert_called_once_with(1)
            self.assertEqual(summary['statuses']['submitted'], 1)
            self.assertEqual(summary['micro_f1'], 1.0)
            result = json.loads((root / 'results.jsonl').read_text(encoding='utf-8').strip())
            self.assertEqual(result['expected_answer'], [
                'Điều 114 Luật Tố tụng hành chính 2010'])
            self.assertEqual(result['correct_answers'], [
                'Điều 114 Luật Tố tụng hành chính 2010'])
            self.assertEqual(result['incorrect_answers'], [])


if __name__ == '__main__':
    unittest.main()
