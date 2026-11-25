"""Embedding endpoint and task-workspace integration checks."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from legal_rule_retrieval.embedding import APIEmbedder
from legal_rule_retrieval.harness import ToolSession


class EmbeddingClientTests(unittest.TestCase):
    def test_embedding_request_uses_embeddings_endpoint_and_response_indexes(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append((self.path, json.loads(self.rfile.read(
                    int(self.headers['Content-Length'])))))
                body = json.dumps({'data': [
                    {'index': 1, 'embedding': [0.0, 1.0]},
                    {'index': 0, 'embedding': [1.0, 0.0]},
                ]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = APIEmbedder(f'http://127.0.0.1:{server.server_port}/v1', 'model')
            vectors = client.embed(['query: first', 'query: second'])
        finally:
            server.shutdown()
            worker.join()
            server.server_close()
        self.assertEqual(requests[0][0], '/v1/embeddings')
        self.assertEqual(requests[0][1]['input'], ['query: first', 'query: second'])
        self.assertEqual(vectors.tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_workspace_rejects_escape_and_append_keeps_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = ToolSession(None, Path(tmp) / 'work', 'case')
            with self.assertRaisesRegex(ValueError, 'inside the task workspace'):
                session.dispatch('create_file', {'path': '../outside.txt', 'content': 'x'})
            session.dispatch('create_file', {'path': 'notes.txt', 'content': 'one'})
            session.dispatch('append_file', {'path': 'notes.txt', 'content': ' two'})
            self.assertEqual(session.dispatch('read_file', {'path': 'notes.txt'})['content'],
                             'one two')
            with self.assertRaisesRegex(ValueError, 'Division by zero'):
                session.dispatch('divide', {'a': 1, 'b': 0})


if __name__ == '__main__':
    unittest.main()
