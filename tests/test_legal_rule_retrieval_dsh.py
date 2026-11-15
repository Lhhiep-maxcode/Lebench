"""Check the legal MCP boundary and native DeepSeek Harness event conversion."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from legal_rule_retrieval.corpus import prepare
from legal_rule_retrieval.dsh_runner import trace_from_events, write_patch
from legal_rule_retrieval.vectors import HashEmbedder, build_vectors


RULE_ID = 'Điều 5, 01/2020/NĐ-CP'
DOC_ID = '01/2020/NĐ-CP'


class DeepSeekIntegrationTests(unittest.TestCase):
    @staticmethod
    def payload(result):
        return result.structuredContent or json.loads(result.content[0].text)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        documents = self.root / 'documents.jsonl'
        parts = self.root / 'parts.jsonl'
        documents.write_text(json.dumps({
            'identifier': DOC_ID, 'title': 'Quy định kinh doanh du lịch',
            'effectiveDate': '2020-01-01', 'expirationDate': '2023-01-01'},
            ensure_ascii=False) + '\n', encoding='utf-8')
        parts.write_text(json.dumps({
            'identifier': RULE_ID, 'parent': DOC_ID, 'partId': '5',
            'name': 'Điều 5', 'partType': 'Điều',
            'content': 'Điều kiện cấp phép kinh doanh du lịch là có vốn.'},
            ensure_ascii=False) + '\n', encoding='utf-8')
        self.rules = self.root / 'rules'
        self.index = self.root / 'index'
        self.workspace = self.root / 'workspaces'
        self.submission = self.root / 'submission.json'
        prepare(documents, parts, self.rules)
        build_vectors(self.rules, self.index, HashEmbedder(), 'hash',
                      chunk_chars=100, overlap=10)

    def test_mcp_transport_search_submit_and_helpers(self):
        async def check():
            params = StdioServerParameters(
                command=sys.executable,
                args=['-m', 'legal_rule_retrieval.mcp_server',
                      '--rules-dir', str(self.rules), '--index-dir', str(self.index),
                      '--workspace-root', str(self.workspace), '--task-id', 'case-1',
                      '--submission-path', str(self.submission)],
                cwd=str(Path(__file__).resolve().parent.parent))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    self.assertEqual(names, {'search_legal_rules', 'get_legal_rule',
                                             'check_validity', 'submit_rules',
                                             'create_file', 'read_file', 'add',
                                             'subtract', 'multiply', 'divide'})
                    premature = await session.call_tool('submit_rules', {'rule_ids': [RULE_ID]})
                    self.assertTrue(premature.isError)
                    found = await session.call_tool('search_legal_rules',
                                                    {'query': 'cấp phép du lịch', 'top_k': 1})
                    self.assertFalse(found.isError)
                    self.assertEqual(self.payload(found)['results'][0]['rule_id'], RULE_ID)
                    full = await session.call_tool('get_legal_rule', {'rule_id': RULE_ID})
                    self.assertIn('có vốn', self.payload(full)['rule']['content'])
                    validity = await session.call_tool('check_validity',
                                                       {'rule_id': RULE_ID, 'start': '2021-01-01',
                                                        'end': '2021-12-31'})
                    self.assertEqual(self.payload(validity)['status'], 'valid')
                    added = await session.call_tool('add', {'a': '0.1', 'b': '0.2'})
                    self.assertEqual(self.payload(added)['result'], '0.3')
                    created = await session.call_tool('create_file',
                                                      {'path': 'note.txt', 'content': 'ghi chú'})
                    self.assertFalse(created.isError)
                    readback = await session.call_tool('read_file', {'path': 'note.txt'})
                    self.assertEqual(self.payload(readback)['content'], 'ghi chú')
                    submitted = await session.call_tool('submit_rules', {'rule_ids': [RULE_ID]})
                    self.assertFalse(submitted.isError)
        asyncio.run(check())
        self.assertEqual(json.loads(self.submission.read_text(encoding='utf-8')),
                         {'submitted_rule_ids': [RULE_ID]})

    def test_patch_and_native_event_trace(self):
        patch = self.root / 'legal.patch.yml'
        write_patch(patch, rules_dir=self.rules, index_dir=self.index,
                    workspace_root=self.workspace, task_id='case-1',
                    submission_path=self.submission)
        text = patch.read_text(encoding='utf-8')
        self.assertIn("name: '@deepseek-ai/dsh-mcp-client'", text)
        self.assertIn('serverName: legal_rules', text)
        self.assertIn('- id: persistent-pwsh\n  disabled: true', text)
        events = [
            {'type': 'tool/call', 'data': {'step': 1, 'callId': 'c1',
             'name': 'mcp__legal_rules__search_legal_rules',
             'arguments': '{"query":"du lịch","top_k":1}'}},
            {'type': 'tool/result', 'data': {'message': {'toolCallId': 'c1',
             'content': [{'type': 'text', 'text': 'found'}], 'isError': False}}},
        ]
        trace = trace_from_events(events)
        self.assertEqual(trace[0]['action'], 'search_legal_rules')
        self.assertEqual(trace[0]['arguments']['top_k'], 1)
        self.assertNotIn('error', trace[0]['observation'])


if __name__ == '__main__':
    unittest.main()
