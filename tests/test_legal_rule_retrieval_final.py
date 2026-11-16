"""Final-corpus preparation, article scoring, and chat-template trace conversion."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from legal_rule_retrieval.dsh_runner import conversation_from_events
from legal_rule_retrieval.final_data import FinalRuleDatabase, prepare_final, score_submission
from legal_rule_retrieval.vectors import HashEmbedder


class CapturingEmbedder(HashEmbedder):
    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts.extend(texts)
        return super().embed(texts)


class FinalCorpusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'final.jsonl'
        self.directory = self.root / 'prepared'
        self.rows = [
            {'identifier': 'Điều 5, 01/2020/BL-LD, 2020-01-01, Bộ luật',
             'title': 'Bộ luật Lao động 2020', 'content': 'Điều 5. Quyền của người lao động',
             'partType': 'Điều', 'effectiveDate': '2020-01-01', 'expirationDate': '2030-01-01',
             'arbitrary_field': {'x': [1, 2]}},
            {'identifier': 'Khoản 2, Điều 5, 01/2020/BL-LD, 2020-01-01, Bộ luật',
             'title': 'Bộ luật Lao động 2020', 'content': 'Khoản 2. Nghĩa vụ',
             'partType': 'Khoản', 'effectiveDate': 1234, 'expirationDate': None},
            {'identifier': 'Điều 5, 02/2020/BL-DS, 2020-01-01, Bộ luật',
             'title': 'Bộ luật Dân sự 2020', 'content': 'Điều 5. Tập quán',
             'partType': 'Điều', 'effectiveDate': '2020-01-01', 'expirationDate': None},
        ]
        self.source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                       for row in self.rows), encoding='utf-8')

    def test_prepare_preserves_fields_embeds_only_unique_titles_and_searches(self):
        embedder = CapturingEmbedder()
        manifest = prepare_final(self.source, self.directory, embedder, 'hash', batch_size=2)
        self.assertEqual(manifest['embedding_field'], 'title')
        self.assertEqual(embedder.texts, ['Bộ luật Lao động 2020', 'Bộ luật Dân sự 2020'])
        with (self.directory / 'rules.jsonl').open(encoding='utf-8') as stream:
            prepared = [json.loads(line) for line in stream]
        for source, row in zip(self.rows, prepared):
            self.assertEqual({key: value for key, value in row.items() if key != 'emb_vector'}, source)
            self.assertEqual(len(row['emb_vector']), 256)
        with FinalRuleDatabase(self.directory, HashEmbedder()) as database:
            self.assertEqual(len(database.search('Bộ luật Lao động', 2)), 2)
            self.assertEqual(database.get(self.rows[0]['identifier'])['arbitrary_field'], {'x': [1, 2]})
            self.assertEqual(database.check_validity(self.rows[0]['identifier'], '2021-01-01',
                                                     '2022-01-01')['status'], 'valid')
            self.assertEqual(database.check_validity(self.rows[1]['identifier'], '2021-01-01',
                                                     '2022-01-01')['status'], 'unknown')

    def test_submit_scores_article_document_and_year_once(self):
        prepare_final(self.source, self.directory, HashEmbedder(), 'hash')
        gold = [{'year': 2020, 'provisions': [
            {'citation_str': 'Điểm a khoản 2 Điều 5 Bộ luật Lao động 2020'},
            {'citation_str': 'Điều 5 Bộ luật Lao động 2020'}]}]
        with FinalRuleDatabase(self.directory, HashEmbedder()) as database:
            score = score_submission([self.rows[0]['identifier'], self.rows[1]['identifier']], gold, database)
            self.assertEqual((score['true_positives'], score['predicted_count'], score['gold_count']), (1, 1, 1))
            self.assertEqual((score['precision'], score['recall']), (1.0, 1.0))
            self.assertEqual(score['matched_rule_ids'], [self.rows[0]['identifier'], self.rows[1]['identifier']])
            wrong = score_submission([self.rows[2]['identifier']], gold, database)
            self.assertEqual((wrong['precision'], wrong['recall']), (0.0, 0.0))
            wrong_year = score_submission([self.rows[0]['identifier']],
                                          [{'provisions': [{'citation_str': 'Điều 5 Bộ luật Lao động 2019'}]}],
                                          database)
            self.assertEqual(wrong_year['true_positives'], 0)

    def test_conversation_contains_tool_calls_observations_and_reasoning(self):
        events = [
            {'type': 'request/header', 'data': {'header': {'tools': [{
                'name': 'mcp__legal_rules__submit', 'description': 'Submit',
                'parameters': {'type': 'object', 'properties': {'identifiers': {'type': 'array'}}}}]}}},
            {'type': 'assistant/message', 'data': {'message': {'content': [
                {'type': 'reasoning', 'text': 'Find the article.'},
                {'type': 'tool-call', 'id': 'call_1', 'name': 'mcp__legal_rules__submit',
                 'arguments': '{"identifiers":[]}'}]}}},
            {'type': 'tool/call', 'data': {'callId': 'call_1', 'name': 'mcp__legal_rules__submit',
                                          'arguments': '{"identifiers":[]}'}},
            {'type': 'tool/result', 'data': {'message': {'toolCallId': 'call_1',
                'content': [{'type': 'text', 'text': '{"precision":1.0}'}]}}},
        ]
        conversation = conversation_from_events(events, 'system', 'user')
        self.assertEqual(conversation['tools'][0]['type'], 'function')
        self.assertEqual(conversation['messages'][2]['reasoning_content'], 'Find the article.')
        self.assertEqual(len(conversation['messages'][2]['tool_calls']), 1)
        self.assertEqual(conversation['messages'][3]['tool_call_id'], 'call_1')
        self.assertTrue(all('reasoning_content' in message for message in conversation['messages']))

    def test_final_mcp_submit_returns_score(self):
        prepare_final(self.source, self.directory, HashEmbedder(), 'hash')
        gold_path = self.root / 'gold.json'
        submission = self.root / 'submission.json'
        gold_path.write_text(json.dumps([{'year': 2020, 'provisions': [
            {'citation_str': 'Điều 5 Bộ luật Lao động 2020'}]}], ensure_ascii=False), encoding='utf-8')

        async def run():
            params = StdioServerParameters(command=sys.executable,
                args=['-m', 'legal_rule_retrieval.mcp_server',
                      '--rules-dir', str(self.directory), '--index-dir', str(self.directory),
                      '--workspace-root', str(self.root / 'workspace'), '--task-id', 'demo',
                      '--submission-path', str(submission), '--gold-path', str(gold_path)],
                cwd=str(Path(__file__).resolve().parent.parent))
            async with stdio_client(params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    self.assertIn('submit', names)
                    self.assertNotIn('submit_rules', names)
                    response = await session.call_tool('submit', {'identifiers': [self.rows[0]['identifier']]})
                    self.assertFalse(response.isError)
                    data = response.structuredContent or json.loads(response.content[0].text)
                    self.assertEqual((data['precision'], data['recall']), (1.0, 1.0))
        asyncio.run(run())
        self.assertEqual(json.loads(submission.read_text(encoding='utf-8'))['true_positives'], 1)


if __name__ == '__main__':
    unittest.main()
