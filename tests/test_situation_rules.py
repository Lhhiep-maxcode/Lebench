import json
import subprocess
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import httpx

from server.extract_situations_rules import extract_document, load_vocabulary, repair_ocr, strip_page_markers
from server.extract_situations_simple import extract_one
from server.situation_prompts import INPUT_SEPARATOR, JUDGE_SYSTEM_PROMPT


class RuleTests(unittest.TestCase):
    def test_vocabulary_loader_handles_phrases_unicode_bom_and_nonword_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'words.txt'
            path.write_text('Người lao động\nthẩm phán\nnguye\u0302n\nA-ba-toa\n2024\nabc_01\n# comment\nkhởi kiện # note\n',
                            encoding='utf-8-sig')
            self.assertEqual(load_vocabulary(path), {'người', 'lao', 'động', 'thẩm', 'phán', 'nguyên', 'khởi', 'kiện'})

    def test_cli_uses_viet74k_by_default_when_ocr_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory)/'input.jsonl', Path(directory)/'result.jsonl'
            text = 'NỘI DUNG VỤ ÁN:\nBà A giao tiền.\nQUYẾT ĐỊNH:\nBuộc trả.'
            source.write_text(json.dumps({'id': 1, 'markdown': text})+'\n', encoding='utf-8')
            script = Path(__file__).resolve().parents[1]/'server'/'extract_situations_rules.py'
            process = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(destination), '--fix-ocr'],
                                     capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(destination.read_text(encoding='utf-8'))
            self.assertEqual(result['situation'], 'Bà A giao tiền.')
            self.assertEqual(result['situation_extraction']['ocr_vocabulary'], 'Viet74K.txt')
            self.assertGreater(result['situation_extraction']['ocr_vocabulary_size'], 1000)

    def test_extract_is_a_source_slice_and_clears_stale_llm_verdict(self):
        body = 'Bà A cho ông B vay 1.650.000đ ngày 01/02/2020. Ông B phủ nhận.'
        row = {'id': 1, 'markdown': 'NHÂN DANH\nNỘI DUNG VỤ ÁN:\n'+body+'\nNHẬN ĐỊNH CỦA TÒA ÁN:\nLý do.\nQUYẾT ĐỊNH:\nBuộc trả.',
               'situation': 'old', 'situation_judge': {'passed': True}, 'situation_attempts': 4}
        result = extract_document(row)
        self.assertEqual(result['situation_raw'], body)
        self.assertEqual(result['situation'], body)
        self.assertEqual(result['situation_status'], 'extracted')
        meta = result['situation_extraction']
        self.assertEqual(result['markdown'][meta['start']:meta['end']], body)
        self.assertEqual(result['markdown_chars'], len(result['markdown']))
        self.assertIsNone(result['situation_judge'])
        self.assertEqual(result['situation_attempts'], 0)
        self.assertEqual(row['situation'], 'old')

    def test_page_markers_are_removed_including_inline_braces_and_page_lists(self):
        text = 'trước## Page 1\nsau ## Page {2} tiếp ## Page {3/4} hết'
        cleaned = strip_page_markers(text)
        self.assertNotIn('Page', cleaned)
        self.assertNotIn('{', cleaned)
        self.assertNotIn('}', cleaned)
        self.assertIn('trước\n', cleaned)
        result = extract_document({'markdown': '## Page 1\nNỘI DUNG VỤ ÁN:\nBà A khai.\n## Page 2\nÔng B phủ nhận.\nQUYẾT ĐỊNH:\nBuộc trả.'})
        self.assertNotIn('Page', result['markdown'])
        self.assertIn('Bà A khai.', result['situation'])
        self.assertIn('Ông B phủ nhận.', result['situation'])

    def test_accentless_decomposed_and_multiline_headers_preserve_body(self):
        body = 'Nguyễn Văn A, sinh năm 1981, giao 300.000đ.'
        for begin, end in [('NOI DUNG VU AN', 'NHAN DINH CUA TOA AN'),
                           ('NỘI DUNG\nVỤ ÁN', 'NHẬN ĐỊNH CỦA TOÀ ÁN'),
                           (unicodedata.normalize('NFD', 'NỘI DUNG VIỆC DÂN SỰ'),
                            unicodedata.normalize('NFD', 'NHẬN ĐỊNH CỦA TÒA ÁN'))]:
            with self.subTest(begin=begin):
                result = extract_document({'markdown': begin+':\n'+body+'\n'+end+':\nPhán quyết.'})
                self.assertEqual(result['situation'], body)
                self.assertEqual(result['situation_status'], 'extracted')

    def test_flattened_page_uses_uppercase_labels_not_ordinary_decision_references(self):
        body = 'Anh A phản đối Quyết định số 58 về việc thu hồi đất.'
        result = extract_document({'markdown': 'Thông tin tòa án. NỘI DUNG VỤ ÁN: '+body+' NHẬN ĐỊNH CỦA TÒA ÁN: Lý do. QUYẾT ĐỊNH: 1. Bác.'})
        self.assertEqual(result['situation'], body)

    def test_old_style_facts_stop_before_xet_thay_reasoning(self):
        result = extract_document({'markdown': 'NHÂN DANH\nNHẬN THẤY:\nA giao tiền cho B.\nXÉT THẤY:\nYêu cầu có căn cứ.\nQUYẾT ĐỊNH:\nBuộc B trả.'})
        self.assertEqual(result['situation'], 'A giao tiền cho B.')
        self.assertEqual(result['situation_extraction']['end_heading']['heading'], 'XÉT THẤY')

    def test_xet_thay_procedural_decision_is_kept_but_flagged_for_review(self):
        result = extract_document({'markdown': 'QUYẾT ĐỊNH\nĐÌNH CHỈ\nXét thấy: Bà A rút yêu cầu, cần chấp nhận.\nQUYẾT ĐỊNH\n1. Đình chỉ.'})
        self.assertEqual(result['situation'], 'Bà A rút yêu cầu, cần chấp nhận.')
        self.assertEqual(result['situation_status'], 'needs_review')
        self.assertIn('xet_thay_may_include_legal_reasoning', result['situation_extraction']['warnings'])

    def test_fuzzy_heading_does_not_turn_nhan_danh_or_quy_dinh_into_a_boundary(self):
        result = extract_document({'markdown': 'NHÂN DANH\nNỘI D VỤ ÁN:\nA khai.\nNHÂN THÂN:\nB khai.\nQUY ĐỊNH\nC khai.\nNH N Đ NH CỦA TÒA N:\nLý do.'})
        self.assertTrue(result['situation'].startswith('A khai.'))
        self.assertIn('C khai.', result['situation'])
        self.assertEqual(result['situation_status'], 'needs_review')
        self.assertIn('fuzzy_heading_match', result['situation_extraction']['warnings'])

    def test_missing_boundaries_never_fall_back_to_the_whole_judgment_or_its_reasoning(self):
        for text in ('Không có các tiêu đề cần thiết.', 'NỘI DUNG VỤ ÁN:\nA khai.',
                     'NHẬN ĐỊNH CỦA TÒA ÁN:\nXét thấy: A phải trả.\nQUYẾT ĐỊNH:\nBuộc A trả.'):
            with self.subTest(text=text):
                result = extract_document({'markdown': text})
                self.assertIsNone(result['situation'])
                self.assertEqual(result['situation_status'], 'missing_section')

    def test_recommendations_inside_content_are_flagged_instead_of_silently_deleted(self):
        body = 'A khai nhận. Viện kiểm sát đề nghị phạt A 12 tháng tù.'
        result = extract_document({'markdown': 'NỘI DUNG VỤ ÁN:\n'+body+'\nNHẬN ĐỊNH CỦA TÒA ÁN:\nLý do.'})
        self.assertEqual(result['situation'], body)
        self.assertEqual(result['situation_status'], 'needs_review')
        self.assertIn('may_include_prosecutor_recommendations', result['situation_extraction']['warnings'])

    def test_ocr_joins_broken_syllables_and_logs_unique_one_edit_repairs(self):
        vocabulary = {'thẩm', 'khởi', 'kiện', 'cứu'}
        original = 'sơ th ẩm, khởl kiện và nghiên c ứu.'
        fixed, edits = repair_ocr(original, vocabulary)
        self.assertEqual(fixed, 'sơ thẩm, khởi kiện và nghiên cứu.')
        self.assertEqual(len(edits), 3)
        for edit in edits:
            self.assertEqual(original[edit['start']:edit['end']], edit['before'])
        result = extract_document({'markdown': 'NỘI DUNG VỤ ÁN:\n'+original+'\nQUYẾT ĐỊNH:\nBuộc trả.'}, vocabulary)
        self.assertEqual(result['situation_raw'], original)
        self.assertEqual(result['situation'], fixed)
        self.assertEqual(result['situation_status'], 'needs_review')

    def test_ocr_preserves_names_identifiers_numbers_negation_tone_only_and_ambiguous_words(self):
        vocabulary = {'khởi', 'không', 'kiện', 'nghiêm', 'nghiên'}
        text = 'Ông Khởl; AB-khởl; 123khởl; 01/02/2020; 1.650.000đ; khôg; kiên; nghiêx.'
        self.assertEqual(repair_ocr(text, vocabulary), (text, []))

    def test_small_vocabulary_does_not_change_valid_vietnamese_words_into_other_words(self):
        # These are real false positives found in the supplied corpus before adding the syllable guard.
        vocabulary = {'trộm', 'thi', 'khác', 'phí', 'nghiên', 'người', 'giam', 'khởi'}
        text = 'Mâu thuẫn trầm trọng; thời hạn; khách quan; phía công ty; tuy nhiên; 0,170 gram; 15 khởl.'
        self.assertEqual(repair_ocr(text, vocabulary), (text, []))
        fixed, edits = repair_ocr('ngƣời', vocabulary)
        self.assertEqual(fixed, 'người')
        self.assertEqual(len(edits), 1)

    def test_large_vocabulary_does_not_rewrite_borrowed_words_or_delete_letters_to_force_a_match(self):
        vocabulary = {'photon', 'nylon', 'modem', 'phong', 'thuồi', 'heroin', 'định'}
        text = 'photo, nilon, model, phone, thuhồi, heroine.'
        self.assertEqual(repair_ocr(text, vocabulary), (text, []))
        fixed, edits = repair_ocr('địnhh', vocabulary)
        self.assertEqual(fixed, 'định')
        self.assertEqual(len(edits), 1)

    def test_cli_processes_valid_rows_and_reports_truncated_final_json(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory)/'input.jsonl', Path(directory)/'result.jsonl'
            text = '## Page 1\nNỘI DUNG VỤ ÁN:\nBà A giao tiền.\nQUYẾT ĐỊNH:\nBuộc trả.'
            source.write_text(json.dumps({'id': 1, 'markdown': text})+'\n{"id":2', encoding='utf-8-sig')
            before = source.read_bytes()
            script = Path(__file__).resolve().parents[1]/'server'/'extract_situations_rules.py'
            process = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(destination)],
                                     capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(process.returncode, 1)
            self.assertIn('Line 2:', process.stderr)
            self.assertEqual(json.loads(process.stdout), {'extracted': 1, 'invalid_input': 1})
            result = json.loads(destination.read_text(encoding='utf-8'))
            self.assertEqual(result['situation'], 'Bà A giao tiền.')
            self.assertEqual(source.read_bytes(), before)
            same = subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(source)],
                                  capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(same.returncode, 0)
            self.assertEqual(source.read_bytes(), before)


class LLMIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_writer_and_judge_receive_page_free_source(self):
        received = []

        def handle(request):
            body = json.loads(request.content)
            payload = json.loads(body['messages'][1]['content'].split(INPUT_SEPARATOR, 1)[1])
            received.append(payload['judgment'])
            judge = body['messages'][0]['content'] == JUDGE_SYSTEM_PROMPT
            text = '{"passed":true,"feedback":[]}' if judge else 'draft'
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]})

        args = SimpleNamespace(max_retries=0, writer_model='writer', judge_model='judge', max_tokens=128,
                               base_url='http://model.test/v1', judge_base_url=None, extra_body={}, use_law_hints=False)
        text = '## Page 1\nA khai.\n## Page 2\nB phủ nhận.'
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await extract_one({'doc_name': 'one', 'markdown': text}, client, args)
        self.assertEqual(result['situation_status'], 'passed')
        self.assertEqual(received, [strip_page_markers(text)]*2)
        self.assertTrue(all('## Page' not in source for source in received))


if __name__ == '__main__':
    unittest.main()
