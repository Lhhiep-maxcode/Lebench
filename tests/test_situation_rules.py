import json
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

from server.extract_situations_rules import (
    HEADINGS, clean_facts, extract_document, extract_judgment_date, filter_extracted, strip_page_markers,
)
from server.extract_situations_simple import extract_one
from server.situation_prompts import INPUT_SEPARATOR, JUDGE_SYSTEM_PROMPT


def judgment(facts='Bà A cho ông B vay 100.000.000đ.', application='Yêu cầu có căn cứ.',
             conclusion='1. Buộc ông B trả tiền.', facts_heading='NỘI DUNG VỤ ÁN',
             application_heading='NHẬN ĐỊNH CỦA TÒA ÁN', header='Ngày 13 tháng 10 năm 2023', category='Civil'):
    return {'id': 42, 'doc_name': 'test', 'category': category, 'markdown': (
        header + '\n' + facts_heading + ':\n' + facts + '\n'
        + application_heading + ':\n' + application + '\nQUYẾT ĐỊNH:\n' + conclusion
    )}


class RuleTests(unittest.TestCase):
    def test_three_sections_are_source_slices_and_input_is_preserved(self):
        row = judgment()
        row['situation'] = 'Existing field'
        before = dict(row)
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(row, before)
        self.assertEqual({k: v for k, v in result.items() if k != 'benchmark_seed'}, row)
        self.assertEqual(result['benchmark_seed'], {
            'facts': 'Bà A cho ông B vay 100.000.000đ.',
            'facts_clean': 'Bà A cho ông B vay 100.000.000đ.',
            'application': 'Yêu cầu có căn cứ.',
            'conclusion': '1. Buộc ông B trả tiền.',
            'judgment_date': '2023-10-13',
        })

    def test_every_heading_alias_and_xet_thay_as_application(self):
        for name, role in HEADINGS.items():
            with self.subTest(heading=name):
                row = judgment(**({'facts_heading': name} if role == 'facts' else
                                   {'application_heading': name} if role == 'application' else {}))
                result, status = extract_document(row)
                self.assertEqual(status, 'kept')
                self.assertEqual(result['benchmark_seed']['application'], 'Yêu cầu có căn cứ.')

    def test_missing_sections_are_filtered_without_facts_fallback(self):
        for missing in ('NỘI DUNG VỤ ÁN', 'NHẬN ĐỊNH CỦA TÒA ÁN', 'QUYẾT ĐỊNH'):
            row = judgment()
            row['markdown'] = row['markdown'].replace(missing, 'OTHER LABEL')
            self.assertEqual(extract_document(row), (None, 'missing_headings'))
        row = {'markdown': 'XÉT THẤY:\nCó căn cứ.\nQUYẾT ĐỊNH:\nChấp nhận.'}
        self.assertEqual(extract_document(row), (None, 'missing_headings'))

    def test_duplicate_sections_are_filtered(self):
        for field, heading in [('facts', 'NHẬN THẤY'), ('application', 'XÉT THẤY'),
                               ('conclusion', 'QUYẾT ĐỊNH')]:
            with self.subTest(field=field):
                row = judgment(**{field: 'Nội dung.\n' + heading + ':\nNội dung khác.'})
                self.assertEqual(extract_document(row), (None, 'duplicate_headings'))

    def test_wrong_order_and_empty_sections_are_filtered(self):
        row = judgment()
        row['markdown'] = row['markdown'].replace('NỘI DUNG VỤ ÁN', 'TEMP').replace(
            'NHẬN ĐỊNH CỦA TÒA ÁN', 'NỘI DUNG VỤ ÁN').replace('TEMP', 'NHẬN ĐỊNH CỦA TÒA ÁN')
        self.assertEqual(extract_document(row), (None, 'invalid_order'))
        for field in ('facts', 'application', 'conclusion'):
            self.assertEqual(extract_document(judgment(**{field: '  \n'})), (None, 'empty_section'))

    def test_decision_document_title_is_not_a_second_conclusion(self):
        row = judgment(header='Ngày 13/10/2023\nQUYẾT ĐỊNH\nGIẢI QUYẾT VIỆC DÂN SỰ')
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['conclusion'], '1. Buộc ông B trả tiền.')

    def test_case_references_and_nhan_danh_are_not_section_headings(self):
        facts = 'Bà A phản đối Quyết định số 58. Ông B viện dẫn nhận định của tòa án cấp trước.'
        row = judgment(facts=facts, header='NHÂN DANH\nNgày 13/10/2023')
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts'], facts)

    def test_flattened_and_unicode_multiline_headers_preserve_body(self):
        for facts_header, application_header in [
            ('NOI DUNG VU AN', 'NHAN DINH CUA TOA AN'),
            ('NỘI DUNG\nVỤ ÁN', 'NHẬN ĐỊNH CỦA TOÀ ÁN'),
            (unicodedata.normalize('NFD', 'NỘI DUNG VỤ ÁN'),
             unicodedata.normalize('NFD', 'NHẬN ĐỊNH CỦA TÒA ÁN')),
        ]:
            result, status = extract_document(judgment(
                facts_heading=facts_header, application_heading=application_header))
            self.assertEqual(status, 'kept')
            self.assertEqual(result['benchmark_seed']['facts'], 'Bà A cho ông B vay 100.000.000đ.')
        row = {'markdown': 'NỘI DUNG VỤ ÁN: A khai. NHẬN ĐỊNH CỦA TÒA ÁN: Lý do. QUYẾT ĐỊNH: 1. Bác.'}
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts'], 'A khai.')

    def test_unknown_ocr_spelling_is_not_guessed(self):
        row = judgment(facts_heading='NỘI D VỤ ÁN')
        self.assertEqual(extract_document(row), (None, 'missing_headings'))

    def test_clean_facts_removes_opinion_paragraph_and_everything_after_it(self):
        prefix = 'Bà A yêu cầu trả tiền.\n\nÔng B thừa nhận khoản vay.'
        suffix = ('\n\nĐại diện Viện kiểm sát phát biểu ý kiến: Đề nghị chấp nhận.\n\n'
                  'Bà A tiếp tục trình bày một thông tin khác.')
        raw = prefix + suffix
        result, status = extract_document(judgment(facts=raw))
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts'], raw)
        self.assertEqual(result['benchmark_seed']['facts_clean'], prefix)

    def test_clean_facts_handles_wrapped_paragraphs_and_prosecutor_variants(self):
        for opinion in (
            'Tại phiên tòa, đại diện Viện\nkiểm sát nhân dân huyện A có ý kiến như sau:',
            'Kiểm sát viên phát bi ểu ý ki ến về việc tuân theo pháp luật.',
            'Ý kiến của đại diện VKSND huyện A tại phiên tòa:',
            'Ý kiến phát biểu của Vị đại diện Viện kiểm sát huyện A:',
            '* Quan điểm của Viện kiểm sát nhân dân huyện A:',
            'Đại diện Viện kiểm sát huyện A thể hiện quan điểm:',
            'Đại diện Viện kiểm sát tham gia phiên tòa pht biểu\ný kiến về vụ án:',
            'Đại diện VKS huyện A đề nghị bác yêu cầu.',
            'Đại diện Vện kiểm sát huyện A đề nghị bác yêu cầu.',
            'Đại diện V.K.S.N.D. huyện A đề nghị bác yêu cầu.',
        ):
            with self.subTest(opinion=opinion):
                raw = 'Nguyên đơn trình bày\nkhoản vay chưa được trả.\n' + opinion + '\nPhần sau bị loại.'
                self.assertEqual(clean_facts(raw), 'Nguyên đơn trình bày\nkhoản vay chưa được trả.')

    def test_clean_facts_preserves_bare_mentions_and_party_requests(self):
        raw = ('Theo cáo trạng, Viện kiểm sát truy tố A về tội trộm cắp.\n'
               'Người bị hại đề nghị bồi thường.\n'
               'Kiểm sát viên tham gia phiên tòa theo quyết định phân công.')
        self.assertEqual(clean_facts(raw), raw)

    def test_final_filter_rejects_remaining_mentions_with_case_accents_and_ocr_spacing(self):
        for phrase in ('viện kiểm sát', 'VIỆN KIỂM SÁT', 'Vien kiem sat', 'Viện\nkiểm  sát',
                       'Vi ện ki ểm s át', unicodedata.normalize('NFD', 'Viện kiểm sát'),
                       'Vện kiểm sát', 'Việm kiểm sát', 'Vịn kiểm sát', 'Viện kiểm xát',
                       'V1ện k1ểm sát', 'VKS', 'vksnd', 'VKSNDTC', 'VKSNDCC', 'VKSTC',
                       'V.K.S.', 'V K S N D', 'Viện KS', 'Kiểm sát viên', 'KSV'):
            with self.subTest(phrase=phrase):
                self.assertEqual(extract_document(judgment(facts='Hồ sơ gửi đến ' + phrase + '.')),
                                 (None, 'prosecutor_in_facts_clean'))

    def test_filter_only_checks_clean_facts_and_preserves_all_retained_fields(self):
        row = {'id': 1, 'markdown': 'Viện kiểm sát đề nghị bác yêu cầu.',
               'benchmark_seed': {'facts': 'Viện kiểm sát phát biểu.', 'facts_clean': 'Ông A yêu cầu trả tiền.',
                                  'application': 'Viện kiểm sát tham gia phiên tòa.'}}
        self.assertEqual(filter_extracted(row), (row, 'kept'))
        for value in (None, [], {}, {'benchmark_seed': {}}, {'benchmark_seed': {'facts_clean': 123}}):
            self.assertEqual(filter_extracted(value), (None, 'invalid_input'))

    def test_empty_clean_facts_are_filtered_in_both_modes(self):
        self.assertEqual(extract_document(judgment(facts='Viện kiểm sát đề nghị chấp nhận.')),
                         (None, 'empty_facts_clean'))
        for value in ('', '  \n\t', '\u00a0'):
            self.assertEqual(filter_extracted({'benchmark_seed': {'facts_clean': value}}),
                             (None, 'empty_facts_clean'))

    def test_abbreviations_require_word_boundaries(self):
        row = {'benchmark_seed': {'facts_clean': 'Mã sản phẩm AVKSNDX; tài liệu KSVECTOR.'}}
        self.assertEqual(filter_extracted(row), (row, 'kept'))

    def test_criminal_resumes_at_final_statement_and_keeps_all_following_paragraphs(self):
        prefix = 'A lấy tài sản của B.\nB yêu cầu bồi thường.'
        prosecutor = ('Cáo trạng của VKSND truy tố A.\n\n'
                      'Đại diện Vện kiểm sát đề nghị phạt A 12 tháng tù.\n'
                      'Về vật chứng: Đề nghị tịch thu chiếc xe.')
        for marker in ('Lời nói sau cùng', 'Lời sau cùng', 'Lời cuối cùng', 'Lời nói cuối cùng',
                       'LỜI NÓI SAU CÙNG', 'Lời nói\nsau cùng'):
            with self.subTest(marker=marker):
                tail = (marker + ': Bị cáo xin giảm nhẹ so với đề nghị của VKS.\n\n'
                        'Bị hại xác nhận đã được bồi thường.')
                raw = prefix + '\n\n' + prosecutor + '\n\n' + tail
                result, status = extract_document(judgment(facts=raw, category='Criminal'))
                self.assertEqual(status, 'kept')
                self.assertEqual(result['benchmark_seed']['facts'], raw)
                self.assertEqual(result['benchmark_seed']['facts_clean'], prefix + '\n\n' + tail)

    def test_criminal_keeps_final_statement_that_is_its_first_prosecutor_mention(self):
        raw = 'A đã lấy tài sản.\nLời nói sau cùng: A xin chấp nhận đề nghị của VKSND.'
        result, status = extract_document(judgment(facts=raw, category='criminal'))
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts_clean'], raw)

    def test_criminal_final_statement_heading_and_wrapped_body_are_one_paragraph(self):
        prefix = 'A đã lấy tài sản.'
        tail = 'Lời nói sau cùng:\nBị cáo chấp nhận mức hình phạt mà Viện kiểm sát\nđã đề nghị.'
        raw = prefix + '\nVKS đề nghị xử phạt A.\n' + tail
        result, status = extract_document(judgment(facts=raw, category='CRIMINAL'))
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts_clean'], prefix + '\n\n' + tail)

    def test_criminal_without_later_final_statement_keeps_only_prefix(self):
        prefix = 'A đã lấy tài sản.'
        raw = prefix + '\nVKSND truy tố A.\nTài liệu khác phía sau.'
        result, status = extract_document(judgment(facts=raw, category='Criminal'))
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['facts_clean'], prefix)
        early = 'Lời nói sau cùng được nhắc trong tài liệu trước đó.\nA đã lấy tài sản.'
        self.assertEqual(clean_facts(early + '\nVKS đề nghị xử phạt A.\nPhần bị loại.', 'Criminal'), early)

    def test_final_statement_exception_is_only_for_criminal(self):
        raw = ('A đã lấy tài sản.\nVKS đề nghị xử phạt A.\n'
               'Lời nói sau cùng: A xin chấp nhận mức phạt VKS đề nghị.')
        for category in ('Civil', None, 'Criminal procedure'):
            result, status = extract_document(judgment(facts=raw, category=category))
            self.assertEqual(status, 'kept')
            self.assertEqual(result['benchmark_seed']['facts_clean'], 'A đã lấy tài sản.')
            row = {'category': category, 'benchmark_seed': {'facts_clean': raw.split('\n')[-1]}}
            self.assertEqual(filter_extracted(row), (None, 'prosecutor_in_facts_clean'))

    def test_criminal_final_statement_does_not_exempt_other_prosecutor_paragraphs(self):
        statement = 'Lời nói sau cùng: A xin giảm nhẹ so với đề nghị của VKS.'
        for facts_clean in ('VKSND truy tố A.\n' + statement,
                            statement + '\nVKS đề nghị thêm một nội dung khác.'):
            row = {'category': 'Criminal', 'benchmark_seed': {'facts_clean': facts_clean}}
            self.assertEqual(filter_extracted(row), (None, 'prosecutor_in_facts_clean'))

    def test_reextracting_restores_final_statement_from_original_markdown(self):
        raw = ('A đã lấy tài sản.\nVKS đề nghị xử phạt A.\n'
               'Lời cuối cùng: A xin được giảm nhẹ hình phạt.')
        row = judgment(facts=raw, category='Criminal')
        row['benchmark_seed'] = {'facts': raw, 'facts_clean': 'A đã lấy tài sản.'}
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertIn('Lời cuối cùng', result['benchmark_seed']['facts_clean'])
        self.assertEqual(row['benchmark_seed']['facts_clean'], 'A đã lấy tài sản.')

    def test_conclusion_excludes_recipient_lists_and_signatures(self):
        for footer in ('Nơi nhận:\n- Các bên;\nThẩm phán Nguyễn A',
                       'TM. HỘI ĐỒNG XÉT XỬ\nTHẨM PHÁN - CHỦ TỌA PHIÊN TÒA\nNguyễn A',
                       'THẨM PHÁN\nNguyễn A'):
            result, status = extract_document(judgment(conclusion='Buộc trả tiền.\n\n' + footer))
            self.assertEqual(status, 'kept')
            self.assertEqual(result['benchmark_seed']['conclusion'], 'Buộc trả tiền.')

    def test_page_markers_do_not_enter_sections_and_original_markdown_is_preserved(self):
        row = judgment(facts='A khai.\n## Page {2/3}\nB phủ nhận.')
        original = row['markdown']
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(result['markdown'], original)
        self.assertNotIn('Page', result['benchmark_seed']['facts'])
        self.assertEqual(strip_page_markers('trước## Page 1\nsau'), 'trước\n\nsau')

    def test_date_formats_and_calendar_validation(self):
        for text in ('Ngày 29 tháng 02 năm 2024', 'Ngày: 29/02/2024',
                     'Hà Nội, ngày 29-02-2024', 'Ngày 29.02.2024',
                     'Ngày 29 – 02 - 2024', 'Ngày 29 tháng 02 n ăm 20 24'):
            self.assertEqual(extract_judgment_date(text), '2024-02-29')
        for text in ('Ngày 29/02/2023', 'Ngày 31/04/2024', 'Không có ngày.',
                     'Ngày 01/01/2024\nNgày 02/01/2024'):
            self.assertIsNone(extract_judgment_date(text))

    def test_judgment_date_is_not_the_filing_date_or_unverified_metadata(self):
        header = ('Bản án số: 12/2024/DS-ST\nNgày: 20/09/2024\n'
                  'Ngày 20 tháng 9 năm 2024, tại trụ sở Tòa án xét xử sơ thẩm vụ án '
                  'thụ lý ngày 29 tháng 5 năm 2023.')
        row = judgment(header=header)
        row['issued_date'] = '2023-05-29'
        result, status = extract_document(row)
        self.assertEqual(status, 'kept')
        self.assertEqual(result['benchmark_seed']['judgment_date'], '2024-09-20')
        self.assertEqual(extract_judgment_date(header.split('\n', 2)[2]), '2024-09-20')
        self.assertIsNone(extract_judgment_date('Vụ án thụ lý ngày 29 tháng 5 năm 2023.'))

    def test_birth_dates_and_wrapped_procedural_dates_do_not_override_caption(self):
        header = ('Ngày: 03 - 3 - 2025\nNHÂN DANH\n'
                  'Ngày 03 tháng 3 năm 2025, tại trụ sở Tòa án xét xử sơ thẩm vụ án.\n'
                  'Quyết định đưa vụ án ra xét xử số: 47/2025/QĐXXST, ngày 12/02/2025,\n'
                  '+ Ông A, sinh ngày: 10/02/1988; địa chỉ: X.\n')
        self.assertEqual(extract_judgment_date(header), '2025-03-03')
        self.assertIsNone(extract_judgment_date('NHÂN DANH\nÔng A, sinh ngày: 10/02/1988; địa chỉ: X.'))

    def test_flattened_caption_and_hearing_without_location_word_tai(self):
        self.assertEqual(extract_judgment_date(
            'Thành phố Thủ Đức, ngày 18 tháng 12 năm 2024 QUYẾT ĐỊNH Nội dung.'), '2024-12-18')
        self.assertEqual(extract_judgment_date(
            'NHÂN DANH\nNgày 18 tháng 12 năm 2024, Tòa án mở phiên họp công khai.'), '2024-12-18')

    def test_invalid_markdown(self):
        for row in (None, [], {}, {'markdown': None}, {'markdown': ' '}):
            self.assertEqual(extract_document(row), (None, 'invalid_input'))

    def test_multiprocessing_cli_preserves_order_reports_bad_json_and_does_not_modify_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.jsonl'
            rows = [dict(judgment(), id=i) for i in range(5)]
            rows[2]['markdown'] = 'NỘI DUNG VỤ ÁN:\nKhông có đủ các phần.'
            source.write_text('\n'.join(json.dumps(row) for row in rows) + '\n{"id":', encoding='utf-8-sig')
            before = source.read_bytes()
            script = Path(__file__).resolve().parents[1] / 'server' / 'extract_situations_rules.py'
            outputs = []
            for workers in (1, 2):
                destination = Path(directory) / f'output-{workers}.jsonl'
                process = subprocess.run(
                    [sys.executable, '-B', str(script), '--input', str(source), '--output', str(destination),
                     '--workers', str(workers), '--chunksize', '1'],
                    capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(process.returncode, 1, process.stderr)
                self.assertIn('Line 6:', process.stderr)
                self.assertEqual(json.loads(process.stdout), {'kept': 4, 'missing_headings': 1, 'invalid_input': 1})
                outputs.append(destination.read_text(encoding='utf-8'))
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual([r['id'] for r in map(json.loads, outputs[0].splitlines())], [0, 1, 3, 4])
            self.assertEqual(source.read_bytes(), before)
            same = subprocess.run([sys.executable, '-B', str(script), '--input', str(source), '--output', str(source)],
                                  capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(same.returncode, 0)
            self.assertEqual(source.read_bytes(), before)

    def test_filter_only_cli_works_without_markdown_and_preserves_retained_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / 'input.jsonl', Path(directory) / 'filtered.jsonl'
            rows = [
                {'id': 1, 'benchmark_seed': {'facts_clean': 'Ông A yêu cầu trả tiền.', 'application': 'Viện kiểm sát.'}},
                {'id': 2, 'benchmark_seed': {'facts_clean': 'Văn bản gửi đến Viện kiểm sát.'}},
                {'id': 3, 'benchmark_seed': {'facts_clean': 'Bà B phản đối.'}, 'other': [1, 2]},
                {'id': 4, 'category': 'Criminal', 'benchmark_seed': {
                    'facts_clean': 'A đã lấy tài sản.\nLời nói sau cùng: A xin chấp nhận đề nghị của VKS.'}},
                {'id': 5, 'benchmark_seed': {'facts_clean': '  '}},
                {'id': 6, 'benchmark_seed': {'facts_clean': 'Hồ sơ gửi Vện kiểm sát.'}},
            ]
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8-sig')
            before = source.read_bytes()
            script = Path(__file__).resolve().parents[1] / 'server' / 'extract_situations_rules.py'
            process = subprocess.run(
                [sys.executable, '-B', str(script), '--input', str(source), '--output', str(destination),
                 '--filter-only', '--workers', '2', '--chunksize', '1', '--no-progress'],
                capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout), {
                'kept': 3, 'prosecutor_in_facts_clean': 2, 'empty_facts_clean': 1})
            self.assertEqual([json.loads(line) for line in destination.read_text(encoding='utf-8').splitlines()],
                             [rows[0], rows[2], rows[3]])
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
        self.assertEqual(received, [strip_page_markers(text)] * 2)


if __name__ == '__main__':
    unittest.main()
