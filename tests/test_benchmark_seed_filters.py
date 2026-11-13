import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from server.filter_benchmark_seeds import analyze_line, cutoff_metrics, percentile, select_reason


def seed_row(identifier, category, remaining_chars, facts_clean='Factual story.'):
    suffix = 'VKS ' + 'x' * (remaining_chars - 4) if remaining_chars else ''
    facts = 'Factual story.' + ('\n\n' + suffix if suffix else '')
    return {'id': identifier, 'doc_name': f'doc-{identifier}', 'category': category,
            'extra': ['preserved', identifier], 'benchmark_seed': {
                'facts': facts, 'facts_clean': facts_clean, 'application': 'Reasoning.',
                'conclusion': 'Decision.', 'judgment_date': '2024-01-02'}}


class CutoffTests(unittest.TestCase):
    def test_length_starts_at_paragraph_start_not_prosecutor_word(self):
        prefix = 'Events before the cutoff.\n\n'
        tail = 'Tại phiên tòa, đại diện Vện kiểm sát đề nghị bác yêu cầu.\nPhần tiếp theo.'
        metrics = cutoff_metrics(prefix + tail)
        self.assertEqual(metrics['first_prosecutor_paragraph_start'], len(prefix))
        self.assertEqual(metrics['remaining_chars'], len(tail))
        self.assertAlmostEqual(metrics['remaining_ratio'], len(tail) / len(prefix + tail))

    def test_no_prosecutor_means_zero_remaining_length(self):
        metrics = cutoff_metrics('Bà A yêu cầu trả tiền.\nÔng B không đồng ý.')
        self.assertIsNone(metrics['first_prosecutor_paragraph_start'])
        self.assertEqual(metrics['remaining_chars'], 0)
        self.assertEqual(metrics['remaining_ratio'], 0)

    def test_original_suffix_includes_restored_criminal_final_statement(self):
        prefix = 'Bị cáo lấy tài sản.\n\n'
        block = 'VKS đề nghị xử phạt.\n\n'
        tail = 'Lời nói sau cùng: Bị cáo xin giảm nhẹ.\nBị hại đã nhận bồi thường.'
        facts = prefix + block + tail
        clean = prefix + tail
        metrics = cutoff_metrics(facts)
        self.assertEqual(metrics['remaining_chars'], len(block + tail))
        self.assertNotEqual(metrics['remaining_chars'], len(facts) - len(clean))

    def test_linear_percentile_and_ties(self):
        self.assertEqual(percentile([10, 20, 30, 40, 50], 10), 14)
        self.assertEqual(percentile([10, 20, 30], 0), 10)
        self.assertEqual(percentile([10, 20, 30], 100), 30)
        self.assertEqual(percentile([15], 10), 15)
        self.assertIsNone(percentile([], 10))
        threshold = percentile([10, 10, 10], 10)
        audit = {'reason': 'candidate', 'remaining_chars': 10, 'prosecutor_in_facts_clean': False}
        self.assertEqual(select_reason(audit, threshold), 'kept')

    def test_final_gate_rejects_all_supported_mentions_even_criminal_last_words(self):
        for phrase in ('Viện kiểm sát', 'Vện kiểm sát', 'VKS', 'V.K.S.N.D.', 'Kiểm sát viên'):
            row = seed_row(1, 'Criminal', 100, f'Lời nói sau cùng: Bị cáo chấp nhận đề nghị của {phrase}.')
            audit = analyze_line((1, json.dumps(row)))
            self.assertTrue(audit['prosecutor_in_facts_clean'])
            self.assertEqual(select_reason(audit, 100), 'prosecutor_in_facts_clean')
            self.assertEqual(row['benchmark_seed']['facts_clean'],
                             f'Lời nói sau cùng: Bị cáo chấp nhận đề nghị của {phrase}.')

    def test_blank_clean_and_missing_original_facts_are_not_selected(self):
        blank = analyze_line((1, json.dumps(seed_row(1, 'Civil', 100, ' \n\t'))))
        self.assertEqual(select_reason(blank, None), 'empty_facts_clean')
        missing = analyze_line((2, json.dumps({'benchmark_seed': {'facts_clean': 'Known facts.'}})))
        self.assertEqual(missing['reason'], 'invalid_input')
        self.assertIsNone(missing['remaining_chars'])

    def test_clean_text_only_is_checked_for_final_mentions(self):
        row = seed_row(1, 'Civil', 10)
        row['benchmark_seed']['application'] = 'Viện kiểm sát đề nghị xử phạt.'
        audit = analyze_line((1, json.dumps(row)))
        self.assertFalse(audit['prosecutor_in_facts_clean'])
        self.assertEqual(select_reason(audit, 10), 'kept')

    def test_category_percentiles_multiprocessing_reports_and_row_fidelity(self):
        script = Path(__file__).resolve().parents[1] / 'server' / 'filter_benchmark_seeds.py'
        rows = [seed_row(i + 1, 'Civil' if i % 2 == 0 else ' civil ', length)
                for i, length in enumerate((10, 20, 30, 40, 50))]
        rows += [seed_row(i + 6, 'Criminal', length)
                 for i, length in enumerate((100, 200, 300, 400, 500))]
        rows.append(seed_row(11, None, 0))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.jsonl'
            source.write_text('\n' + ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8-sig')
            before = source.read_bytes()
            outputs = []
            for workers in (1, 2):
                destination = Path(directory) / f'result-{workers}.jsonl'
                process = subprocess.run(
                    [sys.executable, '-B', str(script), '--input', str(source), '--output', str(destination),
                     '--cutoff-percentile', '10', '--workers', str(workers), '--chunksize', '1', '--no-progress'],
                    capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout), {'kept': 3, 'cutoff_above_percentile': 8})
                outputs.append(destination.read_text(encoding='utf-8'))
                retained = [json.loads(line) for line in outputs[-1].splitlines()]
                self.assertEqual(retained, [rows[0], rows[5], rows[10]])
                with destination.with_suffix('.cutoff.csv').open(encoding='utf-8-sig', newline='') as stream:
                    audit = list(csv.DictReader(stream))
                self.assertEqual(len(audit), len(rows))
                self.assertEqual(audit[0]['line_number'], '2')
                self.assertEqual(audit[0]['remaining_chars'], '10')
                self.assertEqual(audit[0]['category_threshold_chars'], '14.0')
                self.assertEqual(audit[1]['reason'], 'cutoff_above_percentile')
                summary = json.loads(destination.with_suffix('.cutoff_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['categories']['civil']['percentile_population'], 5)
                self.assertEqual(summary['categories']['criminal']['threshold_remaining_chars'], 140)
                self.assertEqual(summary['categories']['uncategorized']['threshold_remaining_chars'], 0)
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(source.read_bytes(), before)

    def test_100th_percentile_still_rejects_mentions_empty_clean_and_bad_json(self):
        script = Path(__file__).resolve().parents[1] / 'server' / 'filter_benchmark_seeds.py'
        rows = [seed_row(1, 'Civil', 10), seed_row(2, 'Civil', 100),
                seed_row(3, 'Criminal', 100, 'Lời cuối cùng: Xin chấp nhận đề nghị của VKS.'),
                seed_row(4, 'Civil', 10, ' ')]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.jsonl', Path(directory) / 'selected.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows) + '{"id":', encoding='utf-8')
            before = source.read_bytes()
            process = subprocess.run(
                [sys.executable, '-B', str(script), '--input', str(source), '--output', str(output),
                 '--cutoff-percentile', '100', '--workers', '2', '--no-progress'],
                capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(process.returncode, 1)
            self.assertIn('Line 5:', process.stderr)
            self.assertEqual(json.loads(process.stdout), {
                'kept': 2, 'prosecutor_in_facts_clean': 1, 'empty_facts_clean': 1, 'invalid_input': 1})
            self.assertEqual([json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()], rows[:2])
            self.assertEqual(source.read_bytes(), before)

    def test_report_cannot_overwrite_input_and_invalid_percentiles_are_rejected(self):
        script = Path(__file__).resolve().parents[1] / 'server' / 'filter_benchmark_seeds.py'
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'input.jsonl', Path(directory) / 'selected.jsonl'
            source.write_text(json.dumps(seed_row(1, 'Civil', 10)) + '\n', encoding='utf-8')
            before = source.read_bytes()
            for flags in (['--report', str(source)], ['--cutoff-percentile', '-1'],
                          ['--cutoff-percentile', '101'], ['--cutoff-percentile', 'nan']):
                process = subprocess.run(
                    [sys.executable, '-B', str(script), '--input', str(source), '--output', str(output), *flags],
                    capture_output=True, text=True, encoding='utf-8')
                self.assertNotEqual(process.returncode, 0)
                self.assertEqual(source.read_bytes(), before)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
