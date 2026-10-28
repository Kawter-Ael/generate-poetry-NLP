"""Vérifie le pilote sans charger le modèle : cas fixes, rapports et notes humaines vides."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from evaluate_poetry import load_cases, save_json, build_report, PARAMETERS


class PilotTests(unittest.TestCase):
    def test_ten_cases_have_fixed_constraints(self):
        cases = load_cases(Path(__file__).with_name('evaluation_cases.json'))
        self.assertEqual(len(cases), 10)
        self.assertEqual(PARAMETERS['seed'], 42)
        self.assertEqual(PARAMETERS['line_count'], 8)
        self.assertEqual(len({case['tone'] for case in cases}), 7)

    def test_reports_use_common_evaluator_and_leave_human_ratings_blank(self):
        project = Path(__file__).resolve().parent
        cases = load_cases(project/'evaluation_cases.json')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save_json(output/'manifest.json', dict(cases=cases, parameters=PARAMETERS))
            records = []
            for case in cases:
                text = case['first_line'] + '\n' + '\n'.join([
                    'A quiet path awaits.', 'Soft waves touch the sand.', 'Our shadows meet at dawn.',
                    'One bird awakens nearby.', 'The wind carries a song.', 'We watch the open sky.',
                    'The garden greets us gently.'])
                records.append(dict(case_id=case['id'], result=dict(content=text, token_limit_reached=False),
                                    elapsed_seconds=.25, prompt_tokens=100))
            for phase in ['before', 'after']:
                save_json(output/(phase+'.json'), dict(status='complete', records=records))
            build_report(project, output)
            report = json.loads((output/'common_evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['records']), 20)
            self.assertEqual(report['human_evaluation'], 'not_performed')
            self.assertTrue(all(item['generation_calls'] == 10 for item in report['summaries']))
            with (output/'metrics.csv').open(encoding='utf-8') as stream:
                metrics = list(csv.DictReader(stream))
            self.assertEqual(len(metrics), 20)
            self.assertEqual({row['phase'] for row in metrics}, {'before', 'after'})
            with (output/'human_eval_blind.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(row[key] == '' for row in rows for key in ['coherence','fluency','imagery','tone','ending','comment']))
            self.assertTrue(all('phase' not in row for row in rows))


if __name__ == '__main__':
    unittest.main()
