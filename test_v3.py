"""Contrôles V3 sans modifier les adaptateurs ni la bibliothèque utilisateur."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import adaptateurs
from bibliotheque import load_poems, save_poems
from poesie import compare_model_candidates
from test_adaptateurs import FakePeft


class V3Tests(unittest.TestCase):
    def test_streamlit_v3_selection_generation_and_save(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        fake = types.ModuleType('generer')
        model = FakePeft()
        fake.load_model = Mock(return_value=(object(), model))
        fake.generate_poem = Mock(return_value='A quiet sea.\nThe golden waves return.\nA gull circles overhead.\nYour steps warm the sand.\nThe wind carries a song.\nWe follow silver trails.\nAn open shell reflects dawn.\nOur journey finds its home.')
        with patch.dict('sys.modules', {'generer': fake}), patch('bibliotheque.save_poems', return_value=1) as save:
            at = AppTest.from_file('app.py', default_timeout=20).run()
            at.button[0].click().run()
            for version in ('v1', 'v2', 'v3'):
                at.selectbox(key='adapter_version').set_value(version)
                at.button[0].click().run()
                self.assertFalse(at.exception)
                self.assertEqual(model.active_adapter, version)
                self.assertEqual(at.session_state.draft['model']['version'], version)
            next(b for b in at.button if b.label == 'Enregistrer le poème').click().run()
            self.assertFalse(at.exception)
            self.assertEqual(save.call_args.args[0][0]['model']['version'], 'v3')
            fake.load_model.assert_called_once()

    def test_three_distinct_weights_and_expected_hash(self):
        infos = [adaptateurs.adapter_info(v) for v in adaptateurs.ADAPTER_LABELS]
        self.assertEqual(len({i['adapter_sha256'] for i in infos}), 3)
        self.assertEqual(infos[2]['adapter_sha256'], adaptateurs.V3_SHA256)

    def test_nested_discovery_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / 'adapter_VERSION3' / 'export' / 'adapter_final'
            nested.mkdir(parents=True)
            config = json.loads((adaptateurs.PROJECT_ROOT / 'adapter_VERSION3' / 'adapter_config.json').read_text())
            (nested / 'adapter_config.json').write_text(json.dumps(config))
            (nested / 'adapter_model.safetensors').write_bytes(b'fake-test-weights')
            with self.assertRaisesRegex(ValueError, 'SHA-256 V3 incorrect'):
                adaptateurs.adapter_info('v3', root=tmp)
            with patch.object(adaptateurs, 'V3_SHA256', adaptateurs.fingerprint(nested / 'adapter_model.safetensors')):
                self.assertEqual(adaptateurs.adapter_info('v3', root=tmp)['adapter_path'], str(nested.resolve()))

    def test_missing_v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'V3 incomplet'):
                adaptateurs.adapter_info('v3', root=tmp)

    def test_switch_and_reuse(self):
        model = FakePeft()
        for v in ('v1', 'v2', 'v3', 'v1', 'v3'):
            adaptateurs.activate_adapter(model, v)
            self.assertEqual(model.active_adapter, v)
        self.assertEqual([entry[1] for entry in model.loads], ['v2', 'v3'])

    def test_v3_comparison_and_save_preserves_previous_records(self):
        model = FakePeft()
        settings = dict(adapter_version='v3', line_count=4, required_words=[],
                        allow_retries=False, first_line='', theme='sea', max_new_tokens=100,
                        temperature=.8, top_p=.9, repetition_penalty=1.15, no_repeat_ngram_size=3)
        generate = Mock(return_value='The sea is calm.\nA gull flies home.\nThe shore is warm.\nWe rest together.')
        result = compare_model_candidates(generate, None, model, settings, adaptateurs.activate_adapter)
        self.assertEqual(list(result['models']), ['v1', 'v2', 'v3'])
        self.assertEqual(model.active_adapter, 'v3')
        self.assertEqual(result['protocol']['maximum_calls'], 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'library.json'
            old = dict(id='old', content='An existing poem')
            save_poems([old], path)
            new = dict(id='v3', content=result['models']['v3']['content'], model=result['models']['v3']['model'])
            save_poems([new], path)
            self.assertEqual(load_poems(path), [old, new])


if __name__ == '__main__':
    unittest.main()
