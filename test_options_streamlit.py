"""Les options dépendantes réagissent sans lancer le modèle."""
import unittest
from unittest.mock import Mock, patch
import types

from streamlit.testing.v1 import AppTest


class OptionsTests(unittest.TestCase):
    def test_comparison_seed_enables_before_generation(self):
        fake = types.ModuleType('generer')
        fake.load_model = loader = Mock()
        with patch.dict('sys.modules', {'generer': fake}):
            app = AppTest.from_file('app.py', default_timeout=30).run()
            app.button[0].click().run()
            seed = next(w for w in app.number_input if w.label == 'Seed de comparaison')
            self.assertTrue(seed.disabled)
            toggle = next(w for w in app.checkbox if w.label == 'Comparer LoRA V1 et V2')
            toggle.check().run()
            self.assertFalse(app.exception)
            seed = next(w for w in app.number_input if w.label == 'Seed de comparaison')
            self.assertFalse(seed.disabled)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
