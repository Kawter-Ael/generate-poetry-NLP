"""Tests des deux adaptateurs avec modèle simulé ; aucun poids Qwen chargé."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import adaptateurs


class FakePeft:
    def __init__(self):
        self.peft_config={"v1":{}}
        self.active_adapter="v1"
        self.loads=[]
        self.switches=[]
    def load_adapter(self,path,adapter_name,is_trainable):
        self.loads.append((path,adapter_name,is_trainable))
        self.peft_config[adapter_name]={}
    def set_adapter(self,name,inference_mode=False):
        self.active_adapter=name
        self.switches.append((name,inference_mode))
    def eval(self):return self


class AdapterTests(unittest.TestCase):
    def test_actual_paths_base_and_weights_differ(self):
        a,b=adaptateurs.adapter_info("v1"),adaptateurs.adapter_info("v2")
        self.assertNotEqual(a["adapter_path"],b["adapter_path"])
        self.assertNotEqual(a["adapter_sha256"],b["adapter_sha256"])
        self.assertEqual(a["base_model"],"Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(a["base_model"],b["base_model"])
        self.assertEqual(Path(b["adapter_path"]).name,"adapter_V2")

    def test_switch_back_reuses_loaded_adapters_and_freezes(self):
        model=FakePeft()
        for version in ["v1","v2","v2","v1","v2"]:
            metadata=adaptateurs.activate_adapter(model,version)
            self.assertEqual(model.active_adapter,version)
            self.assertEqual(metadata["version"],version)
        self.assertEqual(len(model.loads),1)
        self.assertEqual(model.loads[0][1:],("v2",False))
        self.assertEqual([x[0] for x in model.switches],["v1","v2","v2","v1","v2"])
        self.assertTrue(all(frozen for _,frozen in model.switches))

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):adaptateurs.adapter_info("v4")

    def test_missing_adapter_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError,"V2 incomplet"):
                adaptateurs.adapter_info("v2",root=directory)

    def test_wrong_base_model_rejected_before_fingerprinting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Données factices uniquement dans le dossier temporaire du test.
            (root/"adapter_model.safetensors").write_bytes(b"simulated-test-data")
            (root/"adapter_config.json").write_text(json.dumps(dict(base_model_name_or_path="different/model")))
            with self.assertRaisesRegex(ValueError,"incompatible"):
                adaptateurs.adapter_info("v1",root=root)

    def test_changed_weights_require_restart_not_false_provenance(self):
        model=FakePeft();adaptateurs.activate_adapter(model,"v1")
        changed=adaptateurs.adapter_info("v1");changed["adapter_sha256"]="changed"
        with patch.object(adaptateurs,"adapter_info",return_value=changed):
            with self.assertRaisesRegex(RuntimeError,"Redémarrer"):
                adaptateurs.activate_adapter(model,"v1")

    def test_failed_load_does_not_activate_or_record_v2(self):
        model=FakePeft();model.load_adapter=Mock(side_effect=RuntimeError("incompatible tensor"))
        with self.assertRaises(RuntimeError):adaptateurs.activate_adapter(model,"v2")
        self.assertEqual(model.active_adapter,"v1")
        self.assertFalse(model.switches)

    def test_old_poem_has_no_invented_provenance(self):
        poem=dict(id="old",content="An old poem")
        original=copy.deepcopy(poem)
        self.assertEqual(adaptateurs.model_label(poem),"Version du modèle non renseignée")
        self.assertEqual(poem,original)

    def test_loader_uses_requested_path_and_same_base_without_real_model(self):
        import generer
        for version in ("v1","v2","v3"):
            model=Mock()
            with patch.object(generer.AutoTokenizer,"from_pretrained",return_value=object()) as tokenizer, \
                 patch.object(generer.AutoModelForCausalLM,"from_pretrained",return_value=object()) as base, \
                 patch.object(generer.PeftModel,"from_pretrained",return_value=model) as peft:
                generer.load_model(version)
            self.assertEqual(base.call_args.args[0],adaptateurs.BASE_MODEL)
            self.assertEqual(peft.call_args.args[1],adaptateurs.adapter_info(version)["adapter_path"])
            self.assertEqual(peft.call_args.kwargs["adapter_name"],version)
            self.assertFalse(peft.call_args.kwargs["is_trainable"])
            self.assertEqual(tokenizer.call_args.args[0],str(Path(generer.__file__).resolve().parent))


class StreamlitSelectionTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()

    def test_selection_cache_provenance_and_save_with_simulated_model(self):
        from streamlit.testing.v1 import AppTest
        fake=types.ModuleType("generer");model=FakePeft()
        fake.load_model=Mock(return_value=(object(),model))
        def generate(tokenizer,current,**kwargs):
            self.assertNotIn("adapter_version",kwargs)
            return ("A quiet sea.\nThe golden waves return.\nA gull circles overhead.\n"
                    "Your footsteps warm the sand.\nThe wind carries a song.\n"
                    "We follow silver trails.\nAn open shell reflects dawn.\n"
                    + ("Our journey finds its home." if current.active_adapter=="v1" else "The harbour welcomes us."))
        fake.generate_poem=Mock(side_effect=generate)
        with patch.dict("sys.modules",{"generer":fake}),patch("bibliotheque.save_poems",return_value=1) as save:
            at=AppTest.from_file("app.py",default_timeout=20).run()
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox(key="adapter_version").value,"v1")
            at.button[0].click().run()
            self.assertFalse(at.exception)
            first=copy.deepcopy(at.session_state.draft)
            self.assertEqual(first["model"]["version"],"v1")
            at.selectbox(key="adapter_version").set_value("v2")
            at.button[0].click().run()
            self.assertFalse(at.exception)
            second=at.session_state.draft
            self.assertEqual(second["model"]["version"],"v2")
            self.assertEqual(second["validation"]["model"],second["model"])
            self.assertNotEqual(first["model"]["adapter_sha256"],second["model"]["adapter_sha256"])
            self.assertNotEqual(first["content"],second["content"])
            self.assertEqual(fake.generate_poem.call_count,6)
            fake.load_model.assert_called_once()
            self.assertEqual(len(model.loads),1)
            next(b for b in at.button if b.label=="Enregistrer le poème").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(save.call_args.args[0][0]["model"],second["model"])
            at.selectbox(key="adapter_version").set_value("v1").run()
            self.assertEqual(at.session_state.draft["model"]["version"],"v2")
            self.assertTrue(any("LoRA V2" in c.value for c in at.caption))
            fake.load_model.assert_called_once()


if __name__=="__main__":unittest.main()
