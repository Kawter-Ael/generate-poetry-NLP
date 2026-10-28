"""Comparaison et contrôles simulés : aucun chargement de Qwen."""
import types
import unittest
from unittest.mock import Mock, patch

from poesie import check_poem, literary_review, compare_model_candidates, build_messages


def settings(**extra):
    return dict(theme="the sea",first_line="A quiet sea.",tone="Paisible",line_count=4,
        required_words=["shore"],temperature=.85,top_p=.8,repetition_penalty=1.2,
        no_repeat_ngram_size=4,max_new_tokens=175,allow_retries=True,auto_budget=True,
        selection_mode="economy",comparison_mode=False,comparison_seed=123,
        adapter_version="v2",model_comparison=True,**extra)


class ComparisonTests(unittest.TestCase):
    def test_paired_arguments_prompts_and_seeds_despite_different_failures(self):
        model=types.SimpleNamespace(active=None)
        switches=[]
        def activate(current,version):
            current.active=version;switches.append(version)
            return dict(version=version,label=version)
        seen=[]
        def generate(tokenizer,current,**kwargs):
            seen.append((current.active,dict(kwargs),build_messages(**{k:kwargs[k] for k in
                ("theme","first_line","line_count","required_words","correction","tone")})))
            content="A quiet sea.\nshore shore\nshore shore\nshore shore\nEXTRA" if current.active=="v1" else "A quiet sea.\nThe shore is warm.\nA gull flies home.\nWe rest together."
            return dict(content=content,token_limit_reached=current.active=="v1")
        result=compare_model_candidates(generate,None,model,settings(),activate)
        self.assertEqual(len(seen),6)
        self.assertEqual(switches,["v1","v2","v2"])
        for a,b in zip(seen[:3],seen[3:]):
            self.assertEqual(a[1:],b[1:])
        self.assertEqual([r[1]["seed"] for r in seen[:3]],[123,124,125])
        for _,args,_ in seen:
            self.assertEqual(args["correction"],"")
            self.assertEqual(args["max_new_tokens"],175)
            self.assertEqual(args["temperature"],.85)
            self.assertEqual(args["repetition_penalty"],1.2)
            self.assertEqual(args["no_repeat_ngram_size"],4)
        self.assertFalse(result["models"]["v1"]["validation"]["conformant"])
        self.assertTrue(result["models"]["v2"]["validation"]["conformant"])
        self.assertEqual(result["protocol"]["literary_winner"],"requires_human_review")

    def test_one_candidate_per_model_when_retries_disabled(self):
        opts=settings();opts["allow_retries"]=False
        generate=Mock(return_value="A quiet sea.\nThe shore is warm.\nA gull flies home.\nWe rest together.")
        result=compare_model_candidates(generate,None,None,opts,lambda _,v:dict(version=v))
        self.assertEqual(generate.call_count,2)
        self.assertEqual(result["protocol"]["maximum_calls"],2)

    def test_selected_adapter_restored_on_generation_error(self):
        activate=Mock(return_value={})
        with self.assertRaisesRegex(RuntimeError,"simulated"):
            compare_model_candidates(Mock(side_effect=RuntimeError("simulated")),None,None,settings(),activate)
        self.assertEqual(activate.call_args.args[1],"v2")

    def test_best_candidate_is_not_necessarily_first(self):
        good="A quiet sea.\nThe shore is warm.\nA gull flies home.\nWe rest together."
        generate=Mock(side_effect=["bad",good,"bad","bad",good,"bad"])
        result=compare_model_candidates(generate,None,None,settings(),lambda _,v:dict(version=v))
        for item in result["models"].values():
            self.assertEqual(item["content"],good)
            self.assertEqual(item["validation"]["selected_attempt"],2)

    def test_modes_cannot_be_combined_silently(self):
        opts=settings();opts["comparison_mode"]=True
        generate=Mock()
        with self.assertRaises(ValueError):compare_model_candidates(generate,None,None,opts,Mock())
        generate.assert_not_called()

    def test_whole_words_and_apostrophe_variants(self):
        report=check_poem("A heartfelt greeting.\nStarry skies.\nIt’s calm.\nOne shoreline.",4,["heart","stars","shore","it's"])
        self.assertEqual(report["present_words"],["it's"])
        self.assertEqual(report["missing_words"],["heart","stars","shore"])

    def test_dangling_punctuation_is_signal_not_creativity_score(self):
        for ending in ("We reach the shore,","We reach the shore;","We reach the shore:","We reach the shore—"):
            review=literary_review("A quiet sea.\n"+ending)
            self.assertEqual(review["ending_status"],"suspected_incomplete")
            self.assertIsNone(review["human_score"])
        self.assertEqual(literary_review("We reach the shore")["ending_status"],"no_signal")

    def test_no_verse_deleted_to_force_validity(self):
        raw="A quiet sea.\nA shore.\nA gull.\nA cloud.\nEXTRA"
        generate=Mock(return_value=raw)
        result=compare_model_candidates(generate,None,None,settings(),lambda _,v:dict(version=v))
        for item in result["models"].values():
            self.assertEqual(item["content"],raw)
            self.assertFalse(item["validation"]["line_count_ok"])


class ComparisonUITest(unittest.TestCase):
    def test_compare_export_and_save_peer_without_model_load(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from test_adaptateurs import FakePeft
        st.cache_resource.clear()
        fake=types.ModuleType("generer");model=FakePeft()
        fake.load_model=Mock(return_value=(object(),model))
        good="A quiet sea.\nThe golden waves return.\nA gull circles overhead.\nYour steps warm the sand.\nThe wind carries a song.\nWe follow silver trails.\nAn open shell reflects dawn.\nOur journey finds its home."
        fake.generate_poem=Mock(side_effect=lambda _,m,**kw:good if m.active_adapter=="v2" else "Moon\n"*8)
        with patch.dict("sys.modules",{"generer":fake}),patch("bibliotheque.save_poems",return_value=1) as save:
            at=AppTest.from_file("app.py",default_timeout=20).run()
            at.button[0].click().run()
            next(c for c in at.checkbox if c.label=="Comparer LoRA V1 et V2").check()
            at.button[0].click().run()
            self.assertFalse(at.exception)
            draft=at.session_state.draft
            self.assertFalse(draft["validation"]["conformant"])
            self.assertTrue(draft["adapter_comparison"]["poems"]["v2"]["validation"]["conformant"])
            self.assertEqual(fake.generate_poem.call_count,6)
            fake.load_model.assert_called_once()
            self.assertEqual(model.active_adapter,"v1")
            next(b for b in at.button if b.label=="Enregistrer le candidat V2").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(save.call_args.args[0][0]["model"]["version"],"v2")


if __name__=="__main__":unittest.main()
