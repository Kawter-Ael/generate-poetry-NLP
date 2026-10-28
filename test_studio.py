"""Vérifications légères : aucun chargement de modèle, aucun téléchargement."""
import tempfile
import unittest
import types
import csv
import io
from pathlib import Path
from unittest.mock import Mock, patch

from bibliotheque import import_records, load_poems, save_poems
from poesie import (build_messages, check_poem, complete_poem, generate_checked,
                    parse_words, repetition_report, token_budget, literary_review,
                    human_comparison_csv, HUMAN_RUBRIC, TONES)


class PoetryTests(unittest.TestCase):
    def test_four_prompt_combinations(self):
        for theme, first in [("Ocean", ""), ("", "A quiet moon"), ("Ocean", "A quiet moon"), ("", "")]:
            with self.subTest(theme=theme, first=first):
                messages = build_messages(theme, first, 4, ["stars"])
                prompt = messages[1]["content"]
                self.assertEqual("Theme:" in prompt, bool(theme))
                self.assertEqual("first line is already provided" in prompt, bool(first))
                self.assertEqual("Choose your own subject freely." in prompt, not theme and not first)
                self.assertIn("exactly 4", prompt)
                self.assertIn('"stars"', prompt)

    def test_words_are_whole_and_case_insensitive(self):
        self.assertEqual(parse_words(" moon, MOON, stars, heart "), ["moon", "stars", "heart"])
        report = check_poem("MOON and stars\nA dreamer", 2, ["moon", "star", "dream", "stars"])
        self.assertEqual(report["present_words"], ["moon", "stars"])
        self.assertEqual(report["missing_words"], ["star", "dream"])
        with self.assertRaises(ValueError):
            parse_words("one,two,three,four,five,six,seven,eight,nine")
        with self.assertRaises(ValueError):
            parse_words("two words")

    def test_exact_first_line_and_no_repetition(self):
        first = "  A quiet moon.  "
        content = complete_poem("A quiet moon.\nWaves return.\nA quiet moon.\nStars glow.\nNight falls.", first)
        self.assertEqual(content.splitlines()[0], first)
        report = check_poem(content, 4, [], first)
        self.assertFalse(report["conformant"])
        self.assertEqual(report["actual_lines"], 6)
        self.assertFalse(report["first_line_ok"])
        self.assertEqual(content.count("A quiet moon."), 3)

    def test_retry_is_bounded_and_does_not_cut_lines(self):
        settings = dict(theme="", first_line="", line_count=4, required_words=["moon"],
                        temperature=0.8, top_p=0.9, max_new_tokens=200)
        generate = Mock(side_effect=["A moon\nA sea", "A moon\nA sea\nA sky\nA dream", "A moon\nA sea\nA sky\nA dream"])
        content, report = generate_checked(generate, None, None, settings)
        self.assertTrue(report["conformant"])
        self.assertEqual(report["attempts"], 3)
        self.assertEqual(report["selected_attempt"], 2)
        self.assertEqual(generate.call_count, 3)
        generate = Mock(return_value="A moon\nA sea\nA sky\nA dream\nAn extra complete verse")
        content, report = generate_checked(generate, None, None, settings)
        self.assertFalse(report["conformant"])
        self.assertEqual(report["actual_lines"], 5)
        self.assertTrue(content.endswith("An extra complete verse"))
        self.assertEqual(generate.call_count, 3)
        generate = Mock(return_value={"content": "A moon\nA sea\nA sky\nA dream", "token_limit_reached": True})
        _, report = generate_checked(generate, None, None, settings)
        self.assertFalse(report["conformant"])
        self.assertTrue(report["token_limit_reached"])

    def test_repeated_words_and_lines_are_rejected_even_with_all_constraints(self):
        first = "The moon shines upon your smile"
        content = "\n".join([first, "My heart is yours", "The stars shine upon your eyes",
                              "You are the one I want to see", "Forever, forever, FOREVER",
                              "I want to see forever, forever, forever",
                              "The stars shine upon your eyes", "We rest beside the sea"])
        report = check_poem(content, 8, ["heart", "stars", "forever"], first)
        self.assertTrue(report["line_count_ok"])
        self.assertEqual(report["missing_words"], [])
        self.assertFalse(report["conformant"])
        self.assertFalse(report["repetitions_ok"])
        self.assertTrue(report["repetitions"]["repeated_lines"])
        self.assertTrue(any(r["phrase"] == "forever" and r["occurrences"] == 3
                            for r in report["repetitions"]["word_runs"]))

    def test_repeated_blocks_and_phrases_are_normalized(self):
        report = repetition_report(["Bright waves!", "Soft wind.", "BRIGHT WAVES", "soft wind?"])
        self.assertFalse(report["ok"])
        self.assertTrue(any(b == {"starts": [1, 3], "length": 2} for b in report["repeated_blocks"]))
        phrase = repetition_report(["I want to see, I want to see."])
        self.assertTrue(any(r["phrase"] == "i want to see" for r in phrase["word_runs"]))

    def test_non_adjacent_blocks_and_cross_line_word_runs(self):
        report = repetition_report(["Silver tides", "Warm sand", "A quiet hill",
                                    "Silver tides", "Warm sand"])
        self.assertTrue(any(b["starts"] == [1, 4] and b["length"] == 2 for b in report["repeated_blocks"]))
        self.assertFalse(repetition_report(["We dream forever,", "FOREVER beside you"])["ok"])

    def test_repeated_candidate_triggers_retry_with_adapted_parameters(self):
        settings = dict(theme="", first_line="", line_count=4, required_words=["moon"],
                        temperature=.8, top_p=.9, max_new_tokens=200)
        bad = "A moon\nA moon\nA moon\nA moon"
        good = "A moon\nA sea\nA sky\nA dream"
        generate = Mock(side_effect=[bad, bad, good])
        content, report = generate_checked(generate, None, None, settings)
        self.assertEqual(content, good)
        self.assertTrue(report["conformant"])
        self.assertEqual(report["selected_attempt"], 3)
        self.assertEqual([c.kwargs["repetition_penalty"] for c in generate.call_args_list], [1.15, 1.2, 1.25])
        self.assertEqual([c.kwargs["no_repeat_ngram_size"] for c in generate.call_args_list], [3, 3, 3])
        self.assertIn("Repeated line positions", generate.call_args_list[1].kwargs["correction"])
        self.assertEqual(len(report["attempt_history"]), 3)
        self.assertTrue(all("parameters" in h and "score" in h for h in report["attempt_history"]))

    def test_best_invalid_candidate_has_no_repeats_and_keeps_extra_verse(self):
        bad = "A moon\nA moon\nA moon\nA moon"
        better = "A moon\nA sea\nA sky\nA dream\nAn extra complete verse"
        settings = dict(theme="", first_line="", line_count=4, required_words=["moon"],
                        temperature=.8, top_p=.9, max_new_tokens=200)
        generate = Mock(side_effect=[bad, better, bad])
        content, report = generate_checked(generate, None, None, settings)
        self.assertEqual(content, better)
        self.assertEqual(report["selected_attempt"], 2)
        self.assertEqual(report["actual_lines"], 5)
        self.assertFalse(report["conformant"])
        self.assertEqual(report["attempt_budgets"], [200, 200, 160])

    def test_all_repeated_attempts_fail_and_retry_can_be_disabled(self):
        settings = dict(theme="", first_line="", line_count=4, required_words=[],
                        temperature=.8, top_p=.9, max_new_tokens=200, repetition_penalty=1.3)
        generate = Mock(return_value="Moon\nMoon\nMoon\nMoon")
        _, report = generate_checked(generate, None, None, settings)
        self.assertFalse(report["conformant"])
        self.assertEqual(generate.call_count, 3)
        self.assertTrue(all(c.kwargs["repetition_penalty"] == 1.3 for c in generate.call_args_list))
        generate.reset_mock()
        _, report = generate_checked(generate, None, None, {**settings, "allow_retries": False})
        self.assertEqual(generate.call_count, 1)
        self.assertFalse(report["conformant"])

    def test_token_budget_adapts_only_when_appropriate(self):
        settings = dict(theme="", first_line="", line_count=4, required_words=[],
                        temperature=.8, top_p=.9, max_new_tokens=200)
        generate = Mock(return_value={"content": "Moon\nMoon\nMoon\nMoon", "token_limit_reached": True})
        _, report = generate_checked(generate, None, None, settings)
        self.assertEqual(report["attempt_budgets"], [200, 200, 200])
        generate = Mock(return_value={"content": "A moon\nA sea", "token_limit_reached": True})
        _, report = generate_checked(generate, None, None, settings)
        self.assertEqual(report["attempt_budgets"], [200, 250, 300])
        _, report = generate_checked(generate, None, None, {**settings, "auto_budget": False})
        self.assertEqual(report["attempt_budgets"], [200, 200, 200])
        self.assertEqual(token_budget(200, 8), 200)

    def test_wrong_first_line_and_missing_whole_words_are_rejected(self):
        report = check_poem("A different moon\nA heartfelt wish\nA starry sky\nA quiet dream", 4,
                            ["heart", "stars", "forever"], "The moon shines upon your smile")
        self.assertFalse(report["conformant"])
        self.assertFalse(report["first_line_ok"])
        self.assertEqual(report["missing_words"], ["heart", "stars", "forever"])


class LiteraryTests(unittest.TestCase):
    def settings(self, **changes):
        return dict(dict(theme="Love", tone="Romantique", first_line="", line_count=4,
                         required_words=["moon"], temperature=.8, top_p=.9, max_new_tokens=200,
                         allow_retries=True), **changes)

    def test_eight_line_prompt_has_progression_and_natural_rhythm(self):
        first = "  The moon shines upon your smile  "
        prompt = build_messages("Love", first, 8, ["heart", "stars", "forever"], tone="Romantique")[1]["content"]
        for text in ("lines 1-2", "lines 3-4", "lines 5-6", "lines 7-8", "7 NEW non-empty lines",
                     "one central image", "no minimum word count", "natural word order", "shared gestures",
                     "unless explicitly requested", "Do not print this plan"):
            self.assertIn(text, prompt)
        self.assertNotIn("aim for 6 to 12", prompt)
        self.assertIn('  The moon shines upon your smile  ', prompt)
        for size in (4, 12, 16):
            other = build_messages(line_count=size)[1]["content"]
            self.assertNotIn("lines 7-8", other)
            self.assertIn(f"exactly {size}", other)

    def test_each_tone_keeps_its_specific_instructions(self):
        for tone, instructions in TONES.items():
            prompt = build_messages(tone=tone)[1]["content"]
            self.assertIn(instructions, prompt)
        free = build_messages(tone="Libre")[1]["content"]
        self.assertNotIn("mutual companionship", free)

    def test_style_signals_are_soft_and_do_not_certify_coherence(self):
        text = "A moon.\nA sea.\nA sky.\nI wait and"
        formal = check_poem(text, 4, ["moon"])
        review = literary_review(text, "Romantique")
        self.assertTrue(formal["conformant"])
        self.assertTrue(any(item["code"] == "suspended_ending" for item in review["warnings"]))
        self.assertIsNone(review["human_score"])
        self.assertEqual(review["coherence_status"], "requires_human_review")
        self.assertEqual(review["human_quality_status"], "not_evaluated")
        self.assertEqual(len(review["human_rubric"]), 5)
        self.assertTrue(all(item["range"] == [0, 2] for item in review["human_rubric"]))

    def test_long_sentence_across_short_lines_is_only_a_reading_signal(self):
        text = "\n".join(" ".join(f"word{i}" for i in range(start, start + 8)) for start in range(0, 32, 8))
        self.assertTrue(check_poem(text, 4, [])["conformant"])
        review = literary_review(text)
        self.assertIn("long_sentence", [item["code"] for item in review["warnings"]])
        self.assertEqual(review["line_word_counts"], [8, 8, 8, 8])

    def test_review_flags_familiar_phrasing_without_declaring_it_bad_poetry(self):
        review = literary_review("My heart skips a beat.\nI see the shore.\nI see your hand.\nI see our home.")
        codes = [item["code"] for item in review["warnings"]]
        self.assertIn("familiar_expression", codes)
        self.assertIn("similar_openings", codes)
        self.assertNotIn("conformant", review)

    def test_three_formal_candidates_are_compared_instead_of_accepting_the_first(self):
        weak = "A moon.\nA sea.\nA sky.\nI wait and"
        fewer_signals = "A moon.\nA sea.\nA sky.\nWe walk home."
        generate = Mock(side_effect=[weak, fewer_signals, weak])
        content, report = generate_checked(generate, None, None, self.settings())
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(content, fewer_signals)
        self.assertEqual(report["selected_attempt"], 2)
        self.assertTrue(report["conformant"])
        self.assertEqual(report["literary_review"]["warning_count"], 0)
        self.assertEqual([c.kwargs["repetition_penalty"] for c in generate.call_args_list], [1.15] * 3)

    def test_formal_conformity_always_beats_a_clean_style_report(self):
        valid = "A moon.\nA sea.\nA sky.\nI wait and"
        invalid = "A moon.\nA sea.\nWe walk home."
        generate = Mock(side_effect=[invalid, valid, invalid])
        content, report = generate_checked(generate, None, None, self.settings())
        self.assertEqual(content, valid)
        self.assertTrue(report["conformant"])
        self.assertGreater(report["literary_review"]["warning_count"], 0)

    def test_required_words_and_exact_opening_remain_hard_constraints(self):
        first = "  The moon shines upon your smile  "
        good = first + "\nMy heart rests here.\nThe stars guide us.\nForever opens like dawn."
        bad = "The moon shines upon your smile\nMy heartfelt wish.\nA starry night.\nWe walk home."
        generate = Mock(side_effect=[bad, good, bad])
        content, report = generate_checked(generate, None, None,
            self.settings(first_line=first, required_words=["heart", "stars", "forever"]))
        self.assertEqual(content, good)
        self.assertEqual(content.splitlines()[0], first)
        self.assertTrue(report["first_line_ok"])
        self.assertEqual(report["missing_words"], [])

    def test_controlled_comparison_has_three_calls_no_hidden_adaptation(self):
        generate = Mock(return_value={"content": "moon\nmoon\nmoon\nmoon\nEXTRA", "token_limit_reached": True})
        _, report = generate_checked(generate, None, None,
            self.settings(comparison_mode=True, comparison_seed=314, temperature=1.2,
                          top_p=.6, repetition_penalty=1.3, no_repeat_ngram_size=6))
        calls = [call.kwargs for call in generate.call_args_list]
        self.assertEqual(len(calls), 3)
        self.assertEqual([c["temperature"] for c in calls], [.7, .75, .8])
        for key, value in {"top_p": .9, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3,
                           "seed": 314, "max_new_tokens": 200, "correction": ""}.items():
            self.assertEqual([c[key] for c in calls], [value] * 3)
        self.assertFalse(report["conformant"])
        self.assertEqual(report["actual_lines"], 5)
        self.assertEqual(report["experiment"]["human_evaluation"], "not_performed")
        self.assertEqual(report["experiment"]["literary_improvement"], "not_established")

    def test_comparison_cannot_bypass_single_call_choice(self):
        generate = Mock(return_value="A moon\nA sea\nA sky\nA dream")
        with self.assertRaises(ValueError):
            generate_checked(generate, None, None, self.settings(comparison_mode=True, allow_retries=False))
        generate.assert_not_called()
        _, report = generate_checked(generate, None, None, self.settings(allow_retries=False))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report["attempts"], 1)

    def test_human_grid_preserves_candidates_and_leaves_all_ratings_blank(self):
        generate = Mock(return_value="A moon.\nA sea.\nA sky.\nA dream.")
        _, report = generate_checked(generate, None, None, self.settings(comparison_mode=True))
        rows = list(csv.DictReader(io.StringIO(human_comparison_csv(report))))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["candidate"] for row in rows], ["A", "B", "C"])
        self.assertTrue(all(row[key] == "" for row in rows for key, _, _ in HUMAN_RUBRIC))
        self.assertTrue(all(row["comment"] == "" for row in rows))
        self.assertTrue(all("temperature" not in row for row in rows))
        self.assertEqual([h["attempt"] for h in report["attempt_history"]], [1, 2, 3])


class EndingAndEconomyTests(unittest.TestCase):
    def settings(self, **changes):
        return dict(dict(theme='', tone='Libre', first_line='', line_count=4, required_words=['moon'],
                         temperature=.8, top_p=.9, max_new_tokens=200, allow_retries=True), **changes)

    def test_finished_familiar_expression_beats_suspended_ending(self):
        broken = 'A moon.\nA sea.\nA sky.\nI wait and'
        complete = 'A moon.\nA sea.\nA sky.\nYou are my everything.'
        generate = Mock(side_effect=[broken, complete, broken])
        content, report = generate_checked(generate, None, None, self.settings())
        self.assertEqual(content, complete)
        self.assertEqual(report['selected_attempt'], 2)
        self.assertTrue(report['conformant'])

    def test_recurring_phrases_in_different_lines_are_soft_signals(self):
        text = 'We watch silver clouds drift.\nYou let silver clouds drift.\nThey see silver clouds drift.\nA moon brings us home.'
        review = literary_review(text)
        self.assertTrue(check_poem(text, 4, ['moon'])['conformant'])
        self.assertTrue(any(item['phrase'] == 'silver clouds drift' for item in review['recurring_phrases']))
        self.assertIn('recurring_phrase', [w['code'] for w in review['warnings']])

    def test_unclosed_quote_is_flagged_without_rewriting_text(self):
        text = 'A moon.\nA sea.\nA sky.\nShe whispers "come home'
        report = literary_review(text)
        self.assertEqual(report['ending_status'], 'suspected_incomplete')
        self.assertIn('unclosed_delimiter', [w['code'] for w in report['warnings']])
        self.assertIsNone(report['human_score'])

    def test_economy_stops_after_clean_candidate_and_keeps_retry_for_warning(self):
        clean = 'A moon.\nA sea.\nA sky.\nWe walk home.'
        generate = Mock(return_value=clean)
        _, report = generate_checked(generate, None, None, self.settings(selection_mode='economy'))
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(report['attempts'], 1)
        generate = Mock(side_effect=['A moon.\nA sea.\nA sky.\nI wait and', clean])
        _, report = generate_checked(generate, None, None, self.settings(selection_mode='economy'))
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(report['selected_attempt'], 2)

    def test_economy_never_overrides_controlled_comparison_or_three_call_ceiling(self):
        generate = Mock(return_value='A moon.\nA sea.\nA sky.\nWe walk home.')
        _, report = generate_checked(generate, None, None, self.settings(selection_mode='economy', comparison_mode=True))
        self.assertEqual(generate.call_count, 3)
        generate = Mock(return_value='moon\nmoon\nmoon\nmoon')
        _, report = generate_checked(generate, None, None, self.settings(selection_mode='economy'))
        self.assertEqual(generate.call_count, 3)
        self.assertFalse(report['conformant'])


class GeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import generer
        cls.torch = torch
        cls.module = generer

    def fake_pair(self, continuation, invoke_stopper=False, eos=True):
        torch = self.torch

        class Tokenizer:
            eos_token_id = 0

            def __init__(self):
                self.decoded = []

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "USER\nASSISTANT\n"

            def __call__(self, prompt, **kwargs):
                self.prompt = prompt
                return {"input_ids": torch.tensor([[101, 102, 103]])}

            def decode(self, ids, **kwargs):
                self.decoded.append(ids.tolist())
                return "".join(chr(i - 1000) for i in ids.tolist() if i >= 1000)

        tokenizer = Tokenizer()
        model = torch.nn.Linear(1, 1)
        model.generation_config = types.SimpleNamespace(eos_token_id=[0])

        def simulate(**kwargs):
            suffix = [1000 + ord(c) for c in continuation] + ([0] if eos else [])
            output = torch.cat((kwargs["input_ids"], torch.tensor([suffix])), dim=1)
            if invoke_stopper:
                kwargs["stopping_criteria"](output, None)
            return output

        model.generate = Mock(side_effect=simulate)
        return tokenizer, model

    def test_real_generate_function_forwards_parameters_and_decodes_only_new_tokens(self):
        continuation = "A wave rises.\nOur hearts meet.\nWe walk home."
        tokenizer, model = self.fake_pair(continuation)
        with patch.object(self.module, "load_model", side_effect=AssertionError("Qwen must not load")):
            result = self.module.generate_poem(tokenizer, model, first_line="First verse", line_count=4,
                                                temperature=.75, top_p=.85, max_new_tokens=150,
                                                repetition_penalty=1.2, no_repeat_ngram_size=3,
                                                return_metadata=True)
        actual = model.generate.call_args.kwargs
        for key, expected in dict(temperature=.75, top_p=.85, max_new_tokens=150,
                                  repetition_penalty=1.2, no_repeat_ngram_size=3, do_sample=True).items():
            self.assertEqual(actual[key], expected)
        self.assertEqual(tokenizer.prompt.count("First verse"), 1)
        self.assertEqual(result["content"], "First verse\n" + continuation)
        self.assertEqual(result["raw_continuation"], continuation)
        self.assertTrue(all(i not in [101, 102, 103] for call in tokenizer.decoded for i in call))

    def test_line_stopper_waits_for_boundary_and_keeps_overflow(self):
        tokenizer, model = self.fake_pair("A wave\nA shell\nA shore\nEXTRA", invoke_stopper=True)
        result = self.module.generate_poem(tokenizer, model, first_line="First", line_count=4,
                                          return_metadata=True)
        self.assertEqual(result["stop_reason"], "line_overflow")
        self.assertTrue(result["content"].endswith("EXTRA"))
        self.assertFalse(check_poem(result["content"], 4, [], "First")["conformant"])
        tokenizer, _ = self.fake_pair("")
        stop = self.module.PoemLineStoppingCriteria(tokenizer, 3, 4, "First")
        partial = "A wave\nA shell\nA final thought"
        def ids(text):
            return self.torch.tensor([[101, 102, 103] + [1000 + ord(c) for c in text]])
        self.assertFalse(bool(stop(ids(partial), None)[0]))
        self.assertTrue(bool(stop(ids(partial + "\n"), None)[0]))
        self.assertEqual(stop.reason, "line_boundary")

    def test_boundary_at_token_ceiling_is_not_an_incomplete_token_limit(self):
        continuation = "A wave\nA shell\nA shore\n"
        tokenizer, model = self.fake_pair(continuation, invoke_stopper=True, eos=False)
        result = self.module.generate_poem(tokenizer, model, first_line="First", line_count=4,
                                          max_new_tokens=len(continuation), return_metadata=True)
        self.assertFalse(result["token_limit_reached"])
        self.assertEqual(result["stop_reason"], "line_boundary")

    def test_first_line_echo_is_preserved_and_rejected(self):
        continuation = "First\nA wave\nA shell\nA shore"
        tokenizer, model = self.fake_pair(continuation)
        result = self.module.generate_poem(tokenizer, model, first_line="First", line_count=4,
                                          return_metadata=True)
        self.assertEqual(result["content"].splitlines().count("First"), 2)
        self.assertEqual(check_poem(result["content"], 4, [], "First")["actual_lines"], 5)

    def test_seed_is_reproducible_without_changing_global_random_state(self):
        tokenizer, model = self.fake_pair("A wave.\nA shell.\nA shore.")
        draws = []
        original = model.generate.side_effect
        def simulate(**kwargs):
            draws.append(self.torch.rand(4))
            return original(**kwargs)
        model.generate.side_effect = simulate
        before = self.torch.get_rng_state().clone()
        for _ in range(2):
            self.module.generate_poem(tokenizer, model, first_line="First", line_count=4, seed=42)
        self.assertTrue(self.torch.equal(draws[0], draws[1]))
        self.assertTrue(self.torch.equal(before, self.torch.get_rng_state()))


class LibraryTests(unittest.TestCase):
    def test_append_deduplicate_and_preserve(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "poems.json"
            first = {"id": "one", "content": "An existing poem"}
            second = {"id": "two", "content": "A new poem"}
            self.assertEqual(save_poems([first], path), 1)
            self.assertEqual(save_poems([first, second], path), 1)
            self.assertEqual(load_poems(path), [first, second])
            path.write_text("invalid JSON", encoding="utf-8")
            with self.assertRaises(ValueError):
                save_poems([second], path)
            self.assertEqual(path.read_text(), "invalid JSON")

    def test_import_preserves_content_and_missing_dates(self):
        rows = import_records("old.csv", b'title,theme,content,date\nSea,Ocean,"Line one\nLine two",2025-01-01\n')
        self.assertEqual(rows[0]["content"], "Line one\nLine two")
        self.assertEqual(rows[0]["created_at"], "2025-01-01")
        md = import_records("old.md", b"# Sea\n\nA wave.\nAnother wave.")
        self.assertEqual(md[0]["title"], "Sea")
        self.assertIsNone(md[0]["created_at"])
        self.assertEqual(md, import_records("old.md", b"# Sea\n\nA wave.\nAnother wave."))


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()

    def test_generation_and_save_with_fake_model(self):
        from streamlit.testing.v1 import AppTest
        fake = types.ModuleType("generer")
        fake.load_model = Mock(return_value=(object(), types.SimpleNamespace(peft_config={"v1": {}}, set_adapter=Mock(), load_adapter=Mock(), eval=Mock())))
        fake.generate_poem = Mock(return_value="A quiet sea\nThe golden waves return.\nA gull circles overhead.\n"
                                               "Your footsteps warm the sand.\nThe wind carries a song.\n"
                                               "We follow silver trails.\nAn open shell reflects dawn.\nOur journey finds its home.")
        with patch.dict("sys.modules", {"generer": fake}), patch("bibliotheque.save_poems", return_value=1) as save:
            at = AppTest.from_file("app.py", default_timeout=15).run(timeout=10)
            at.button[0].click().run()
            at.text_input[0].set_value("the sea at sunset")
            at.slider[0].set_value(1.0)
            at.button[0].click().run()
            self.assertFalse(at.exception)
            settings = fake.generate_poem.call_args.kwargs
            self.assertEqual(settings["temperature"], 1.0)
            self.assertEqual(settings["theme"], "the sea at sunset")
            next(b for b in at.button if b.label == "Enregistrer le poème").click().run()
            self.assertFalse(at.exception)
            saved = save.call_args.args[0][0]
            self.assertEqual(saved["parameters"]["temperature"], 1.0)
            self.assertTrue(saved["created_at"])
            self.assertTrue(at.success)

    def test_home_and_generation_without_loading_model(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=15).run(timeout=10)
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "AI Poetry Studio")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual([s.value for s in at.slider], [0.8, 0.9, 1.15, 3, 200])
        self.assertTrue(next(c for c in at.checkbox if "tentatives supplémentaires" in c.label).value)

    def test_repetitive_generation_is_displayed_as_failure_and_saved_only_as_draft(self):
        from streamlit.testing.v1 import AppTest
        fake = types.ModuleType("generer")
        fake.load_model = Mock(return_value=(object(), types.SimpleNamespace(peft_config={"v1": {}}, set_adapter=Mock(), load_adapter=Mock(), eval=Mock())))
        repeated = "\n".join(["The stars shine upon your eyes"] * 8)
        fake.generate_poem = Mock(return_value=repeated)
        with patch.dict("sys.modules", {"generer": fake}), patch("bibliotheque.save_poems", return_value=1) as save:
            at = AppTest.from_file("app.py", default_timeout=15).run(timeout=10)
            at.button[0].click().run()
            at.button[0].click().run(timeout=10)
            self.assertFalse(at.exception)
            self.assertEqual(fake.generate_poem.call_count, 3)
            self.assertTrue(any("non conforme" in e.value for e in at.error))
            self.assertTrue(any("répétés" in w.value for w in at.warning))
            self.assertFalse(at.success)
            self.assertTrue(at.get("json"))
            draft = at.session_state.draft
            self.assertEqual(draft["content"], repeated)
            self.assertFalse(draft["validation"]["conformant"])
            self.assertEqual(draft["validation"]["effective_parameters"]["no_repeat_ngram_size"], 3)
            next(b for b in at.button if b.label == "Enregistrer le brouillon").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(save.call_args.args[0][0]["content"], repeated)
            self.assertFalse(at.success)

    def test_legacy_validation_is_rechecked_without_modifying_old_poem(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.navigation.page import calc_hash
        legacy = {"id": "old", "content": "Moon\nMoon\nMoon\nMoon", "title": "Old poem",
                  "validation": {"conformant": True, "requested_lines": 4, "actual_lines": 4,
                                 "attempts": 1, "present_words": [], "missing_words": [],
                                 "first_line_ok": True, "token_limit_reached": False}}
        with patch("bibliotheque.load_poems", return_value=[legacy]), patch("bibliotheque.save_poems") as save:
            at = AppTest.from_file("app.py", default_timeout=15).run(timeout=10)
            at._page_hash = calc_hash("bibliotheque")
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertFalse(at.success)
            self.assertTrue(legacy["validation"]["conformant"])
            save.assert_not_called()

    def test_library_search_and_empty_state(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.navigation.page import calc_hash
        poems = [{"id": "one", "title": "Sea", "theme": "Ocean", "content": "A wave.", "created_at": None}]
        with patch("bibliotheque.load_poems", return_value=poems):
            at = AppTest.from_file("app.py", default_timeout=15).run(timeout=10)
            # AppTest identifie les pages fonctionnelles par le hash de leur URL.
            at._page_hash = calc_hash("bibliotheque")
            at.run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.selectbox), 1)
            at.text_input[0].set_value("missing").run()
            self.assertFalse(at.exception)
            self.assertTrue(at.info)

        with patch("bibliotheque.load_poems", return_value=[]):
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(at.info)

    def test_temperature_comparison_runs_three_simulated_calls_and_keeps_review_subjective(self):
        from streamlit.testing.v1 import AppTest
        fake = types.ModuleType("generer")
        fake.load_model = Mock(return_value=(object(), types.SimpleNamespace(peft_config={"v1": {}}, set_adapter=Mock(), load_adapter=Mock(), eval=Mock())))
        fake.generate_poem = Mock(return_value="A quiet sea.\nThe golden waves return.\nA gull circles overhead.\n"
                                               "Your footsteps warm the sand.\nThe wind carries a song.\n"
                                               "We follow silver trails.\nAn open shell reflects dawn.\nOur journey finds its home.")
        with patch.dict("sys.modules", {"generer": fake}):
            at = AppTest.from_file("app.py", default_timeout=15).run()
            at.button[0].click().run()
            next(c for c in at.checkbox if c.label.startswith("Comparer les températures")).check()
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertEqual(fake.generate_poem.call_count, 3)
            self.assertEqual([c.kwargs["temperature"] for c in fake.generate_poem.call_args_list], [.7, .75, .8])
            report = at.session_state.draft["validation"]
            self.assertTrue(report["comparison_mode"])
            self.assertEqual(report["experiment"]["literary_improvement"], "not_established")
            self.assertIsNone(report["literary_review"]["human_score"])
            self.assertTrue(any("supériorité littéraire" in item.value for item in at.info))

    def test_suspended_ending_is_visible_without_a_green_success_message(self):
        from streamlit.testing.v1 import AppTest
        fake = types.ModuleType('generer')
        fake.load_model = Mock(return_value=(object(), types.SimpleNamespace(peft_config={"v1": {}}, set_adapter=Mock(), load_adapter=Mock(), eval=Mock())))
        fake.generate_poem = Mock(return_value='A quiet sea.\nGolden waves return.\nA gull circles overhead.\n'
                                  'Your steps warm the sand.\nThe wind carries a song.\nWe follow silver trails.\n'
                                  'An open shell reflects dawn.\nOur journey waits and')
        with patch.dict('sys.modules', {'generer': fake}):
            at = AppTest.from_file('app.py', default_timeout=15).run()
            at.button[0].click().run()
            next(s for s in at.selectbox if s.label == 'Sélection des candidats').set_value('Économe')
            at.button[0].click().run()
            self.assertFalse(at.exception)
            self.assertFalse(at.success)
            self.assertTrue(any('fin paraît suspendue' in w.value for w in at.warning))
            self.assertTrue(at.session_state.draft['validation']['conformant'])
            self.assertEqual(fake.generate_poem.call_count, 3)


if __name__ == "__main__":
    unittest.main()
