"""Pilote réel sur dix demandes fixes, distinct de la bibliothèque et de l'entraînement."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

PARAMETERS = dict(line_count=8, temperature=.8, top_p=.9, repetition_penalty=1.15,
                  no_repeat_ngram_size=3, max_new_tokens=200, seed=42)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime(source):
    poetry = module_from(source / 'poesie.py', 'evaluated_poetry')
    original = sys.modules.get('poesie')
    sys.modules['poesie'] = poetry
    try:
        generator = module_from(source / 'generer.py', 'evaluated_generator')
    finally:
        if original is None:
            sys.modules.pop('poesie', None)
        else:
            sys.modules['poesie'] = original
    return generator, poetry


def load_cases(path):
    cases = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if len(cases) != 10 or len({case['id'] for case in cases}) != 10:
        raise ValueError('Dix thèmes distincts requis')
    for case in cases:
        if not case['first_line'] or '\n' in case['first_line'] or len(case['required_words']) != 3:
            raise ValueError('Première ligne et trois mots fixes requis')
    return cases


def run_phase(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Pilote CPU identique pour les deux versions.
    project = args.project.resolve()
    source = args.snapshot.resolve() if args.phase == 'before' else project
    cases = load_cases(args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    protected = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                 'tokenizer_config.json', 'chat_template.jinja']
    manifest = dict(cases=cases, parameters=PARAMETERS, threads=args.threads,
                    weights_and_tokenizer={name: digest(project / name) for name in protected})
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Configuration ou fichiers différents du pilote initial')
    else:
        if args.phase != 'before':
            raise ValueError('Exécuter la référence avant la modification')
        save_json(manifest_path, manifest)
    path = args.output / (args.phase + '.json')
    signatures = {name: digest(source / name) for name in ['generer.py', 'poesie.py']}
    state = dict(phase=args.phase, source_sha256=signatures, records=[], status='running')
    if path.exists():
        if not args.resume:
            raise ValueError('Résultats existants : utiliser --resume ou un autre dossier')
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['source_sha256'] != signatures:
            raise ValueError('Le code de cette phase a changé pendant le pilote')
    if len(state['records']) == len(cases):
        print(args.phase, 'déjà terminé', flush=True)
        return
    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(project))
    import generer as live_generator
    generator, poetry = runtime(source)
    print('Chargement Qwen + LoRA existants, CPU, aucun entraînement', flush=True)
    started = time.perf_counter()
    tokenizer, model = live_generator.load_model()
    state.update(model_load_seconds=time.perf_counter() - started,
                 started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                 model_name=model.config._name_or_path, dtype=str(next(model.parameters()).dtype),
                 resolved_generation_config=model.generation_config.to_dict())
    save_json(path, state)
    for case in cases[len(state['records']):]:
        settings = {key: case[key] for key in ['theme', 'tone', 'first_line', 'required_words']}
        settings.update(PARAMETERS)
        messages = poetry.build_messages(case['theme'], case['first_line'], 8, case['required_words'], tone=case['tone'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + case['first_line'] + '\n'
        started = time.perf_counter()
        result = generator.generate_poem(tokenizer, model, **settings, return_metadata=True)
        elapsed = time.perf_counter() - started
        native = poetry.check_poem(result['content'], 8, case['required_words'], case['first_line'], result['token_limit_reached'])
        state['records'].append(dict(case_id=case['id'], settings=settings, result=result,
                                      elapsed_seconds=elapsed, prompt_tokens=len(tokenizer.encode(prompt, add_special_tokens=False)),
                                      native_validation=native))
        save_json(path, state)
        print(f"{args.phase} {len(state['records'])}/10 {case['id']} {elapsed:.2f}s "
              f"vers={native['actual_lines']} conforme={native['conformant']} arrêt={result['stop_reason']}", flush=True)
    state.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(path, state)


def build_report(project, output):
    """Un même évaluateur courant relit les deux corpus ; pas de notes humaines inventées."""
    poetry = module_from(project / 'poesie.py', 'common_evaluator')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    rows, summaries, texts = [], [], []
    for phase in ['before', 'after']:
        state = json.loads((output / (phase + '.json')).read_text(encoding='utf-8'))
        if state['status'] != 'complete' or len(state['records']) != 10:
            raise ValueError('Les deux phases complètes sont nécessaires')
        total_seconds = sum(row['elapsed_seconds'] for row in state['records'])
        phase_rows = []
        for entry, case in zip(state['records'], manifest['cases']):
            if entry['case_id'] != case['id']:
                raise ValueError('Ordre ou thèmes différents')
            result = entry['result']
            formal = poetry.check_poem(result['content'], 8, case['required_words'], case['first_line'], result['token_limit_reached'])
            review = poetry.literary_review(result['content'], case['tone'], case['first_line'])
            codes = [item['code'] for item in review['warnings']]
            row = dict(phase=phase, case_id=case['id'], conformant=formal['conformant'],
                       lines_ok=formal['line_count_ok'], first_line_ok=formal['first_line_ok'],
                       words_ok=not formal['missing_words'], repetitions_ok=formal['repetitions_ok'],
                       token_limit_reached=result['token_limit_reached'],
                       ending_suspected=review.get('ending_warning_count', 'suspended_ending' in codes) > 0,
                       recurring_formulations=bool(review.get('recurring_phrases', [])),
                       elapsed_seconds=entry['elapsed_seconds'], prompt_tokens=entry['prompt_tokens'],
                       warning_codes=codes, content=result['content'])
            rows.append(row); phase_rows.append(row)
            texts.append(f"## {phase} — {case['id']}\n\nTon : {case['tone']}. Mots : {', '.join(case['required_words'])}.\n\n```text\n{result['content']}\n```\n")
        summary = dict(phase=phase, poems=10, generation_calls=10, generation_seconds=total_seconds,
                       mean_prompt_tokens=sum(r['prompt_tokens'] for r in phase_rows) / 10)
        for metric in ['conformant', 'lines_ok', 'first_line_ok', 'words_ok', 'repetitions_ok', 'token_limit_reached', 'ending_suspected', 'recurring_formulations']:
            summary[metric] = sum(r[metric] for r in phase_rows)
        summaries.append(summary)
    save_json(output / 'common_evaluation.json', dict(evaluator_sha256=digest(project / 'poesie.py'),
              summaries=summaries, records=rows, human_evaluation='not_performed'))
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, 'warning_codes': json.dumps(row['warning_codes'])} for row in rows)
    text = '# Pilote réel avant/après : dix thèmes fixes\n\n'
    text += 'Un seul candidat par thème et version, seed 42, huit vers, température 0.8, top_p 0.9, pénalité 1.15, n-grammes 3, budget 200. Poids et tokenizer identiques. Les deux corpus sont relus avec le même évaluateur après correction.\n\n'
    text += '| Version | Conforme | 8 vers | Première ligne | Tous les mots | Sans répétitions détectées | Budget atteint | Fin suspecte (heuristique) | Formulations récurrentes (heuristique) | Génération (s) | Tokens prompt moyens |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for s in summaries:
        text += f"| {s['phase']} | {s['conformant']}/10 | {s['lines_ok']}/10 | {s['first_line_ok']}/10 | {s['words_ok']}/10 | {s['repetitions_ok']}/10 | {s['token_limit_reached']}/10 | {s['ending_suspected']}/10 | {s['recurring_formulations']}/10 | {s['generation_seconds']:.3f} | {s['mean_prompt_tokens']:.1f} |\n"
    text += '\nDurées mesurées hors chargement du modèle ; la charge CPU peut varier entre les phases. Les alertes de fin sont indicatives. Dix thèmes et une seed ne suffisent pas à généraliser. Ce pilote à un candidat ne mesure pas le gain du classement sur trois candidats.\n\nAucune note de créativité ou de qualité littéraire n’est calculée. La supériorité littéraire reste à évaluer par lecture humaine.\n'
    (output / 'results.md').write_text(text, encoding='utf-8')
    (output / 'samples.md').write_text('\n'.join(texts), encoding='utf-8')
    shuffled = rows[:]; random.Random(2026).shuffle(shuffled)
    with (output / 'human_eval_blind.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['candidate_id', 'theme', 'requested_tone', 'first_line', 'required_words', 'content'] + [key for key, _, _ in poetry.HUMAN_RUBRIC] + ['comment']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        mapping = []
        cases = {case['id']: case for case in manifest['cases']}
        for i, row in enumerate(shuffled, 1):
            case = cases[row['case_id']]; identifier = f'P{i:02}'
            writer.writerow(dict(candidate_id=identifier, theme=case['theme'], requested_tone=case['tone'],
                                 first_line=case['first_line'], required_words=', '.join(case['required_words']), content=row['content']))
            mapping.append(dict(candidate_id=identifier, phase=row['phase'], case_id=row['case_id']))
    save_json(output / 'human_mapping_private.json', mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['before', 'after', 'report'], required=True)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('evaluation_cases.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or (args.phase == 'before' and args.snapshot is None):
        parser.error('threads >= 1 et snapshot requis pour la référence')
    if args.phase == 'report':
        build_report(args.project.resolve(), args.output)
    else:
        run_phase(args)


if __name__ == '__main__':
    main()
