"""Construction des consignes et contrôles, sans charger le modèle."""
import json
import re
import unicodedata
import csv
import io
import random
import time

DEFAULT_TEMPERATURE = .8  # 0.75 n'a pas encore été comparé sur de vrais poèmes.
EXPERIMENT_TEMPERATURES = (.70, .75, .80)
HUMAN_RUBRIC = (
    ("coherence", "Cohérence et progression", "La scène, les images et la voix forment-elles un ensemble cohérent ?"),
    ("fluency", "Naturel et rythme", "L'anglais est-il naturel, fluide et agréable à lire à voix haute ?"),
    ("imagery", "Imagerie et originalité", "Les images sont-elles précises, expressives et peu convenues ?"),
    ("tone", "Justesse du ton", "L'émotion correspond-elle au ton et à la demande sans paraître forcée ?"),
    ("ending", "Force de la conclusion", "La fin apporte-t-elle une résolution sensible plutôt qu'une simple répétition ?"),
)

DEFAULT_REPETITION_PENALTY = 1.15
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
SELECTION_CRITERIA = (
    "conformité complète, absence de répétitions, moins de répétitions, première ligne, "
    "écart au nombre de vers, mots manquants, absence d'interruption, vers trop longs, "
    "moins de fins suspectes, moins de formulations récurrentes, puis moins d'autres "
    "indices stylistiques à relire (heuristiques, pas une note humaine) ; "
    "à égalité, première tentative"
)

# Chaque ton influence l'écriture ; « Libre » ne force aucune émotion.
TONES = {
    "Romantique": "Romantic: convey tenderness, affection, intimacy and warm connection through shared gestures, attentive details and mutual companionship. Show affection rather than stacking abstract declarations. Avoid defaulting to heartbreak, grief or loss unless explicitly requested by the theme or opening line. End in a believable sense of closeness, not a generic promise.",
    "Joyeux": "Joyful: convey delight, playfulness and lively, luminous images through a concrete scene and buoyant phrasing. Avoid forced exclamations and sugary generalities; let joy emerge from a small discovery.",
    "Mélancolique": "Melancholic: convey gentle longing and reflective sadness without melodrama. Ground the feeling in an ordinary detail; allow a quiet shift of understanding rather than an exaggerated tragedy.",
    "Inspirant": "Inspirational: convey hope, courage, renewal and believable possibility through a concrete action or change. Avoid slogans and guaranteed triumph; make the final possibility feel earned.",
    "Mystérieux": "Mysterious: evoke wonder, subtle uncertainty and suggestive imagery within a consistent scene. Leave an intriguing resonance, not random obscurity or unrelated symbols.",
    "Paisible": "Peaceful: use serene imagery, a calm rhythm and a sense of quiet balance. Observe small sensory details and let the ending settle naturally; avoid empty statements that everything is perfect.",
    "Libre": "Choose the emotional tone freely, consistent with the supplied theme and opening line. Keep that voice coherent while allowing a subtle emotional development; do not impose a romantic or tragic plot.",
}


def token_budget(requested, line_count, automatic=True):
    """Budget borné à 600 tokens par appel, adapté aux poèmes longs sur CPU."""
    requested = max(50, min(int(requested), 600))
    recommended = {4: 120, 8: 200, 12: 280, 16: 360}[line_count]
    return max(requested, recommended) if automatic else requested


def parse_words(text):
    """Accepte huit mots au plus, avec dédoublonnage insensible à la casse."""
    words = []
    for value in text.split(","):
        word = value.strip()
        if not word:
            continue
        if len(word) > 40 or not re.fullmatch(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", word):
            raise ValueError("Indiquez des mots seuls, séparés par des virgules (40 caractères maximum par mot).")
        if word.casefold() not in [w.casefold() for w in words]:
            words.append(word)
    if len(words) > 8:
        raise ValueError("Vous pouvez imposer au maximum 8 mots distincts.")
    return words


def build_messages(theme="", first_line="", line_count=8, required_words=None, correction="", tone="Libre"):
    """Ne donne aucun thème précis si le champ correspondant est vide."""
    if line_count not in (4, 8, 12, 16):
        raise ValueError("Choisissez 4, 8, 12 ou 16 vers.")
    if "\n" in first_line or "\r" in first_line:
        raise ValueError("La première ligne doit tenir sur un seul vers.")
    if tone not in TONES:
        raise ValueError("Choisissez un ton proposé dans la liste.")
    instructions = [f"Write an English lyric poem of exactly {line_count} non-empty lines, one verse per line. No title, numbering or commentary.",
                    "Deepen one central image within a consistent scene and voice. Show emotion through sensory details, not a narrated sequence or an explanation.",
                    "Use idiomatic English and natural word order. Vary rhythm: no minimum word count per line, at most 16 words except the fixed opening. Avoid filler and forced rhyme.",
                    "Do not repeat lines, refrains or consecutive words and phrases. End with an expressive turn of the central image and a complete final thought, not a generic moral.",
                    "Emotional tone: " + TONES[tone]]
    if line_count == 8:
        instructions.append(
            "Progression: lines 1-2 scene; lines 3-4 emotion; lines 5-6 deepen the image; "
            "lines 7-8 resolution. The fixed opening counts as line 1. Do not print this plan.")
    else:
        instructions.append(f"Across the {line_count} lines, move from a concrete opening through an emotional "
                            "development to a meaningful final image; scale the movement to this length, without headings.")
    if theme.strip():
        instructions.append("Theme: " + json.dumps(theme, ensure_ascii=False)
                            + ". Keep the imagery focused on it.")
    elif not first_line.strip():
        instructions.append("Choose your own subject freely.")
    if first_line.strip():
        instructions.append("The first line is already provided in the assistant response. "
                            "Continue from it, without repeating it. It counts as line 1: "
                            + json.dumps(first_line, ensure_ascii=False))
        instructions.append(f"Generate exactly {line_count - 1} NEW non-empty lines, "
                            "then end your response. Do not restart the poem.")
    if required_words:
        instructions.append("Integrate these exact whole words naturally, in any letter case; do not list them: "
                            + json.dumps(required_words, ensure_ascii=False))
    if correction:
        instructions.append("Feedback from a previous candidate follows. Create a fresh candidate using "
                            "the same opening instructions above; return only the poem. " + correction)
    return [{"role": "system", "content": "Compose coherent, natural English lyric poetry. Respect the form, tone and fixed opening."},
            {"role": "user", "content": "\n".join(instructions)}]


def complete_poem(continuation, first_line=""):
    """Assemble le préremplissage et TOUTE la suite, sans dédoublonner ni couper."""
    content = continuation.strip()
    if not first_line.strip():
        return content
    return first_line + ("\n" + content if content else "")


def word_tokens(text):
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized)


def repetition_report(lines):
    """Heuristiques explicites : vers, blocs non chevauchants et suites de 1 à 4 mots."""
    normalized = [tuple(word_tokens(line)) for line in lines]
    positions = {}
    for i, words in enumerate(normalized, 1):
        if words:
            positions.setdefault(words, []).append(i)
    repeated_lines = [{"lines": indices, "text": lines[indices[0] - 1]}
                      for indices in positions.values() if len(indices) > 1]
    blocks = []
    for size in range(2, len(lines) // 2 + 1):
        windows = {}
        for start in range(len(lines) - size + 1):
            window = tuple(normalized[start:start + size])
            if not all(window):
                continue
            starts = windows.setdefault(window, [])
            if not starts or start >= starts[-1] + size:
                starts.append(start)
        for starts in windows.values():
            if len(starts) > 1:
                blocks.append({"starts": [i + 1 for i in starts], "length": size})

    flat = [(word, i) for i, words in enumerate(normalized, 1) for word in words]
    words = [word for word, _ in flat]
    runs, covered = [], set()
    for size in range(1, min(4, len(words) // 2) + 1):
        start = 0
        while start + 2 * size <= len(words):
            phrase = words[start:start + size]
            end = start + size
            while words[end:end + size] == phrase:
                end += size
            span = set(range(start, end))
            if end >= start + 2 * size and not span.intersection(covered):
                runs.append({"phrase": " ".join(phrase), "occurrences": (end - start) // size,
                             "lines": sorted({line for _, line in flat[start:end]})})
                covered.update(span)
                start = end
            else:
                start += 1
    # Les blocs réutilisent des vers déjà comptés ; éviter un double comptage du score.
    count = sum(len(item["lines"]) - 1 for item in repeated_lines)
    count += sum(item["occurrences"] - 1 for item in runs)
    return {"repeated_lines": repeated_lines, "repeated_blocks": blocks, "word_runs": runs,
            "count": count, "ok": not (repeated_lines or blocks or runs)}


def check_poem(content, line_count, required_words, first_line="", token_limit_reached=False):
    """Compte les lignes non vides ; les retours visuels du navigateur ne comptent pas."""
    lines = [line for line in content.splitlines() if line.strip()]
    present, missing = [], []
    for word in required_words or []:
        # Les lettres accentuées, les apostrophes et traits d'union restent dans le mot.
        normalized_word = word.casefold().replace("’", "'")
        pattern = r"(?<![\w'’\-])" + re.escape(normalized_word) + r"(?![\w'’\-])"
        (present if re.search(pattern, content.casefold().replace("’", "'")) else missing).append(word)
    first_ok = not first_line.strip() or (
        bool(content.splitlines()) and content.splitlines()[0] == first_line
        and sum(word_tokens(line) == word_tokens(first_line) for line in lines) == 1
    )
    # Heuristique de concision, pas une mesure de qualité littéraire.
    long_lines = [i for i, line in enumerate(lines, 1)
                  if len(line.split()) > 16 and not (i == 1 and first_line.strip())]
    repetitions = repetition_report(lines)
    anomalies = []
    if not content.strip():
        anomalies.append("Sortie vide.")
    if len(lines) != line_count:
        anomalies.append(f"Nombre de vers incorrect : {len(lines)} au lieu de {line_count}.")
    if not first_ok:
        anomalies.append("Première ligne non respectée exactement ou répétée.")
    if missing:
        anomalies.append("Mots obligatoires manquants : " + ", ".join(missing) + ".")
    if repetitions["repeated_lines"]:
        anomalies.append("Vers répétés : " + "; ".join(
            ", ".join(map(str, item["lines"])) for item in repetitions["repeated_lines"]) + ".")
    if repetitions["repeated_blocks"]:
        anomalies.append("Groupes de vers répétés : " + "; ".join(
            f"{item['length']} vers aux positions {item['starts']}"
            for item in repetitions["repeated_blocks"]) + ".")
    if repetitions["word_runs"]:
        anomalies.append("Mots ou expressions consécutifs répétés : " + "; ".join(
            f"{item['phrase']} ({item['occurrences']} fois, vers {item['lines']})"
            for item in repetitions["word_runs"]) + ".")
    if token_limit_reached:
        anomalies.append("Limite de tokens atteinte : fin potentiellement incomplète.")
    if long_lines:
        anomalies.append("Vers dépassant 16 mots : " + ", ".join(map(str, long_lines)) + ".")
    return {"requested_lines": line_count, "actual_lines": len(lines),
            "line_count_ok": len(lines) == line_count, "present_words": present,
            "missing_words": missing, "first_line_ok": first_ok,
            "token_limit_reached": token_limit_reached,
            "long_lines": long_lines,
            "repetitions": repetitions, "repetitions_ok": repetitions["ok"],
            "anomalies": anomalies, "validation_version": 2,
            "conformant": not anomalies}


def recurring_phrases(lines):
    """Expressions de 3–5 mots dans >=3 vers : signal de relecture, pas faute certaine."""
    tokens = [word_tokens(line) for line in lines]
    stopwords = set("a an the in on of to and or is are i you we it that my your our with as for by at".split())
    found = []
    for size in (5, 4, 3):
        positions = {}
        for index, words in enumerate(tokens, 1):
            for start in range(len(words) - size + 1):
                phrase = tuple(words[start:start + size])
                if sum(word not in stopwords for word in phrase) >= 2:
                    positions.setdefault(phrase, set()).add(index)
        for phrase, indices in positions.items():
            if len(indices) < 3:
                continue
            if any(set(indices).issubset(item['lines']) and
                   any(tuple(item['words'][i:i + size]) == phrase for i in range(len(item['words']) - size + 1))
                   for item in found):
                continue
            found.append(dict(words=list(phrase), phrase=' '.join(phrase), lines=sorted(indices)))
    return [{key: value for key, value in item.items() if key != 'words'} for item in found]


def literary_review(content, tone="Libre", first_line=""):
    """Signaux de relecture, sans modèle sémantique ni note littéraire automatique."""
    lines = [line for line in content.splitlines() if line.strip()]
    warnings = []
    sentences = [part for part in re.split(r"[.!?]+", content) if word_tokens(part)]
    sentence_lengths = [len(word_tokens(part)) for part in sentences]
    if any(length > 30 for length in sentence_lengths):
        warnings.append({"code": "long_sentence", "message":
                         "Plus de 30 mots sans ponctuation forte : phrase potentiellement longue ou ponctuation libre, à relire."})
    if lines and word_tokens(lines[-1]):
        if re.search(r"[,;:\-–—]\s*$", lines[-1]):
            warnings.append({"code": "dangling_punctuation", "message":
                             "La dernière ligne finit par une ponctuation de continuation : fin probablement interrompue, à relire."})
        if word_tokens(lines[-1])[-1] in {"and", "or", "but", "because", "with", "of", "to", "the", "a", "an", "that", "which", "if"}:
            warnings.append({"code": "suspended_ending", "message":
                             "Dernier mot de liaison ou déterminant : la conclusion peut rester suspendue ; une intention poétique reste possible."})
    if content.count('"') % 2 or any(content.count(a) > content.count(b) for a, b in [('(', ')'), ('[', ']'), ('“', '”'), ('«', '»')]):
        warnings.append({"code": "unclosed_delimiter", "message":
                         "Guillemet ou parenthèse ouvrante non refermée : vérifier la fin ; ce signal n'est pas une preuve sémantique."})
    phrases = recurring_phrases(lines)
    if phrases:
        warnings.append({"code": "recurring_phrase", "message":
                         "Expressions récurrentes dans au moins trois vers : " + '; '.join(
                             f"{item['phrase']} (vers {item['lines']})" for item in phrases) + ". Une anaphore intentionnelle reste possible."})
    openings = {}
    for i, line in enumerate(lines, 1):
        if i == 1 and first_line.strip():
            continue
        words = tuple(word_tokens(line)[:2])
        if len(words) == 2:
            openings.setdefault(words, []).append(i)
    for words, positions in openings.items():
        if len(positions) >= 3:
            warnings.append({"code": "similar_openings", "lines": positions, "message":
                             f"Même début de deux mots ({' '.join(words)}) dans plusieurs vers : vérifier si l'anaphore est intentionnelle."})
    normalized = " ".join(word_tokens(content))
    for phrase in ("my heart skips a beat", "love is like a rose", "you are my everything", "until the end of time"):
        if f" {phrase} " in f" {normalized} ":
            warnings.append({"code": "familiar_expression", "message":
                             f"Expression courante ({phrase}) : vérifier si son emploi apporte une image personnelle."})
    ending_flags = sum(item['code'] in {'suspended_ending', 'unclosed_delimiter', 'dangling_punctuation'} for item in warnings)
    return {"method": "heuristic_reading_signals", "warnings": warnings,
            "ending_warning_count": ending_flags, "ending_status": "suspected_incomplete" if ending_flags else "no_signal",
            "recurring_phrases": phrases,
            "warning_count": len(warnings), "line_word_counts": [len(word_tokens(line)) for line in lines],
            "sentence_word_counts": sentence_lengths, "requested_tone": tone,
            "coherence_status": "requires_human_review", "human_quality_status": "not_evaluated",
            "human_score": None, "human_rubric": [
                {"key": key, "criterion": label, "question": question, "range": [0, 2]}
                for key, label, question in HUMAN_RUBRIC],
            "limits": "Ces indices ne certifient ni la grammaire, ni la cohérence sémantique, ni l'originalité, ni le ton. Aucun objectif de 8 à 9/10 n'est garanti."}


def selection_score(report, review=None):
    """Ordre lexicographique, sans prétendre mesurer la qualité littéraire."""
    review = review or {}
    endings = review.get('ending_warning_count', sum(w['code'] in {'suspended_ending', 'unclosed_delimiter'} for w in review.get('warnings', [])))
    recurring = len(review.get('recurring_phrases', []))
    return (report["conformant"], report["repetitions_ok"], -report["repetitions"]["count"],
            report["first_line_ok"], -abs(report["actual_lines"] - report["requested_lines"]),
            -len(report["missing_words"]), not report["token_limit_reached"], -len(report["long_lines"]),
            -endings, -recurring, -review.get("warning_count", 0))


def human_comparison_csv(report):
    """Grille aveugle vierge ; aucune note ou préférence n'est inventée."""
    rows = list(report["attempt_history"])
    random.Random(report.get("comparison_seed", 42)).shuffle(rows)
    buffer = io.StringIO(newline="")
    fields = ["candidate", "content", "conformant"] + [item[0] for item in HUMAN_RUBRIC] + ["comment"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for i, row in enumerate(rows):
        writer.writerow({"candidate": chr(ord('A') + i), "content": row["content"],
                         "conformant": row["validation"]["conformant"]})
    return buffer.getvalue()


def generate_checked(generate, tokenizer, model, settings):
    """Un appel, ou trois au maximum ; aucun vers n'est coupé pour ajuster le compte."""
    arguments = dict(settings)
    retry = arguments.pop("allow_retries", True)
    automatic = arguments.pop("auto_budget", True)
    comparison = arguments.pop("comparison_mode", False)
    controlled = arguments.pop("controlled_candidates", False)
    comparison_seed = arguments.pop("comparison_seed", 42)
    selection_mode = arguments.pop('selection_mode', 'complete')
    if selection_mode not in ('complete', 'economy'):
        raise ValueError('Mode de sélection inconnu')
    if comparison and not retry:
        raise ValueError("La comparaison nécessite l'autorisation des trois tentatives.")
    if not isinstance(comparison_seed, int) or not 0 <= comparison_seed < 2 ** 31:
        raise ValueError("La seed de comparaison doit être un entier entre 0 et 2147483647.")
    initial_seed = arguments.pop("seed", None)
    initial_correction = arguments.pop("correction", "")
    arguments.setdefault("repetition_penalty", DEFAULT_REPETITION_PENALTY)
    arguments.setdefault("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE)
    initial_penalty = float(arguments["repetition_penalty"])
    if not 1.0 <= initial_penalty <= 1.3:
        raise ValueError("La pénalité de répétition doit être comprise entre 1.0 et 1.3.")
    requested = arguments["max_new_tokens"]
    arguments["max_new_tokens"] = token_budget(requested, arguments["line_count"], automatic)
    best = None
    feedback = initial_correction
    budgets = []
    history = []
    for attempt in range(1, (3 if retry else 1) + 1):
        if comparison:
            arguments.update(temperature=EXPERIMENT_TEMPERATURES[attempt - 1], top_p=.9,
                             repetition_penalty=1.15, no_repeat_ngram_size=3, seed=comparison_seed)
        elif initial_seed is not None:
            arguments["seed"] = initial_seed + attempt - 1
        budgets.append(arguments["max_new_tokens"])
        parameters = {key: arguments.get(key, default) for key, default in (
            ("temperature", DEFAULT_TEMPERATURE), ("top_p", .9), ("repetition_penalty", DEFAULT_REPETITION_PENALTY),
            ("no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE), ("max_new_tokens", 200), ("seed", None))}
        parameters["do_sample"] = True
        started = time.perf_counter()
        result = generate(tokenizer, model, **arguments, correction=feedback, return_metadata=True)
        elapsed = time.perf_counter() - started
        if isinstance(result, str):  # Compatibilité avec les générateurs simples / simulés.
            result = {"content": result, "token_limit_reached": False}
        content = result["content"]
        report = check_poem(content, settings["line_count"], settings["required_words"],
                            settings["first_line"], result.get("token_limit_reached", False))
        review = literary_review(content, settings.get("tone", "Libre"), settings["first_line"])
        score = selection_score(report, review)
        history.append({"attempt": attempt, "parameters": parameters, "validation": report,
                        "content": content, "literary_review": review, "elapsed_seconds": elapsed,
                        "score": list(score), "stop_reason": result.get("stop_reason", "unknown")})
        if best is None or score > best[0]:
            best = (score, content, report, arguments["max_new_tokens"], attempt)
        if comparison or controlled:
            continue  # Même prompt, seed et budget : pas de correction ni d'adaptation cachée.
        if selection_mode == 'economy' and report['conformant'] and review['warning_count'] == 0:
            break  # Critère de coût explicite ; aucune garantie d'optimalité littéraire.
        feedback = (f"The previous draft had {report['actual_lines']} lines; exactly {settings['line_count']} are required. "
                    f"Missing words: {json.dumps(report['missing_words'])}. "
                    f"Lines exceeding 16 words: {report['long_lines']}. "
                    f"First line respected: {report['first_line_ok']}. "
                    f"Token budget exhausted: {report['token_limit_reached']}. "
                    f"Repeated line positions: {[item['lines'] for item in report['repetitions']['repeated_lines']]}. "
                    f"Repeated blocks: {json.dumps(report['repetitions']['repeated_blocks'])}. "
                    f"Repeated phrases: {json.dumps([item['phrase'] for item in report['repetitions']['word_runs']])}. "
                    "Use different lines and phrases. Follow the prefilled-first-line instructions, "
                    "preserve the tone and theme, and finish after the requested number of lines.")
        if initial_correction:
            feedback = initial_correction + " " + feedback
        feedback += (" Stylistic reading signals (not confirmed errors): "
                     + json.dumps([item["code"] for item in review["warnings"]])
                     + ". Deepen the same central image with natural phrasing and a meaningful ending; "
                     "do not explain your choices or print a plan.")
        if not report["repetitions_ok"]:
            arguments["repetition_penalty"] = round(min(1.3, arguments["repetition_penalty"] + .05), 2)
        # N'augmente le budget que si l'utilisateur a autorisé son adaptation.
        if automatic and report["actual_lines"] > settings["line_count"]:
            arguments["max_new_tokens"] = max(50, int(arguments["max_new_tokens"] * .8))
        elif (automatic and report["token_limit_reached"] and report["repetitions_ok"]
              and report["actual_lines"] <= settings["line_count"]):
            arguments["max_new_tokens"] = min(600, arguments["max_new_tokens"] + 50)
    return best[1], {**best[2], "attempts": attempt, "selected_attempt": best[4],
                     "requested_max_new_tokens": requested, "effective_max_new_tokens": best[3],
                     "attempt_budgets": budgets, "attempt_history": history,
                     "effective_parameters": history[best[4] - 1]["parameters"],
                     "selection_criteria": SELECTION_CRITERIA,
                     "literary_review": history[best[4] - 1]["literary_review"],
                     "comparison_mode": comparison, "comparison_seed": comparison_seed,
                     "controlled_candidates": controlled,
                     "generation_seconds": sum(item["elapsed_seconds"] for item in history),
                     "selection_mode": selection_mode,
                     "experiment": {"human_evaluation": "not_performed", "literary_improvement": "not_established",
                                    "controlled_temperature_comparison": comparison}}


def compare_model_candidates(generate, tokenizer, model, settings, activate):
    """Comparer sous le verrou global : mêmes consignes, paramètres et graines appariées."""
    if settings.get("comparison_mode", False):
        raise ValueError("Comparer les modèles et les températures demande deux expériences séparées.")
    selected = settings.get("adapter_version", "v1")
    from adaptateurs import comparison_versions
    versions = comparison_versions(selected)
    common = {k: v for k, v in settings.items() if k not in ("adapter_version", "model_comparison")}
    common.update(controlled_candidates=True, comparison_mode=False, selection_mode="complete",
                  seed=settings.get("comparison_seed", 42))
    results = {}
    try:
        for version in versions:
            metadata = activate(model, version)
            content, report = generate_checked(generate, tokenizer, model, common)
            report["model"] = metadata
            results[version] = {"content": content, "validation": report, "model": metadata}
    finally:
        activate(model, selected)
    # Pas de vainqueur littéraire automatique ; chaque modèle garde son candidat le plus conforme.
    return {"models": results, "protocol": {"same_instructions": True, "paired_seeds": True,
            "same_parameters": True, "adaptive_feedback": False,
            "settings": dict(common),
            "maximum_calls": len(versions) * (3 if common.get("allow_retries", True) else 1),
            "literary_winner": "requires_human_review"}}
