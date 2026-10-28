"""AI Poetry Studio — lancer avec python -m streamlit run app.py."""

import threading
import re
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import streamlit as st

from adaptateurs import ADAPTER_LABELS, adapter_info, activate_adapter, model_label, comparison_versions
from bibliotheque import import_records, load_poems, save_poems
from poesie import (TONES, generate_checked, parse_words, token_budget, check_poem,
                    DEFAULT_REPETITION_PENALTY, DEFAULT_NO_REPEAT_NGRAM_SIZE,
                    DEFAULT_TEMPERATURE, literary_review, human_comparison_csv, compare_model_candidates)

st.set_page_config(page_title="AI Poetry Studio", page_icon=":material/history_edu:", layout="wide")
THEMES = ["Love", "Nature", "Hope", "Memories", "Ocean", "Autumn", "Dreams", "Loneliness"]
DEFAULTS = {"temperature": DEFAULT_TEMPERATURE, "top_p": 0.9, "max_new_tokens": 200,
            "repetition_penalty": DEFAULT_REPETITION_PENALTY,
            "no_repeat_ngram_size": DEFAULT_NO_REPEAT_NGRAM_SIZE}


@st.cache_resource(max_entries=1)
def cached_model():
    # Import différé : les autres pages ne chargent pas PyTorch ou Qwen.
    from generer import load_model
    tokenizer, model = load_model()
    return tokenizer, model, threading.Lock()


def load_library():
    if os.environ.get("SPACE_ID"):
        return st.session_state.setdefault("saved_poems", [])
    return load_poems()


def save_library(records):
    if not os.environ.get("SPACE_ID"):
        return save_poems(records)
    poems = st.session_state.setdefault("saved_poems", [])
    known = {poem["id"] for poem in poems}
    added = 0
    for poem in records:
        if poem["id"] not in known:
            poems.append(poem)
            known.add(poem["id"])
            added += 1
    return added


def current_report(poem):
    """Revalide en mémoire les anciennes métadonnées, sans modifier la bibliothèque."""
    report = poem.get("validation")
    if report and report.get("validation_version") != 2:
        return {**report, **check_poem(
            poem["content"], report["requested_lines"],
            report.get("present_words", []) + report.get("missing_words", []),
            poem.get("first_line", ""), report.get("token_limit_reached", False))}
    return report


def display_poem(poem):
    with st.container(border=True):
        st.caption("AI POETRY STUDIO · COLLECTION PERSONNELLE")
        st.subheader(poem.get("title") or "Sans titre")
        st.caption("Modèle : " + model_label(poem))
        theme_label = poem.get('theme') or ('Inspiration libre' if poem.get('validation') else 'Non renseigné')
        st.caption(f"Thème : {theme_label} · "
                   f"Création : {poem.get('created_at') or 'Date non renseignée'}")
        if poem.get("parameters", {}).get("tone"):
            st.caption("Ton demandé : " + poem["parameters"]["tone"])
        # Échappe la syntaxe Markdown et préserve les retours à la ligne du poème.
        st.markdown(":material/format_quote:")
        escaped = re.sub(r"([\\`*_{}\[\]()#+.!|>~$<>-])", r"\\\1", poem["content"])
        st.markdown(escaped.replace("\n", "  \n"))
    report = current_report(poem)
    if report:
        review = report.get("literary_review") or literary_review(
            poem["content"], poem.get("parameters", {}).get("tone", "Libre"), poem.get("first_line", ""))
        st.caption(f"Vers : {report['actual_lines']} / {report['requested_lines']} demandés "
                   f"· Tentatives : {report['attempts']} · Lignes vides non comptées")
        st.write("Mots présents : " + (", ".join(report["present_words"]) or "Aucun"))
        st.write("Mots manquants : " + (", ".join(report["missing_words"]) or "Aucun"))
        st.caption("Nombre exact de vers : " + ("respecté" if report["line_count_ok"] else "non respecté")
                   + " · Absence de répétitions détectées : " + ("oui" if report["repetitions_ok"] else "non"))
        if report.get("effective_max_new_tokens"):
            st.caption(f"Budget du résultat retenu : {report['effective_max_new_tokens']} tokens "
                       f"· Tentative retenue : {report.get('selected_attempt', 1)}")
        if poem.get("first_line", "").strip():
            st.caption("Première ligne : " + ("respectée, sans répétition" if report["first_line_ok"] else "non respectée"))
        if report['conformant'] and review.get('ending_warning_count', 0):
            st.warning("Contraintes de forme respectées, mais la fin paraît suspendue : ce poème reste à relire.")
        elif report["conformant"]:
            st.success("Contraintes vérifiées : nombre de vers, mots demandés, première ligne et absence de répétitions détectées.")
        else:
            st.error("Génération non conforme après " + str(report["attempts"]) +
                     " tentative(s). Ce texte reste un brouillon, pas une génération réussie.")
        for anomaly in report.get("anomalies", []):
            st.warning(anomaly)
        if report.get("attempt_history"):
            with st.expander("Détails des tentatives et paramètres utilisés"):
                st.caption("Critères de sélection : " + report["selection_criteria"])
                st.json([
                    {"Tentative": item["attempt"], "Retenue": item["attempt"] == report["selected_attempt"],
                     **item["parameters"], "Conforme": item["validation"]["conformant"],
                     "Vers": item["validation"]["actual_lines"],
                     "Durée (s, hors chargement)": item.get("elapsed_seconds"),
                     "Arrêt": item["stop_reason"], "Anomalies": " · ".join(item["validation"]["anomalies"]),
                     "Indices stylistiques (indicatifs)": [w["message"] for w in item.get("literary_review", {}).get("warnings", [])]}
                    for item in report["attempt_history"]])
        with st.expander("Lecture littéraire : indices et appréciation humaine"):
            st.caption("Contraintes formelles et qualité littéraire sont distinctes. Aucune note humaine n'est calculée automatiquement.")
            if review["warnings"]:
                for signal in review["warnings"]:
                    st.info(signal["message"])
            else:
                st.write("Aucun indice stylistique prévu par ces règles n'a été détecté ; cela ne certifie pas la qualité du texte.")
            st.write("Cohérence des images et du thème : à apprécier par un lecteur.")
            for criterion in review["human_rubric"]:
                st.write(criterion["criterion"] + " (0 à 2, à renseigner par un lecteur) : " + criterion["question"])
            st.caption(review["limits"])
        if report.get("comparison_mode"):
            st.info("Comparaison de températures : un candidat par profil, même seed et même budget. La supériorité littéraire d'un profil reste à établir par une lecture humaine.")
            st.download_button("Exporter les candidats et paramètres", json.dumps(report, ensure_ascii=False, indent=2),
                               "comparaison_poemes.json", "application/json", key="comparison_json_" + poem["id"])
            st.download_button("Grille humaine aveugle sans notes", human_comparison_csv(report),
                               "evaluation_humaine.csv", "text/csv", key="comparison_human_" + poem["id"])
        st.caption("Le ton, le respect du thème et la cohérence de la conclusion restent à apprécier à la lecture. "
                   "La ponctuation seule ne garantit pas une fin complète.")


def download_poem(poem, key):
    text = f"{poem.get('title') or 'Sans titre'}\nModèle : {model_label(poem)}\n\n{poem['content']}"
    st.download_button("Télécharger en TXT", text, "poeme.txt", "text/plain",
                       key=key, icon=":material/download:")


def home_page():
    st.caption("L'IMAGINATION, UNE LIGNE À LA FOIS")
    st.title("AI Poetry Studio")
    st.header("Donnez une voix à vos idées.")
    st.write("Un thème, quelques mots, une première ligne : ouvrez un espace de création "
             "où vos idées deviennent des poèmes en anglais.")
    if st.button("Commencer à écrire", type="primary", icon=":material/edit:"):
        st.switch_page(generation_page)
    st.divider()
    with st.container(horizontal=True):
        for icon, title, body in [
            ("edit", "Imaginez", "Choisissez un thème ou proposez le vôtre."),
            ("auto_awesome", "Composez", "Explorez différentes nuances de créativité."),
            ("bookmarks", "Conservez", "Enregistrez vos textes dans votre bibliothèque locale."),
        ]:
            with st.container(border=True, width="stretch"):
                st.subheader(title, icon=f":material/{icon}:")
                st.write(body)
    with st.container(border=True):
        st.subheader("Un projet de création, une exploration du langage")
        st.write("Ce projet NLP associe Qwen2.5-1.5B-Instruct à un adaptateur LoRA entraîné "
                 "sur la poésie. Qwen apporte ses capacités linguistiques ; LoRA adapte "
                 "la génération à l'écriture poétique.")
        st.caption("Les textes sont générés par une IA. Relisez-les pour affiner leurs images et leur cohérence.")


def poem_record(content, validation, metadata, settings):
    return {
        "id": str(uuid4()), "title": content.splitlines()[0][:100] if content else "Sortie vide",
        "theme": settings["theme"], "first_line": settings["first_line"],
        "content": content, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {**{k: settings[k] for k in (*DEFAULTS, "line_count", "required_words",
                                                   "tone", "allow_retries", "auto_budget")},
                       "comparison_mode": settings["comparison_mode"], "comparison_seed": settings["comparison_seed"],
                       "selection_mode": validation.get("selection_mode", settings['selection_mode']),
                       "requested_selection_mode": settings['selection_mode'], "model_comparison": settings["model_comparison"],
                       "effective_parameters": validation["effective_parameters"]},
        "validation": validation, "model": dict(metadata),
    }


def run_generation(settings):
    # Valeurs compatibles avec les brouillons d'une ancienne session.
    settings = {**DEFAULTS, "line_count": 8, "required_words": [], "tone": "Libre",
                "allow_retries": True, "auto_budget": True,
                "comparison_mode": False, "comparison_seed": 42, "selection_mode": "complete",
                "adapter_version": "v1", "model_comparison": False, **settings}
    try:
        selected_adapter = adapter_info(settings["adapter_version"])
        if settings["model_comparison"]:
            for version in comparison_versions(settings["adapter_version"]):
                adapter_info(version)
            st.info("Comparaison V1/V2, avec V3 s'il est sélectionné : mêmes consignes, graines et paramètres, sans corrections adaptatives. "
                    "Jusqu'à trois candidats par modèle. Aucun gagnant littéraire automatique.")
        effective = token_budget(settings["max_new_tokens"], settings["line_count"], settings["auto_budget"])
        mode_label = ('3 profils contrôlés' if settings['comparison_mode'] else
                      'jusqu’à 3 candidats, arrêt anticipé possible' if settings['selection_mode'] == 'economy' else
                      '3 candidats maximum, même si le premier est conforme')
        st.caption(f"Budget initial : {effective} tokens · "
                   f"{mode_label if settings['allow_retries'] else '1 seule tentative'} "
                   "· Plafond : 600 tokens par tentative.")
        with st.spinner("Préparation de votre atelier… Le premier chargement peut être long."):
            tokenizer, model, lock = cached_model()
        from generer import generate_poem
        with st.spinner("Votre poème prend forme… Sur CPU, cela peut prendre plusieurs minutes."):
            with lock:
                comparison = None
                if settings["model_comparison"]:
                    comparison = compare_model_candidates(generate_poem, tokenizer, model, settings, activate_adapter)
                    chosen = comparison["models"][settings["adapter_version"]]
                    content, validation, selected_adapter = chosen["content"], chosen["validation"], chosen["model"]
                else:
                    selected_adapter = activate_adapter(model, settings["adapter_version"])
                    generation_settings = {k: v for k, v in settings.items()
                                           if k not in ("adapter_version", "model_comparison")}
                    content, validation = generate_checked(generate_poem, tokenizer, model, generation_settings)
        validation["model"] = dict(selected_adapter)
        if not content:
            st.warning("Le modèle a produit un texte vide. Essayez une nouvelle génération.")
            if comparison is None:
                return False
        draft = poem_record(content, validation, selected_adapter, settings)
        if comparison is not None:
            draft["adapter_comparison"] = {"protocol": comparison["protocol"], "poems": {
                version: poem_record(result["content"], result["validation"], result["model"], settings)
                for version, result in comparison["models"].items()}}
        st.session_state.draft = draft
        st.session_state.last_settings = dict(settings)
        return True
    except Exception as exc:
        st.error("La génération n'a pas abouti. Vérifiez les fichiers de l'adaptateur sélectionné, "
                 "le téléchargement de Qwen et la mémoire disponible.")
        with st.expander("Détails pour le diagnostic"):
            st.text(str(exc))
        return False


def generation_view():
    st.caption("L'ATELIER")
    st.title("Générer un poème")
    st.write("Choisissez votre inspiration. Les poèmes sont composés en anglais.")
    # Les options avancées dépendent des cases cochées : elles doivent être
    # recalculées immédiatement, sans lancer la génération.
    with st.container():
        st.subheader("Votre inspiration", icon=":material/edit:")
        adapter_version = st.selectbox("Modèle de poésie", list(ADAPTER_LABELS),
                                       format_func=ADAPTER_LABELS.__getitem__, key="adapter_version")
        st.caption("Même base Qwen2.5-1.5B-Instruct ; seul l'adaptateur LoRA change.")
        selected = st.selectbox("Thème proposé (facultatif)", ["Inspiration libre", *THEMES], index=0)
        custom = st.text_input("Votre thème (facultatif, en anglais)", max_chars=500,
                               placeholder="Ex. the sea at sunset")
        first = st.text_input("Première ligne personnalisée (facultatif, en anglais)", max_chars=500)
        tone = st.selectbox("Ton du poème", list(TONES), index=list(TONES).index("Libre"),
                            help="Le ton guide l'émotion. Love + Romantique privilégie la tendresse plutôt que la tristesse.")
        st.caption("Vous pouvez laisser les deux champs vides : le modèle choisira librement son sujet. "
                   "Votre thème saisi est prioritaire sur la suggestion.")
        line_count = st.selectbox("Nombre de vers", [4, 8, 12, 16], index=1)
        words_text = st.text_input("Mots à inclure", max_chars=350, placeholder="moon, stars, heart, dream",
                                   help="8 mots distincts maximum, séparés par des virgules. Vérification sur les mots entiers.")
        with st.expander("Paramètres avancés", icon=":material/tune:"):
            temperature = st.slider("Créativité · température", 0.1, 1.5, DEFAULT_TEMPERATURE, 0.05,
                                    help="Une valeur élevée favorise la diversité et peut réduire la cohérence.")
            top_p = st.slider("Diversité · Top-p", 0.1, 1.0, 0.9, 0.05)
            repetition_penalty = st.slider("Pénalité de répétition", 1.0, 1.3, DEFAULT_REPETITION_PENALTY, 0.05,
                                            help="Valeur initiale. Après une répétition détectée, l'essai suivant ajoute 0.05, jusqu'à 1.30. Sans garantie de qualité.")
            no_repeat_ngram_size = st.slider("Taille des n-grammes interdits", 0, 6, DEFAULT_NO_REPEAT_NGRAM_SIZE, 1,
                                             help="3 interdit de répéter un groupe de 3 tokens ; 0 désactive cette protection. La validation reste active.")
            maximum = st.slider("Longueur maximale (tokens)", 50, 600, 200, 25,
                                help="Limite par tentative, pas une garantie de longueur. Prévoir davantage pour 12 ou 16 vers.")
            auto_budget = st.checkbox("Adapter automatiquement le budget de tokens", value=True,
                                       help="Budget minimal : 120 / 200 / 280 / 360 tokens pour 4 / 8 / 12 / 16 vers. "
                                            "Après un dépassement, le budget baisse de 20 %. Après interruption sans répétition, il peut augmenter de 50 tokens, jusqu'à 600. "
                                            "Décochez pour respecter strictement votre limite.")
            allow_retries = st.checkbox("Autoriser jusqu'à deux tentatives supplémentaires", value=True,
                                         help="Désactivé : un seul appel. Activé : trois candidats comparés, même si le premier est conforme. Le temps CPU augmente.")
            strategy = st.selectbox('Sélection des candidats', ['Comparaison complète', 'Économe'],
                                    disabled=not allow_retries,
                                    help="Économe : arrêt dès qu'un candidat est conforme sans indice stylistique détecté, trois appels au maximum. Ce choix ne garantit pas le meilleur poème. La comparaison de températures conserve toujours ses trois appels.")
            model_comparison = st.checkbox("Comparer LoRA V1 et V2", value=False,
                                           help="V3 est inclus s'il est sélectionné. Même demande, mêmes paramètres et graines. Trois candidats par modèle si les tentatives sont autorisées, sinon un. Six appels maximum pour V1/V2, neuf avec V3.")
            comparison_mode = st.checkbox("Comparer les températures 0.70 / 0.75 / 0.80", value=False,
                                          disabled=not allow_retries or model_comparison,
                                          help="Trois appels au total. Même seed, prompt et budget ; top_p=0.9, pénalité=1.15, n-grammes=3. Ces profils remplacent les réglages avancés uniquement pendant la comparaison.")
            comparison_seed = st.number_input("Seed de comparaison", min_value=0, max_value=2147483647,
                                              value=42, step=1, disabled=not (model_comparison or comparison_mode and allow_retries))
            if model_comparison:
                st.caption("Comparaison V1/V2 (avec V3 si sélectionné) : sélection complète avec paramètres fixes ; le mode Économe et les adaptations sont suspendus pour garder un protocole identique.")
            if comparison_mode and allow_retries and not model_comparison:
                st.caption("Paramètres contrôlés : seule la température varie ; pas de correction du prompt ni d'adaptation du budget entre profils. Aucun profil n'est présumé meilleur.")
        st.caption("Des vers courts, des images poétiques et une conclusion cohérente sont demandés au modèle. "
                   "La qualité littéraire reste à vérifier à la lecture.")
        submitted = st.button("Générer le poème", type="primary", icon=":material/auto_awesome:")
    if submitted:
        try:
            words = parse_words(words_text)
            theme = custom.strip() or (selected if selected != "Inspiration libre" else "")
            # Une première ligne fournie est conservée à l'identique, espaces compris.
            run_generation({"adapter_version": adapter_version, "theme": theme, "first_line": first if first.strip() else "",
                            "line_count": line_count, "required_words": words,
                            "tone": tone, "allow_retries": allow_retries, "auto_budget": auto_budget,
                            "temperature": temperature, "top_p": top_p, "max_new_tokens": maximum,
                            "repetition_penalty": repetition_penalty,
                            "no_repeat_ngram_size": no_repeat_ngram_size,
                            "comparison_mode": comparison_mode and allow_retries and not model_comparison,
                            "model_comparison": model_comparison,
                            "selection_mode": 'economy' if strategy == 'Économe' else 'complete',
                            "comparison_seed": int(comparison_seed)})
        except ValueError as exc:
            st.warning(str(exc))
    draft = st.session_state.draft
    if draft:
        st.divider()
        title = st.text_input("Titre de votre poème", value=draft["title"], key="title_" + draft["id"], max_chars=200)
        draft["title"] = title.strip() or "Sans titre"
        display_poem(draft)
        if draft.get("adapter_comparison"):
            comparison = draft["adapter_comparison"]
            versions_label = " / ".join(v.upper() for v in comparison["poems"])
            st.subheader("Comparaison " + versions_label)
            st.caption("Chaque candidat retenu est classé par conformité technique et signaux de relecture. "
                       "La cohérence et la créativité restent à apprécier humainement.")
            for version, peer in comparison["poems"].items():
                if version == draft["model"]["version"]:
                    continue  # Le modèle sélectionné est déjà affiché juste au-dessus.
                display_poem(peer)
                download_poem(peer, "download_peer_" + peer["id"])
                label = "Enregistrer le candidat " if peer["validation"]["conformant"] else "Enregistrer le brouillon "
                if st.button(label + version.upper(), key="save_peer_" + peer["id"]):
                    try:
                        count = save_library([peer])
                        st.info("Candidat enregistré." if count else "Ce candidat est déjà enregistré.")
                    except (OSError, ValueError, TimeoutError) as exc:
                        st.error(f"Enregistrement impossible : {exc}")
            st.download_button("Exporter la comparaison " + versions_label, json.dumps(comparison, ensure_ascii=False, indent=2),
                               "comparaison_" + "_".join(comparison["poems"]) + ".json", "application/json", key="models_json_" + draft["id"])
        with st.container(horizontal=True):
            nonconformant = (current_report(draft) or {}).get("conformant") is False
            if st.button("Enregistrer le brouillon" if nonconformant else "Enregistrer le poème", icon=":material/bookmark_add:"):
                try:
                    record = {**draft, "validation": current_report(draft)} if draft.get("validation") else draft
                    count = save_library([record])
                    if nonconformant:
                        st.info("Brouillon non conforme enregistré." if count else "Ce brouillon est déjà enregistré.")
                    else:
                        st.success("Poème enregistré dans la bibliothèque." if count else "Ce poème est déjà enregistré.")
                except (OSError, ValueError, TimeoutError) as exc:
                    st.error(f"Enregistrement impossible : {exc}")
            download_poem(draft, "download_draft")
            if st.button("Générer à nouveau", icon=":material/refresh:"):
                if run_generation(st.session_state.last_settings):
                    st.rerun()


def import_panel():
    with st.expander("Importer d'anciens poèmes", icon=":material/upload_file:"):
        st.write("Importez un CSV (colonnes title/titre, theme/thème et content/poem/texte/contenu) "
                 "ou un Markdown contenant un seul poème. Les fichiers originaux restent intacts.")
        st.caption("Les dates ne sont conservées que si elles figurent dans le CSV. "
                   "Pour un Markdown, le premier titre # devient le titre du poème.")
        uploaded = st.file_uploader("Fichier UTF-8 à importer", type=["csv", "md", "markdown"])
        if uploaded:
            try:
                records = import_records(uploaded.name, uploaded.getvalue())
                st.caption(f"{len(records)} poème(s) reconnu(s)")
                display_poem(records[0])
                if st.button("Ajouter à la bibliothèque", icon=":material/library_add:"):
                    count = save_library(records)
                    st.session_state.import_notice = f"{count} poème(s) ajouté(s). Les doublons ont été ignorés."
                    st.rerun()
            except (OSError, ValueError, TimeoutError) as exc:
                st.error(f"Import impossible : {exc}")


def library_page():
    st.caption("VOTRE COLLECTION")
    st.title("Bibliothèque de poèmes")
    st.write("Retrouvez les textes que vous avez choisi de conserver.")
    if os.environ.get("SPACE_ID"):
        st.caption("Bibliothèque privée à cette session; les poèmes ne sont pas conservés après redémarrage.")
    if st.session_state.import_notice:
        st.success(st.session_state.pop("import_notice"))
    import_panel()
    try:
        poems = load_library()
    except (OSError, ValueError) as exc:
        st.error(f"La bibliothèque ne peut pas être lue : {exc}. Son fichier n'a pas été modifié.")
        return
    if not poems:
        st.info("Votre bibliothèque est vide. Enregistrez un poème depuis l'atelier ou importez un fichier.")
        return
    query = st.text_input("Rechercher par titre ou thème", placeholder="Un mot, une émotion…").casefold().strip()
    filtered = [p for p in poems if query in f"{p.get('title', '')} {p.get('theme', '')}".casefold()]
    st.caption(f"{len(filtered)} poème(s) sur {len(poems)}")
    if not filtered:
        st.info("Aucun poème ne correspond à votre recherche.")
        return
    index = st.selectbox("Choisir un poème", range(len(filtered)),
                         format_func=lambda i: f"{filtered[i].get('title') or 'Sans titre'} · {filtered[i].get('theme') or 'Sans thème'}")
    poem = filtered[index]
    display_poem(poem)
    download_poem(poem, "download_library")


# État commun aux trois pages : les brouillons survivent à la navigation.
for key, value in {"draft": None, "last_settings": None, "import_notice": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Préserve le texte de l'ancienne interface si sa session est encore ouverte.
if st.session_state.draft is None and st.session_state.get("poem"):
    st.session_state.draft = {"id": str(uuid4()), "title": "Poème de la session précédente",
                             "theme": "", "first_line": "", "content": st.session_state.poem,
                             "created_at": None, "parameters": {}}
    st.session_state.last_settings = {"theme": "the sea at sunset", "first_line": "", **DEFAULTS}

with st.sidebar:
    st.header("AI Poetry Studio", icon=":material/history_edu:")
    st.caption("Un espace pour les mots qui vous habitent.")

generation_page = st.Page(generation_view, title="Générer un poème", icon=":material/edit:", url_path="generer")
page = st.navigation([
    st.Page(home_page, title="Accueil", icon=":material/home:", default=True),
    st.Page(library_page, title="Bibliothèque de poèmes", icon=":material/bookmarks:", url_path="bibliotheque"),
    generation_page,
], position="sidebar")
with st.sidebar:
    st.divider()
    st.caption("Qwen2.5 · LoRA\n\nPoésie en anglais · Sauvegarde locale")
page.run()
