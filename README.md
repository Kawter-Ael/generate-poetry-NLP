# AI Poetry Studio — Génération de poèmes

Application Streamlit avec une interface en français pour créer des poèmes **en anglais**, à partir de **Qwen2.5-1.5B-Instruct** et de trois adaptateurs **LoRA**.

## Objectif

Adapter un modèle de langage à la poésie, proposer trois versions du modèle et vérifier les contraintes demandées. LoRA ajoute des poids spécialisés au modèle de base sans réentraîner tous ses paramètres.

## Fonctionnalités

- Thème libre ou proposé, ton et première ligne facultative.
- Poèmes de 4, 8, 12 ou 16 vers, avec jusqu'à huit mots imposés.
- Sélection V1, V2 ou V3 ; réglage de la température et du budget de tokens.
- Vérification des vers, mots, répétitions et respect de la première ligne.
- Jusqu'à trois candidats ; comparaison V1/V2, avec V3 si sélectionnée.
- Bibliothèque locale, import CSV/Markdown et export des résultats.

## Démarrage

Voir [run.md](run.md) pour l'installation et toutes les commandes.

```powershell
powershell -ExecutionPolicy Bypass -File .\lancer.ps1
```

Ouvrir [http://localhost:8501](http://localhost:8501), puis **Générer un poème**.

## Structure du projet

| Fichier ou dossier | Rôle |
| --- | --- |
| `app.py` | Interface Streamlit : accueil, génération, bibliothèque |
| `generer.py` | Chargement du tokenizer, de Qwen et génération des tokens |
| `adaptateurs.py` | Validation des adaptateurs et changement de version |
| `poesie.py` | Construction des consignes, validation et sélection des candidats |
| `bibliotheque.py` | Import, lecture et sauvegarde des poèmes |
| `lancer.ps1` | Lanceur Windows utilisant Python dans `.venv` |
| `requirements.txt` | Dépendances Python |
| `.streamlit/config.toml` | Thème de l'application |
| `adapter_config.json`, `adapter_model.safetensors` | Configuration et poids V1 |
| `adapter_V2/` | Configuration et poids V2 |
| `adapter_VERSION3/` | Configuration et poids V3 |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Tokenizer partagé et format des messages |
| `data/poemes_enregistres.json` | Bibliothèque créée localement ; exclue de Git |
| `evaluate_poetry.py`, `evaluation_cases.json` | Protocole expérimental d'évaluation |
| `test_*.py` | Tests automatisés |
| `poesie_lora_evaluation_finale.ipynb` | Notebook d'entraînement et d'évaluation |
| `RAPPORT_PROJET_POESIE_LORA.md` | Rapport du projet |
| `run.md` | Guide d'installation et d'exécution |

Les copies historiques (`backups/`, `*.backup-*.py`, `adapter_best/`, `lora_v2/`) et l'environnement `.venv/` ne sont pas nécessaires dans le dépôt.

## Versions LoRA

Les trois versions partagent la base `Qwen/Qwen2.5-1.5B-Instruct`.

| Version | Emplacement |
| --- | --- |
| V1 | Racine du projet |
| V2 | `adapter_V2/` |
| V3 | `adapter_VERSION3/` |

Chaque emplacement contient `adapter_config.json` et `adapter_model.safetensors`. V3 est aussi vérifiée par son empreinte SHA-256. Les poids sont stockés avec **Git LFS** : installer Git LFS et exécuter `git lfs pull` après le clonage.

## Fonctionnement

L'interface recueille les paramètres, vérifie l'adaptateur et charge le modèle en cache. Le module de poésie construit les consignes et contrôle les candidats. Le résultat retenu est affiché avec ses diagnostics, puis peut être enregistré ou exporté. Un verrou protège le modèle pendant les changements d'adaptateur et les générations.

CUDA est utilisée si disponible ; sinon, la génération fonctionne sur CPU. Le premier chargement peut télécharger le modèle de base depuis Hugging Face. La génération sur CPU peut prendre plusieurs minutes, particulièrement en comparaison de versions.

## Sauvegarde et limites

En local, les textes sont enregistrés dans `data/poemes_enregistres.json`. Lorsque `SPACE_ID` est défini par un hébergement, la bibliothèque est conservée dans la session Streamlit et ne persiste pas après redémarrage.

Les contrôles automatiques mesurent le respect de contraintes et signalent des éléments à relire. Ils ne garantissent pas la qualité littéraire. Une version plus récente n'est pas nécessairement meilleure : la comparaison des poèmes reste humaine.
