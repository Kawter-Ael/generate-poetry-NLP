# Sélection LoRA V1 / V2

L'application conserve ses trois pages et ses réglages. Dans « Générer un poème »,
le champ « Modèle de poésie » propose LoRA V1 — Ancien modèle et LoRA V2 — Nouveau modèle.
V1 reste le choix initial. Soumettre le formulaire applique la sélection.
« Générer à nouveau » conserve les réglages du dernier poème, y compris sa version.

## Chargement et sécurité entre sessions

- Base unique : `Qwen/Qwen2.5-1.5B-Instruct`.
- V1 : fichiers à la racine ; V2 : fichiers de `adapter_V2/`.
- Le tokenizer commun est celui déjà utilisé à la racine ; les tokenizers ne sont pas modifiés.
- `st.cache_resource(max_entries=1)` conserve une seule base et un verrou partagé.
- V1 est chargé au premier appel ; V2 est ajouté au premier usage, puis reste disponible en mémoire.
- L'activation et toutes les tentatives d'un poème s'exécutent sous le même verrou : une autre session
  ne peut pas changer l'adaptateur pendant la génération. Les poids restent gelés en mode inférence.
- Les chemins, fichiers, modèle de base, type LoRA et tâche sont contrôlés avant utilisation.
- Un changement de fichiers déjà chargés exige un redémarrage ; aucun remplacement silencieux.

Vérifications locales : mêmes 392 noms/formes/types de tenseurs pour les deux adaptateurs,
mais empreintes des poids différentes. Cela établit la compatibilité structurelle et la distinction
des fichiers ; aucune amélioration littéraire n'est affirmée sans comparaison de générations réelles.
Les tests utilisent des modèles simulés, sans charger Qwen ni lancer d'entraînement.

## Provenance des poèmes

Les nouveaux poèmes enregistrent un champ `model` avec version, libellé, modèle de base,
chemin et SHA-256 des poids/configuration. Cette provenance figure dans l'affichage de l'atelier,
la bibliothèque, le téléchargement TXT et le rapport de validation.
Changer la sélection ne réattribue pas la provenance du poème déjà affiché.
Les anciens enregistrements ne sont pas réécrits ; une provenance absente est indiquée comme non renseignée.
La bibliothèque existante stocke déjà les champs supplémentaires sans migration.

## Fichiers

- `app.py` : sélecteur, activation sous verrou, provenance et affichage.
- `generer.py` : chargement paramétrable, base fixe, option CLI `--adapter v1|v2`.
- `adaptateurs.py` : validation, empreintes et activation différée.
- `test_adaptateurs.py` : tests simulés des chemins, chargements, cache, erreurs, provenance et sauvegarde.
- `test_studio.py` : modèles simulés adaptés à la nouvelle activation explicite.
- `poesie.py` et `bibliotheque.py` : analysés et conservés sans modification.

Sauvegarde préalable de tous les fichiers Python : `backups/adapter-selection-20261022-202416/`.
Le fichier `protected_files.json` permet de vérifier les poids, configurations, tokenizer,
poèmes enregistrés et modules métier conservés.

## Lancer l'application

```powershell
Set-Location -LiteralPath 'C:\Users\admin\Desktop\adapter_final-20261021T230132Z-1-001\adapter_final'
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Après modification, arrêter une ancienne instance avec Ctrl+C puis la relancer afin de repartir
d'un cache cohérent. Le premier chargement et la première activation de V2 peuvent prendre du temps.

## Tests

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -B -m unittest test_studio test_evaluation test_adaptateurs -v
```

Les poids, anciens poèmes, notebooks et checkpoints ne sont pas modifiés par cette intégration.
