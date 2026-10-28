# Candidats et comparaison LoRA V1 / V2

## Constat sur le code existant

Les trois candidats, la validation du nombre de vers, de la première ligne, des mots entiers,
des vers et groupes répétés existaient déjà. Les signaux stylistiques restaient explicitement
heuristiques. `generer.py` décodait bien uniquement les nouveaux tokens et conservait les vers
excédentaires afin de les signaler, plutôt que de masquer les échecs.

La comparaison disponible portait sur trois températures, pas sur V1/V2. Les corrections de prompt,
hausses de pénalité et adaptations du budget entre relances empêchaient une comparaison contrôlée
directe des deux adaptateurs.

## Changements

- `poesie.py` : nouveau contrôleur V1/V2, paramètres fixes, graines appariées, conservation des sorties
  et restauration de l'adaptateur choisi, même après une erreur ; normalisation des apostrophes
  droites/courbes dans les mots obligatoires ; détection d'une ponctuation finale de continuation.
- `app.py` : option « Comparer LoRA V1 et V2 », affichage des deux candidats retenus, export JSON
  de tous les candidats et du protocole, téléchargement et enregistrement séparés, affichage
  explicite du nombre de vers et des répétitions détectées.
- `test_comparaison_modeles.py` : tests simulés supplémentaires.
- `generer.py`, les modules de chargement et de bibliothèque ne nécessitaient pas de changement.

## Utilisation

Dans les paramètres avancés, cocher « Comparer LoRA V1 et V2 ». Le modèle du sélecteur reste
le poème principal, l'autre est affiché à la suite. Aucun gagnant littéraire n'est proclamé.
Avec les tentatives supplémentaires : trois candidats par modèle, donc six appels au maximum.
Sans elles : un candidat par modèle, donc deux appels. Le temps CPU peut augmenter sensiblement.
Les poids du modèle de base restent partagés en cache ; les activations sont protégées par le verrou.

La demande, la température, top-p, la pénalité, les n-grammes, le budget initial et les graines
sont identiques entre V1 et V2. Les tentatives utilisent seed, seed+1, seed+2. Les corrections
adaptatives et l'arrêt Économe sont suspendus dans cette comparaison pour assurer des conditions égales.
Le budget automatique initial reste applicable aux deux modèles de la même manière, sans modification
ultérieure. La comparaison de températures reste disponible séparément.

Chaque modèle sélectionne son candidat selon l'ordre technique existant : conformité complète,
répétitions, première ligne, écart au nombre de vers, mots manquants, interruption liée au budget,
longueur des vers, puis signaux heuristiques de fin et de formulations récurrentes. Ces derniers
ne sont ni des preuves de faute ni une mesure objective de créativité. Une fin sans point
n'est pas rejetée automatiquement. Aucun vers n'est supprimé ou réécrit pour satisfaire les contrôles.

L'export contient tous les candidats, paramètres, graines, validations et empreintes des adaptateurs.
Les poèmes existants ne sont pas réécrits. Un candidat non conforme reste affiché comme brouillon.
La cohérence sémantique, le ton et l'originalité nécessitent toujours une lecture humaine.

Sauvegarde préalable : `backups/candidate-comparison-20261022-211917/`.
Les tests sont réalisés avec modèle simulé, sans génération coûteuse ni entraînement.

## Commandes PowerShell

```powershell
Set-Location -LiteralPath 'C:\Users\admin\Desktop\adapter_final-20261021T230132Z-1-001\adapter_final'
& '.\.venv\Scripts\python.exe' -X utf8 -B -m unittest test_studio test_evaluation test_adaptateurs test_comparaison_modeles -v
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Arrêter une instance déjà ouverte avec Ctrl+C avant de relancer.
