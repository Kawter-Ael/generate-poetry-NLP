# Qualité littéraire : méthode et limites

Cette modification n'a entraîné aucun modèle. Les tests utilisent un modèle
simulé et un pilote réel séparé compare les versions sur dix thèmes fixes.
Les mesures techniques de ce pilote ne prouvent pas une amélioration littéraire.
L'objectif de 8 à 9/10 est
une ambition d'évaluation humaine, jamais une note automatique ni une garantie.

## Génération courante

Le prompt adapte le ton et la longueur, développe une image centrale et propose,
pour huit vers, une progression souple : scène (1–2), émotion (3–4), image
approfondie (5–6), résolution (7–8). La première ligne fournie est préremplie
à l'identique. Les mots imposés doivent apparaître comme mots entiers.
Aucun minimum de mots par vers n'est imposé ; le maximum de 16 mots est conservé,
sauf pour la première ligne imposée.

Les répétitions et vers excédentaires ne sont pas retirés pour rendre le texte
conforme. Si les tentatives supplémentaires sont autorisées, au plus trois
candidats sont produits en mode « Comparaison complète », même si le premier est
conforme. Le mode « Économe » s'arrête dès qu'un candidat est conforme sans aucun
indice stylistique détecté ; il ne garantit pas le meilleur poème possible.
Si les tentatives supplémentaires sont désactivées, un seul appel est effectué.
En génération courante, une répétition détectée peut augmenter la
pénalité suivante de 0.05, jusqu'à 1.30. Le budget ne s'adapte que si son adaptation
a été autorisée. Le coût CPU peut atteindre celui de trois générations.

## Sélection et relecture

Le classement lexicographique privilégie la conformité complète, l'absence de
répétitions, leur nombre, la première ligne, le nombre de vers, les mots imposés,
l'absence d'interruption et les vers trop longs. Les indices stylistiques ne
départagent qu'ensuite les candidats : les fins suspectes sont prioritaires,
puis les formulations récurrentes, puis les autres alertes. Une expression
courante ne pèse donc plus autant qu'une conclusion apparemment interrompue.
À égalité, le premier est retenu.

Les alertes stylistiques portent sur les séquences de plus de 30 mots sans
ponctuation forte, certaines fins suspendues, les débuts de vers très similaires
et quelques expressions courantes. Les expressions de 3 à 5 mots contenant au
moins deux mots autres que les mots-outils, présentes dans au moins trois vers,
sont aussi signalées, ainsi que certains délimiteurs non refermés.
Ces heuristiques peuvent signaler un choix
poétique intentionnel. Elles ne modifient pas la conformité formelle, ne
certifient pas la grammaire ou la cohérence sémantique et ne constituent pas une
note littéraire. L'absence d'alerte ne signifie pas qu'un poème est réussi.

La cohérence, le naturel, l'originalité, la justesse du ton et la conclusion
restent à évaluer par des lecteurs. Aucun juge LLM supplémentaire n'est chargé.
Une fin suspecte produit un avertissement visible, même si les contraintes de
forme sont respectées. Aucun vers n'est supprimé et aucune ponctuation n'est
ajoutée pour masquer le problème.

## Pilote avant/après sur dix thèmes

`evaluation_cases.json` fixe dix demandes : amour, automne, océan, espoir,
enfance, nuit, distance, amitié, ville à l'aube et renouveau. Chacune impose huit
vers, une première ligne, trois mots entiers et la seed 42. Les deux phases
emploient température 0.8, top_p 0.9, pénalité 1.15, n-grammes 3 et plafond de
200 tokens, avec quatre threads CPU. Elles font dix appels chacune, un seul
candidat par thème, sans adaptation ni réessai.

La référence utilise la copie du code conservée dans
`backups/pilot-before-after-20261022-144706`. Les résultats sont distincts de la
bibliothèque : `data/evaluation_pilot_20261022/`. Les poids et le tokenizer
sont vérifiés par empreintes avant chaque phase. Les résultats avant/après sont
relus avec le même évaluateur après correction pour éviter de confondre évolution
du détecteur et évolution des poèmes. Les validations d'origine sont aussi
conservées dans les données brutes.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B evaluate_poetry.py --phase before --snapshot backups/pilot-before-after-20261022-144706 --output data/evaluation_pilot_20261022 --resume
.\.venv\Scripts\python.exe -X utf8 -B evaluate_poetry.py --phase after --output data/evaluation_pilot_20261022 --resume
.\.venv\Scripts\python.exe -X utf8 -B evaluate_poetry.py --phase report --output data/evaluation_pilot_20261022
```

Les phases terminées ne sont pas régénérées avec `--resume`. Le rapport indique
les taux de conformité, mots présents, répétitions, fins suspectes et temps réels
hors chargement. Une grille humaine aveugle à notes vides et sa correspondance
privée sont produites. La charge CPU et les caches peuvent différer entre phases.
Dix thèmes et une seed ne suffisent pas à généraliser. Ce pilote à un candidat
ne mesure pas le bénéfice réel du classement sur trois candidats ou du mode
économe ; ces chemins sont vérifiés par des simulations.

## Comparaison contrôlée préparée, résultats littéraires en attente

Le mode optionnel compare les températures 0.70, 0.75 et 0.80. Il garde top_p=0.90,
repetition_penalty=1.15, no_repeat_ngram_size=3, le prompt, les contraintes, la
seed et le budget identiques. Il effectue un seul appel par profil, trois au
total, sans correction du prompt, adaptation du budget ni réessai imbriqué.
Ces profils remplacent temporairement les paramètres avancés lorsque le mode
est explicitement activé. Le défaut courant reste 0.80 ; aucun profil n'est
présumé universellement meilleur. Les RNG sont restaurés après un appel seedé.
Une seed ne garantit pas une identité bit à bit entre matériels et versions.
Les durées de chaque appel sont mesurées hors chargement du modèle.

L'interface peut exporter les candidats et leurs paramètres en JSON ainsi qu'une
grille CSV à labels A/B/C mélangés, sans paramètres ni notes préremplies.
Le JSON technique permet de retrouver les correspondances ; il doit être conservé
séparément des fichiers remis aux évaluateurs si l'on souhaite une lecture aveugle.

Pour une comparaison réelle ultérieure, utiliser plusieurs demandes couvrant les
tons, longueurs et mots imposés, ainsi que plusieurs seeds appariées. Ne pas
choisir uniquement les meilleurs exemples. Rapporter la conformité, les durées,
les notes humaines et leur dispersion. Le nombre de prompts et de lecteurs et
les résultats doivent provenir d'exécutions et annotations réelles.

Chaque lecteur peut noter de 0 à 2 les cinq axes suivants : cohérence/progression,
naturel/rythme, imagerie/originalité, justesse du ton et conclusion, soit 10 points
au total. Les cellules de notation exportées sont vides. L'amélioration
littéraire reste « non établie » tant que cette évaluation n'a pas eu lieu.

## Vérification sans charger Qwen

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest test_studio test_evaluation -v
```

Les tests utilisent des sorties simulées, y compris pour Streamlit. Ils vérifient
le protocole et les contrôles, pas une note de qualité obtenue avec le modèle.
