# Génération automatique de poésie anglaise avec Deep Learning, Qwen2.5 et LoRA

## Rapport technique, pédagogique et de valorisation

**Version documentaire :** 24 octobre 2026  
**Statut :** prototype documenté à partir des fichiers présents dans le projet  
**Périmètre analysé :** application Streamlit, adaptateurs LoRA, notebooks, tests et pilote d’évaluation

> Ce rapport distingue les résultats effectivement observés dans les fichiers du projet des hypothèses ou informations restant à confirmer. Toute donnée non établie est signalée par **À vérifier**.

## Résumé exécutif

Le projet vise à générer des poèmes en anglais à partir d’instructions contrôlées : thème, ton, première ligne, nombre de vers, mots obligatoires et paramètres de décodage. Le socle est `Qwen/Qwen2.5-1.5B-Instruct`, adapté par LoRA/PEFT. L’application [app.py](./app.py) fournit un atelier de génération, une bibliothèque locale et des contrôles de conformité.

Les fichiers disponibles établissent plusieurs faits : le corpus Poetry Foundation contient 13 854 poèmes dans le contrôle de dataset du notebook joint ; les adaptateurs V2 et V3 sont compatibles structurellement avec Qwen et leurs fichiers sont vérifiables par SHA-256 ; l’application conserve les sorties non conformes au lieu de les masquer ; un pilote de dix cas avant/après existe. Dans ce pilote, aucun des vingt résultats n’est entièrement conforme : les mots obligatoires sont le principal échec. Le temps moyen mesuré passe d’environ 59,6 secondes avant à 48,9 secondes après, mais ce pilote ne permet pas d’attribuer cette différence à une amélioration littéraire.

Le projet est donc un prototype de recherche et de démonstration solide, mais pas encore un produit commercial prêt à l’emploi. Les priorités sont l’augmentation et la diversification des données d’instruction, une évaluation humaine réellement renseignée, une comparaison contrôlée des versions LoRA et une stratégie d’inférence GPU ou de service distant.

## Table des matières

1. Contexte et problématique  
2. Fondements NLP  
3. Données et préparation  
4. Architectures et choix techniques  
5. Entraînement et fine-tuning  
6. Génération et contrôle des contraintes  
7. Évaluation des résultats  
8. Application Streamlit  
9. Qualité logicielle et reproductibilité  
10. Déploiement et exploitation  
11. Valorisation commerciale  
12. Limites, feuille de route et conclusion  
Annexe A. Glossaire  
Annexe B. Sources et éléments à vérifier

## Liste des abréviations

| Abréviation | Signification |
|---|---|
| IA | Intelligence artificielle |
| NLP | Natural Language Processing, traitement automatique du langage |
| NLU | Natural Language Understanding |
| NLG | Natural Language Generation |
| LLM | Large Language Model |
| RNN | Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| SFT | Supervised Fine-Tuning |
| EOS | End Of Sequence |
| PPL | Perplexité |

## 1. Contexte et problématique

### 1.1 Problématique scientifique

La génération poétique ajoute au problème général de génération de texte une tension entre liberté créative et contraintes explicites. Un système doit produire un texte lisible et évocateur tout en respectant simultanément un sujet, un ton, une première ligne, un nombre de vers et des mots imposés.

**Problématique proposée :** comment adapter efficacement un petit modèle de langage instructionnel afin de produire des poèmes anglais cohérents et originaux, tout en mesurant séparément la qualité littéraire et le respect vérifiable de contraintes formelles ?

### 1.2 Objectifs

Les objectifs fonctionnels sont :

- accepter une consigne structurée ou libre ;
- générer un ou plusieurs candidats ;
- conserver exactement la première ligne lorsqu’elle est fournie ;
- compter les vers et détecter les mots obligatoires ;
- repérer certaines répétitions et lignes trop longues ;
- exposer les paramètres et la provenance du modèle ;
- enregistrer ou exporter un brouillon sans le présenter abusivement comme conforme.

Les objectifs scientifiques sont plus exigeants : comparer des stratégies LoRA sous protocole contrôlé, quantifier les contraintes avec un évaluateur déterministe et faire compléter l’évaluation technique par une lecture humaine aveugle.

### 1.3 Utilisateurs et valeur

Les utilisateurs cibles sont une étudiante ou un enseignant en IA, un auteur recherchant des idées, un démonstrateur de technologie générative et, à terme, une équipe éditoriale ou culturelle. La valeur ajoutée n’est pas seulement le texte produit : elle réside dans la traçabilité du modèle, les contrôles de contraintes, la comparaison de candidats et la conservation locale des poèmes.

## 2. Fondements NLP

Le **NLP** regroupe les méthodes permettant de traiter le langage. La **NLU** analyse le sens ou l’intention d’une entrée ; la **NLG** produit une sortie textuelle. Ce projet relève principalement de la NLG conditionnelle.

Un **corpus** est un ensemble de textes. Le modèle ne manipule pas directement des phrases mais des **tokens**, unités issues du tokenizer. Le tokenizer Qwen transforme une chaîne en `input_ids`, puis en vecteurs appris ou calculés par le modèle. Les tokens spéciaux délimitent notamment les messages du format conversationnel et la fin de séquence.

Un modèle autorégressif estime :

\[
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\ldots,x_{t-1}).
\]

À chaque étape, il prédit une distribution sur le prochain token. La **cross-entropy** moyenne mesure la pénalité de la vraie suite ; la perplexité est `exp(loss)` et s’interprète comme un nombre moyen de choix équivalents, avec prudence lorsque les corpus ou les tokenizers diffèrent.

Les RNN et LSTM traitent les tokens séquentiellement. Les portes d’un LSTM améliorent la conservation de l’information, mais les longues dépendances restent coûteuses. Le Transformer remplace cette récurrence par l’attention. Pour une tête d’attention :

\[
Attention(Q,K,V)=softmax(QK^T/\sqrt{d_k})V.
\]

Le modèle peut ainsi relier des mots éloignés, par exemple une contrainte déclarée au début et la réponse attendue à la fin. Qwen2.5-1.5B-Instruct est utilisé comme modèle préentraîné et instructionnel ; le fine-tuning adapte ses comportements à un format de poésie sans réentraîner tous ses paramètres.

Le **prompt engineering** structure la demande : thème, ton, première ligne, longueur et vocabulaire obligatoire. Il guide une distribution probabiliste mais ne constitue pas une contrainte logique. Un mot peut être oublié, un texte peut s’arrêter trop tôt et un nombre de vers peut être dépassé.

## 3. Données et préparation

### 3.1 Poetry Foundation

Le notebook joint rapporte un téléchargement du dataset Poetry Foundation et vérifie le fichier `/kaggle/input/poetry-foundation-poems/PoetryFoundationData.csv`. Les faits observés sont : 13 854 poèmes, colonnes `Title`, `Poem`, `Poet`, `Tags` ainsi qu’un index `Unnamed: 0`, et SHA-256 identique lors du contrôle.

Le nettoyage attendu comprend suppression ou traitement des valeurs manquantes, détection des doublons, conservation des retours à la ligne et gestion des poèmes longs. La provenance et les droits de réutilisation doivent être documentés avant toute diffusion commerciale. **À vérifier :** licence exacte du dataset, conditions des textes individuels et méthode finale de filtrage utilisée pour chaque version LoRA.

### 3.2 Données d’instruction

Le prompt maître mentionne un dataset synthétique de 416 exemples, répartis en 284 entraînement, 61 validation et 71 test, avec les colonnes `INSTRUCTION`, `RESPONSE`, `theme`, `tone`, `n_lines`, `first_line`, `required_words` et `split`. Ces nombres ne sont pas corroborés par un fichier de dataset présent à la racine ; ils sont donc conservés ici comme **À vérifier**.

Une paire instruction/réponse associe une demande structurée à un poème cible. Les risques sont la répétition de patrons, la faible diversité des thèmes, des réponses artificielles et la fuite de données si des poèmes quasi identiques apparaissent dans plusieurs splits. Le split doit être fait avant les transformations susceptibles de dupliquer le texte, puis contrôlé par empreintes et similarité.

### 3.3 Prétraitement et labels

Le pipeline conceptuel est :

`instruction → chat template Qwen → tokenisation → input_ids/attention_mask → labels → loss`.

Pour un entraînement instruction-réponse, on peut masquer les tokens du prompt par `-100` dans `labels`. La cross-entropy ignore alors ces positions et apprend principalement à produire la réponse. Une longueur maximale tronque les séquences trop longues ; cette opération doit être mesurée, car elle peut supprimer la conclusion du poème ou des contraintes.

## 4. Architectures et choix techniques

| Approche | Rôle | Atout | Limite dans ce projet |
|---|---|---|---|
| LSTM | Baseline séquentielle | Simple à expliquer | Dépendances longues et qualité limitée |
| Transfer learning | Réutilisation d’un modèle préentraîné | Moins de données et de calcul | Dépendance au modèle de base |
| Qwen2.5-1.5B-Instruct | Générateur principal | Suit déjà des instructions | Coût CPU élevé et contraintes non garanties |
| LoRA V1/V2/V3 | Adaptation légère | Peu de paramètres à stocker | Sensible aux données et au protocole |

LoRA approxime une mise à jour de poids par deux petites matrices :

\[
W'=W+\frac{\alpha}{r}BA.
\]

Le poids de base `W` reste gelé et seuls `A` et `B` sont entraînés. Les configurations locales des adaptateurs V2 et V3 indiquent `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, tâche `CAUSAL_LM` et les modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Les deux configurations ciblent le même modèle de base.

Les fichiers du projet indiquent également un libellé V3 associé à un nouveau dataset de 416 exemples. Les performances complètes V3 annoncées dans le prompt maître — 36 étapes, 10,29 minutes, validation loss 0,8672, test loss 0,9907 et PPL 2,6932 — ne sont pas retrouvées de manière suffisamment traçable dans les fichiers exploités ici : **À vérifier dans le notebook ou les logs d’entraînement originaux**.

## 5. Entraînement et fine-tuning

Le notebook de validation indique 28 tests unitaires, un code de sortie nul, mais `training_executed=False`, `generation_executed=False` et `notebook_executed=False`. Une sortie du notebook V2 joint indique également `MAX_LEN=512`, `EPOCHS=2`, `LEARNING_RATE=1e-4`, `BATCH_SIZE=1` et entraînement désactivé. Ces valeurs décrivent la configuration prévue, pas une exécution confirmée.

| Paramètre | Valeur documentée | Effet |
|---|---:|---|
| Epochs | 2 | Deux passages sur les exemples ; davantage peut surajuster |
| Learning rate | 0,0001 | Pas d’optimisation ; trop élevé, il déstabilise |
| Batch size | 1 | Faible mémoire, gradients bruités |
| Accumulation | 16 | Batch effectif approximatif de 16 |
| MAX_LEN | 512 | Limite de contexte d’entraînement annoncée |
| LoRA rank | 16 | Capacité de l’adaptation |
| Alpha | 32 | Échelle de la mise à jour |
| Dropout | 0,05 | Régularisation |
| Optimiseur | AdamW | Mise à jour adaptative avec weight decay |
| Scheduler | Cosine | Décroissance progressive du learning rate |
| Seed | 42 | Reproductibilité partielle |

Le gradient checkpointing réduit la mémoire au prix de recalculs. FP16 ou BF16 accélère généralement l’entraînement sur GPU ; la disponibilité réelle dépend du matériel. Les risques sont l’overfitting sur un petit corpus, l’underfitting si les données ne couvrent pas les contraintes et l’oubli catastrophique si l’adaptation déforme excessivement le comportement instructionnel.

## 6. Génération et contrôle des contraintes

Le code de [generer.py](./generer.py) applique le chat template, préremplit la première ligne lorsqu’elle est fournie, puis ne décode que les nouveaux tokens. La génération est échantillonnée (`do_sample=True`) avec température, `top_p`, pénalité de répétition et taille de n-gramme sans répétition. Une stopping criterion observe les frontières de vers, mais conserve les vers excédentaires afin que l’échec soit visible.

La température élevée augmente la diversité et le risque d’incohérence. `top-k` limite le nombre de candidats ; `top-p` conserve une masse de probabilité ; `max_new_tokens` borne le budget ; `repetition_penalty` et `no_repeat_ngram_size` réduisent les boucles mais peuvent dégrader la naturalité. Un seed rend une expérience reproductible dans des conditions identiques, sans rendre le modèle déterministe dans tous les environnements.

La validation de [poesie.py](./poesie.py) et les rapports associés contrôlent notamment le nombre de vers, la première ligne, les mots entiers, les vers trop longs, les groupes répétés et certains signaux heuristiques. La cohérence sémantique, le ton et l’originalité ne sont pas prouvés par ces règles : ils nécessitent une évaluation humaine ou un protocole sémantique indépendant.

## 7. Évaluation des résultats

### 7.1 Métriques

Pour une métrique binaire de conformité, on peut écrire :

\[
Taux(C)=\frac{\#\{générations\ conformes\}}{\#\{générations\ évaluées\}}.
\]

Les taux de lignes, de première ligne et de mots obligatoires sont calculés séparément. La conformité complète est leur combinaison selon le contrat fonctionnel. La loss et la perplexité mesurent l’ajustement tokenisé ; elles ne mesurent ni la beauté ni l’originalité. Une grille humaine devrait noter séparément cohérence, fluidité, imagerie, ton, conclusion et originalité, avec des critères et une procédure aveugle.

### 7.2 Pilote disponible

Le fichier `data/evaluation_pilot_20261022/metrics.csv` contient 10 cas `before` et 10 cas `after`, avec seed 42, huit vers demandés et des mots obligatoires. Le résumé calculé à partir de ce fichier est :

| Phase | Cas | Conformité complète | Nombre de vers correct | Première ligne | Mots obligatoires | Répétitions | Temps moyen |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before | 10 | 0/10 | 8/10 | 10/10 | 0/10 | 10/10 | 59,6 s |
| After | 10 | 0/10 | 7/10 | 10/10 | 0/10 | 10/10 | 48,9 s |

Ce résultat est utile pour identifier un défaut : le mécanisme de contrôle détecte bien l’absence des mots obligatoires, mais la génération ne les respecte pas dans ces cas. La première ligne est robuste, tandis que le nombre de vers varie. Le pilote ne contient pas de notes humaines renseignées ; il ne permet donc pas de conclure à une amélioration littéraire.

Les valeurs de loss V2 ou V3 ne doivent pas être comparées directement si les datasets, objectifs, masques de labels ou tokenizers diffèrent. La comparaison V1/V2 décrite dans [COMPARAISON_MODELES.md](./COMPARAISON_MODELES.md) recommande des seeds appariées, paramètres fixes, candidats conservés et absence de gagnant littéraire proclamé sans lecture humaine.

## 8. Application Streamlit

### 8.1 Parcours fonctionnel

```text
Utilisateur
   ↓ thème, ton, première ligne, contraintes, paramètres
Formulaire Streamlit
   ↓ validation des entrées et choix V1/V2/V3
Chargement Qwen + adaptateur sous verrou/cache
   ↓ génération d’un ou plusieurs candidats
Évaluateur technique
   ↓ conformité, répétitions, provenance, empreintes
Sélection technique du meilleur candidat
   ↓ sans réécriture silencieuse
Affichage / export / bibliothèque locale
```

L’application expose trois pages : accueil, génération et bibliothèque. Elle propose le choix de l’adaptateur, la génération de candidats, la comparaison V1/V2 (et V3 si sélectionné selon le code), le téléchargement et l’enregistrement. Les nouveaux poèmes conservent un champ de provenance comprenant version, modèle de base, chemin et SHA-256 lorsque disponible.

### 8.2 Architecture Streamlit observée

Le point d’entrée est [app.py](./app.py), qui utilise `st.navigation` et `st.Page`, initialise `st.session_state` et présente la navigation dans la barre latérale. Le cache de ressources conserve la base et un verrou partagé ; les poids restent en mode inférence. Le formulaire regroupe les champs et ne déclenche la génération qu’à la soumission, ce qui est adapté à une opération coûteuse.

La stratégie est cohérente avec les bonnes pratiques Streamlit : `st.cache_resource` pour un modèle partagé, état de session pour les brouillons propres à l’utilisateur, formulaire pour regrouper les entrées et `st.navigation` pour les pages. En production, il faudrait tester la concurrence, la limite mémoire et la durée de vie du cache, et envisager un fragment ou une file d’inférence si l’application doit gérer plusieurs utilisateurs.

## 9. Qualité logicielle et reproductibilité

Les tests présents couvrent la validation des poèmes, les adaptateurs, la comparaison, la bibliothèque et des scénarios simulés. Le fichier de validation V2 indique 28 tests et un code de sortie 0 ; les modèles simulés évitent de charger Qwen et de lancer une génération coûteuse. Cela établit la robustesse logique, pas la qualité de génération réelle.

La stratégie recommandée avant production est :

1. tests unitaires purs sur parsing, métriques et validation ;
2. tests d’intégration sur chargement réel des adaptateurs et tokenizer ;
3. tests d’application headless avec `streamlit.testing.v1.AppTest` ;
4. test de non-régression sur un jeu figé de prompts et seeds ;
5. vérification SHA-256 et journalisation de la provenance ;
6. benchmark CPU/GPU avec temps de chargement, génération et mémoire ;
7. évaluation humaine aveugle séparant V1, V2, V3 et baseline.

Les empreintes observées incluent notamment : adaptateur V2 `e214f523…e096a5`, configuration V2 `5594eef8…0ac788`, adaptateur V3 `6f113d8e…ea97c18` et configuration V3 `7aea1214…6f675c`. Les valeurs complètes sont conservées dans les fichiers du projet ; les ellipses sont utilisées ici pour éviter de transformer le rapport en inventaire de poids.

## 10. Déploiement et exploitation

### 10.1 Local

Le lancement documenté est :

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
```

Le fichier [requirements.txt](./requirements.txt) et l’environnement `.venv` fournissent la base de reproductibilité. En CPU, le pilote montre des temps de génération de plusieurs dizaines de secondes ; un GPU est donc souhaitable pour une expérience interactive. **À vérifier :** RAM minimale, VRAM, consommation énergétique et nombre maximal d’utilisateurs simultanés.

### 10.2 Cloud et production

Un déploiement cloud nécessite un stockage fiable des poids, une construction reproductible, une politique de cache, des timeouts et une protection des entrées. Les options sont une instance GPU, un service d’inférence séparé ou une exécution CPU limitée à la démonstration. L’application actuelle sauvegarde localement les poèmes : cette solution ne convient pas directement à plusieurs utilisateurs ou à une exigence de confidentialité forte.

Les contrôles à ajouter sont authentification, quotas, limitation de taille des prompts, journalisation sans données sensibles, contrôle de licence des textes, supervision des erreurs et versionnement des adaptateurs. Les résultats doivent afficher clairement qu’un poème peut être non conforme ou issu d’un modèle expérimental.

## 11. Valorisation commerciale

La proposition de valeur est un atelier de création poétique pilotable, traçable et démontrable, destiné à des expériences éducatives, culturelles ou éditoriales. Une démonstration commerciale peut montrer la même consigne avec plusieurs températures, puis comparer les adaptateurs en affichant les contraintes détectées plutôt qu’un score littéraire artificiel.

Les clients potentiels sont établissements d’enseignement, ateliers d’écriture, médiateurs culturels, studios créatifs et équipes de prototypage IA. Les modèles économiques possibles sont une licence d’installation, une offre de démonstration, un abonnement avec quotas ou un service d’intégration personnalisé.

Les risques principaux sont les droits d’auteur, la similarité involontaire avec des textes existants, les sorties incohérentes, le coût de calcul, les licences de Qwen/PEFT/datasets et l’absence de preuve de supériorité entre versions. La commercialisation ne doit être envisagée qu’après audit de licences, benchmark et évaluation humaine. Le prototype est une base de produit, non une preuve de préparation industrielle.

## 12. Limites, feuille de route et conclusion

Les limites actuelles sont la petite quantité de données d’instruction non confirmée, l’échec systématique des mots obligatoires dans le pilote, la variabilité du nombre de vers, le coût CPU et l’absence d’évaluation humaine renseignée. Les heuristiques de qualité ne doivent pas être confondues avec une mesure de créativité.

### Feuille de route

**Court terme :** confirmer la provenance et les splits, ajouter des tests d’intégrité de dataset, corriger le contrôle des mots obligatoires par génération de candidats ou reranking, et compléter l’évaluation humaine.

**Moyen terme :** enrichir les exemples avec des thèmes et tons équilibrés, comparer V1/V2/V3 à paramètres identiques, tester plusieurs seeds et publier un rapport de confiance plutôt qu’un unique score.

**Long terme :** séparer le service d’inférence, ajouter une authentification, mesurer coût et latence sur GPU, auditer les licences et proposer une API ou une offre multi-utilisateur.

En conclusion, le projet mobilise correctement les notions centrales de NLP génératif : tokenizer, Transformer, instruction tuning, LoRA, décodage probabiliste, validation déterministe et interface Streamlit. Sa contribution la plus défendable aujourd’hui est une chaîne expérimentale traçable qui expose ses échecs. Pour devenir une solution commercialisable, elle doit maintenant démontrer la qualité littéraire sur un protocole indépendant et résoudre la conformité des contraintes dans des conditions reproductibles.

## Annexe A — Glossaire

**Adaptateur :** paramètres additionnels appliqués à un modèle de base.  
**Attention :** mécanisme de pondération des positions d’une séquence.  
**Chat template :** formatage des messages selon le modèle conversationnel.  
**Conformité :** satisfaction simultanée des règles techniques définies.  
**Fine-tuning :** adaptation d’un modèle préentraîné à une tâche ou un domaine.  
**LoRA :** adaptation basse-rang ajoutant peu de paramètres entraînables.  
**Perplexité :** exponentielle de la loss moyenne, dépendante de la tokenisation.  
**Reranking :** sélection d’un candidat parmi plusieurs sorties.  
**Seed :** état initial pseudo-aléatoire utilisé pour reproduire une expérience.  
**Token :** unité de texte manipulée par le tokenizer.

## Annexe B — Sources et éléments à vérifier

### Fichiers examinés

- [app.py](./app.py), [generer.py](./generer.py), [poesie.py](./poesie.py), [adaptateurs.py](./adaptateurs.py) ;
- [COMPARAISON_MODELES.md](./COMPARAISON_MODELES.md), [ADAPTATEURS_V1_V2.md](./ADAPTATEURS_V1_V2.md) ;
- [data/evaluation_pilot_20261022/metrics.csv](./data/evaluation_pilot_20261022/metrics.csv), `common_evaluation.json`, `verification.json` ;
- [lora_v2/validation_report.json](./lora_v2/validation_report.json) ;
- notebooks locaux et notebook joint en lecture seule.

### À confirmer avant soutenance ou diffusion

1. licence et provenance détaillées de Poetry Foundation ;
2. existence exacte, contenu et licence du dataset synthétique 416 ;
3. logs d’entraînement complets et résultats V3 annoncés ;
4. matériel, durée, mémoire et énergie de chaque expérience ;
5. évaluation humaine aveugle avec annotateurs et accord inter-annotateurs ;
6. licence et conditions d’utilisation de Qwen2.5-1.5B-Instruct ;
7. exigences de sécurité, confidentialité et rétention pour un usage multi-utilisateur.

### Références techniques à vérifier

Les références bibliographiques formelles — article LoRA, documentation Qwen, documentation PEFT, documentation Hugging Face Transformers et documentation Streamlit — doivent être ajoutées dans le format institutionnel demandé par l’établissement, avec leurs versions et dates de consultation. Elles ne sont pas inventoriées comme résultats expérimentaux du projet.
