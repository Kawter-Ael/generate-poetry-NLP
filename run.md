# Commandes d'installation et d'exécution

Commandes pour **Windows PowerShell**, depuis le dossier du projet. Python **3.11** est recommandé. Internet est nécessaire pour installer les dépendances et télécharger Qwen au premier chargement.

## Récupérer le projet depuis GitHub

Installer Git, Git LFS et GitHub CLI, puis :

```powershell
gh repo clone Kawter-Ael/generate-poetry-NLP
cd generate-poetry-NLP
git lfs install
git lfs pull
```

Si le projet est déjà sur votre ordinateur, ouvrir simplement son dossier dans PowerShell.

## Lancement recommandé

```powershell
powershell -ExecutionPolicy Bypass -File .\lancer.ps1
```

Le script utilise le Python de `.venv`. Il tente de créer l'environnement avec Python 3.11 s'il est absent et installe les dépendances si les principaux modules sont absents.

Ouvrir [http://localhost:8501](http://localhost:8501). Pour arrêter le serveur, appuyer sur **Ctrl+C** dans le terminal.

## Installation manuelle

Pour une première installation :

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si `.venv` existe déjà et fonctionne, conserver cet environnement. Son activation n'est pas nécessaire avec ces commandes.

Pour les lancements suivants :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Utiliser l'application

1. Ouvrir **Générer un poème**.
2. Choisir V1, V2 ou V3 dans **Modèle de poésie**.
3. Saisir éventuellement un thème et une première ligne en anglais.
4. Choisir le ton, le nombre de vers et les mots à inclure.
5. Cliquer sur **Générer le poème** et attendre le résultat.

Pour un premier essai sur CPU, choisir 4 vers et décocher les tentatives supplémentaires dans les paramètres avancés.

## Génération dans le terminal

Exécuter une commande selon la version souhaitée :

```powershell
.\.venv\Scripts\python.exe generer.py --adapter v1 --theme "the sea at sunset"
```

```powershell
.\.venv\Scripts\python.exe generer.py --adapter v2 --theme "the sea at sunset"
```

```powershell
.\.venv\Scripts\python.exe generer.py --adapter v3 --theme "the sea at sunset"
```

Ces commandes affichent le poème sans l'ajouter à la bibliothèque.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Tests ciblés :

```powershell
.\.venv\Scripts\python.exe -m unittest test_adaptateurs test_v3 test_options_streamlit -v
```

Les tests utilisent notamment des modèles simulés. Ils ne remplacent pas un essai avec les vrais poids.

## Autre port

```powershell
powershell -ExecutionPolicy Bypass -File .\lancer.ps1 --server.port 8502
```

Ouvrir [http://localhost:8502](http://localhost:8502).

## Dépannage

| Problème | Solution |
| --- | --- |
| `python` ouvre le Microsoft Store | Utiliser le lanceur ou `.\.venv\Scripts\python.exe`. |
| `py` est introuvable | Installer Python 3.11 avec le lanceur Python Windows puis rouvrir PowerShell. |
| Module absent | Exécuter `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`. |
| Port occupé | Choisir le port 8502. |
| Adaptateur absent ou invalide | Exécuter `git lfs pull` et vérifier ses fichiers de configuration et de poids. |
| Chargement initial long | Attendre le téléchargement de Qwen et vérifier la connexion Internet. |
| Génération lente | Réduire le nombre de vers, désactiver les tentatives supplémentaires et les comparaisons. |
| Poids modifiés pendant l'exécution | Arrêter et relancer Streamlit pour vider son cache. |

La bibliothèque locale est dans `data/poemes_enregistres.json`. Conserver ce fichier pour retrouver vos poèmes.
