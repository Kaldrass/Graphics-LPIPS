# Contexte projet Graphics-LPIPS

## Objectif

Ce dépôt implémente **Graphics-LPIPS**, une adaptation de LPIPS à l'évaluation perceptuelle de qualité d'objets 3D texturés. L'idée générale est :

- on rend un objet 3D selon un ou plusieurs points de vue ;
- on découpe les vues en patches 2D ;
- on compare patch de référence et patch dégradé avec un backbone CNN de type LPIPS ;
- on agrège les distances patch par patch pour obtenir un score par vue, puis un score global par stimulus.

Le score prédit est utilisé comme approximation d'un score perceptuel de type MOS après calibration/régression.

## Scripts principaux

### `GraphicsLpips_csvFile.py`

Script historique d'évaluation :

- lit un CSV décrivant, pour chaque stimulus, le modèle de référence, le stimulus dégradé, le MOS et le nombre de patches ;
- charge des patches déjà exportés sur disque dans `./dataset/References_patches_withVP_threth0.6` et `./dataset/PlaylistsStimuli_patches_withVP_threth0.6` ;
- évalue chaque paire de patches via `lpips.LPIPS(net='alex', model_path=...)` ;
- tronque chaque distance à `1` si nécessaire ;
- moyenne les distances pour produire un score Graphics-LPIPS par stimulus ;
- écrit les scores dans un CSV de sortie ;
- ajuste ensuite une régression logistique avec `statsmodels.GLM(..., family=Binomial())` pour corréler Graphics-LPIPS et MOS ;
- calcule Pearson et Spearman, puis sauve un graphique et un résumé CSV.

Limites importantes :

- chemins codés en dur vers l'ancien format patchifié ;
- dépend d'une base déjà patchifiée sur disque ;
- peu flexible pour les nouvelles sorties d'expérience ;
- suppose souvent GPU actif (`--use_gpu` vaut `True` par défaut).

### `Light_GraphicsLPIPS_csv.py`

Version plus récente et plus utile pour les workflows actuels :

- ne stocke plus toute la base patchifiée ;
- lit les vues de référence et dégradées depuis des dossiers `views/` ;
- relit la liste des coordonnées de patches depuis le CSV patchlist de la référence ;
- recrée les patches en mémoire via OpenCV ;
- batchifie les patches par vue avant appel réseau, ce qui réduit les appels unitaires ;
- supporte les folds (`--use_folds`) ;
- écrit les résultats par objet dans `out/.../_METRIC_RESULTS_TESTSET_/.../GLPIPS_results_testset.csv`.

Ce script est aujourd'hui le pont entre les checkpoints entraînés et les sorties d'évaluation utilisées ensuite par `correlation_VP.py`.

### `correlation_VP.py`

Post-traitement statistique :

- relit les `GLPIPS_results_testset.csv` générés par objet ;
- normalise les MOS ;
- moyenne les scores LPIPS sur les viewpoints ;
- calcule corrélations et régressions logistiques ;
- gère les cas mono-config et k-fold ;
- génère des CSV récapitulatifs et des figures dans `out/...`.

### `train.py`

Entraînement principal :

- construit le `Trainer` dans `lpips/trainer.py` ;
- charge les données via `data/data_loader.py` ;
- entraîne par epochs avec agrégation au niveau stimulus ;
- sauvegarde les checkpoints dans `checkpoints/<name>/` ;
- peut aussi gérer les folds.

## Chargement des données

La chaîne de chargement pour l'entraînement est :

1. `train.py`
2. `data/data_loader.py`
3. `data/custom_dataset_data_loader.py`
4. `data/dataset/twoafc_dataset.py`

Points importants dans `TwoAFCDataset` :

- malgré le nom `2afc`, le dataset a été détourné pour un protocole de qualité de type DSIS/MOS ;
- chaque ligne du CSV d'entrée correspond à un stimulus ;
- le dataset lit le fichier de patchlist du modèle de référence ;
- il échantillonne `maxNbPatches` patches par stimulus ;
- il retourne :
  - `ref` : patch référence normalisé dans `[-1, 1]`
  - `p0` : patch dégradé normalisé
  - `judge` : ici le plus souvent le MOS
  - `stimuli_id` : identifiant entier pour regrouper les patches d'un même stimulus.

Le `Trainer` agrège ensuite les sorties patch-wise par `stimuli_id` pour calculer une prédiction par stimulus et comparer cette moyenne au MOS cible.

## Cœur modèle

Le cœur réseau repose sur le package local `lpips/` :

- `lpips/lpips.py` : définition du modèle LPIPS ;
- `lpips/pretrained_networks.py` : backbones ;
- `lpips/trainer.py` : logique d'entraînement, loss, sauvegarde, test set.

Le choix le plus courant dans ce dépôt est :

- backbone `alex` ;
- calibration linéaire LPIPS ;
- checkpoint chargé depuis `checkpoints/<model>/latest_net_.pth`.

## Flux typiques

### Évaluer un checkpoint sur un dataset rendu

1. Générer ou disposer des vues `Source/.../views` et `Distorted/.../views`, ainsi que des patchlists côté référence.
2. Lancer `Light_GraphicsLPIPS_csv.py` avec le bon modèle, dataset, nombre de vues et chemins CSV.
3. Lancer `correlation_VP.py`.
4. Lire les résultats dans `out/<database>/<render_method>/<view_method>/<model>/<n>VP/`.

### Entraîner un nouveau modèle

1. Préparer les CSV train/test et l'arborescence des vues.
2. Lancer `train.py` avec `--datasets`, `--testcsv`, `--src_root`, `--root_refPatches`, `--root_distPatches`, `--target`.
3. Récupérer les checkpoints dans `checkpoints/<name>/`.
4. Évaluer ensuite avec `Light_GraphicsLPIPS_csv.py`.

## Arborescences importantes

### Racine du dépôt

- `checkpoints/` : poids des modèles entraînés.
- `dataset/` : CSV de train/test, folds, listes de stimuli.
- `out/` : sorties d'évaluation et de corrélation.
- `data/` : loaders PyTorch.
- `lpips/` : implémentation du modèle et du trainer.

### Arborescence attendue côté rendu

Le code récent suppose en pratique des structures du type :

- `.../Source/<n>VP/<ref_obj>/views/view_1.png`
- `.../Source/<n>VP/<ref_obj>/patchs/<ref_obj>_patchlist.csv`
- `.../Distorted/<n>VP/<dist_obj>/views/view_1.png`

Attention : le dossier des patchlists est nommé `patchs` dans `twoafc_dataset.py`, alors que certains commentaires ou outils parlent de `patches`. C'est un point de vigilance récurrent.

## Sorties importantes

- `GLPIPS_results_testset.csv` : scores par objet dégradé, souvent une colonne MOS suivie d'une valeur par viewpoint.
- `correlation_folds_stats.csv` : synthèse Pearson/Spearman par fold.
- `correlation_summary_kfolds.csv` : résumé global au niveau expérience.

## Hypothèses et fragilités connues

- Beaucoup de chemins Windows sont codés en dur, notamment dans `Light_GraphicsLPIPS_csv.py` et `correlation_VP.py`.
- Le booléen `--use_gpu` est activé par défaut dans plusieurs scripts ; certains appels utilisent `cuda()` sans fallback explicite.
- Le projet mélange ancien pipeline "patches enregistrés sur disque" et pipeline récent "patches reconstruits en mémoire".
- Les noms de variables historiques (`judge`, `2afc`) ne reflètent pas toujours l'usage actuel basé sur le MOS.
- Certains fichiers contiennent des commentaires ou chaînes avec encodage imparfait, sans impact direct mais à garder en tête lors des éditions.

## Recommandations pour futurs prompts

- Pour toute évaluation récente, partir de `Light_GraphicsLPIPS_csv.py` et `correlation_VP.py`.
- Pour comprendre la logique de training, lire `train.py` puis `data/dataset/twoafc_dataset.py` puis `lpips/trainer.py`.
- Si un bug touche les scores finaux, vérifier en priorité :
  - la correspondance des noms d'objets entre CSV MOS, test list et dossiers ;
  - la structure `views/` et `patchs/` ;
  - la cohérence folds/modèle/checkpoint ;
  - la normalisation MOS dans `correlation_VP.py`.
