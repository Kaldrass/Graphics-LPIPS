# AGENTS

## But du dépôt

`Graphics-LPIPS` implémente une métrique perceptuelle pour objets 3D texturés, dérivée de LPIPS. Le dépôt contient :

- un pipeline d'entraînement PyTorch ;
- un pipeline d'évaluation sur vues multi-view ;
- un pipeline de corrélation entre scores prédits et MOS.

## Fichiers à lire en priorité

1. `docs/project_context.md`
2. `README.md`
3. `GraphicsLpips_csvFile.py`
4. `Light_GraphicsLPIPS_csv.py`
5. `correlation_VP.py`
6. `train.py`
7. `data/dataset/twoafc_dataset.py`
8. `lpips/trainer.py`

## Carte rapide du code

- `GraphicsLpips_csvFile.py` : pipeline historique sur patches déjà exportés.
- `Light_GraphicsLPIPS_csv.py` : pipeline d'évaluation actuel, patches reconstruits en mémoire.
- `correlation_VP.py` : métriques de corrélation, plots et résumés CSV.
- `train.py` : entraînement des modèles.
- `data/` : chargement dataset/patches.
- `lpips/` : modèle, backbone et trainer.
- `out/` : résultats d'expériences.
- `checkpoints/` : checkpoints des modèles.

## Hypothèses de travail utiles

- En pratique, le script central pour les expériences récentes est `Light_GraphicsLPIPS_csv.py`, pas `GraphicsLpips_csvFile.py`.
- Le nom `2afc` dans les loaders est historique ; l'usage réel est centré sur DSIS/MOS.
- Plusieurs chemins sont spécifiques à Windows et parfois absolus.
- Le dossier des patchlists peut s'appeler `patchs` dans le code récent.
- Les résultats finaux se lisent généralement dans `out/.../_METRIC_RESULTS_TESTSET_/`.

## Quand un futur prompt demande du contexte

- Résumer d'abord l'architecture avec `docs/project_context.md`.
- Si la demande porte sur l'évaluation, inspecter d'abord `Light_GraphicsLPIPS_csv.py`, `find_dis_ref.py`, `correlation_VP.py`.
- Si la demande porte sur l'entraînement, inspecter d'abord `train.py`, `data/dataset/twoafc_dataset.py`, `lpips/trainer.py`.
- Si la demande cite `GraphicsLpips_csvFile.py`, préciser qu'il s'agit du script historique sur base patchifiée.

## Vigilances

- Ne pas supposer que les chemins sont portables.
- Vérifier systématiquement la convention de nommage des objets entre CSV et dossiers.
- Vérifier si l'expérience utilise des folds avant de toucher aux chemins de checkpoints ou de sorties.
- En cas d'écart de résultats, contrôler la normalisation MOS et l'agrégation multi-view.
