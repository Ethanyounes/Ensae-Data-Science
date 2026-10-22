# CLAUDE.md

Projet ENSAE 2A (groupe de 3, rendu début décembre 2026) : prédire un short squeeze à 24 h sur les perpétuels USDT Binance (25 cryptos, 2022-2026) à partir du positionnement (funding, OI, ratios long/short, ratio taker). Hypothèses : H1 (shorts informés) contre H2 (shorts fragiles). Réponse statistique (PR-AUC face à une base naïve et à un modèle « prix seuls ») et réponse économique (backtest après frais et funding).

La note porte sur les données, l'analyse descriptive, la démarche et la reproductibilité. Le modèle n'est pas noté pour lui-même. `rapport.ipynb` doit tourner sans erreur depuis un environnement vierge.

## Conventions
- Environnement uv avec Python 3.12. Tout se lance par `uv run ...`. Après `uv add`, régénérer `requirements.txt` avec `uv export --no-hashes --no-dev > requirements.txt`.
- pyarrow est plafonné à <25 : Smart App Control bloque la DLL `_parquet` de pyarrow 25 sur le PC Windows d'Ethan.
- Chemins : toujours passer par `src/config.py` (`raw_path`, `INTERIM_DIR`, `PROCESSED_DIR`). `data/` n'est jamais commité.
- Données brutes : parquet, un fichier par archive Binance, timestamps en UTC.
- Pas de fuite d'information : la cible n'utilise que ce qui se passe après t, les variables que ce qui précède t (fenêtres glissantes passées). Validation temporelle avec un trou de 24 h. Jamais d'accuracy.
- Code et commentaires en français. Une branche et une PR par tâche.

## Cible (valeurs de base)
- **C1** : max High(t, t+24h] / Close_{t-1} − 1 ≥ max(3·σ_30j·√24, 5 %).
- **C2** : OI au pic ≤ (1 − 10 %)·OI_{t-1}.
- **C3** : ratio taker moyen entre t et le pic > 1.
- **Variantes** : k 2.5/4, plancher 3/8 %, δ 5/15 %, horizon 12/48 h.

## Consignes d'évaluation (https://pythonds.linogaliana.fr/content/annexes/evaluation.html)
- Barème ENSAE sur 20 : collecte et nettoyage 4, analyse descriptive 4, démarche scientifique et reproductibilité 5, propreté du code 2, soutenance 5. La modélisation n'est pas notée en tant que telle.
- Si une cellule du notebook plante, la moyenne est perdue. Les données doivent être accessibles depuis une URL publique ou le S3 du SSPCloud, jamais dans Git. Le dépôt GitHub doit être public.
- Le notebook appelle des fonctions de `src/` : pas de cellules copiées-collées. Chaque statistique et chaque graphique sont commentés et reliés à la question, avec des axes et des légendes lisibles.
- Collecte : plusieurs sources (API, scraping ou fichiers bruts enrichis), avec un nettoyage explicite (valeurs manquantes, formats, renommage, types). Les données de compétitions Kaggle et l'API Twitter/X sont interdites.
- IA autorisée comme assistance, mais pas de « vibe coding ». Le groupe doit comprendre et retravailler tout le code, et l'usage abusif est sanctionné.
- Commits fréquents et PR. Éviter d'éditer le même notebook à plusieurs sur la même branche.

## Points de données connus
Sur BTC, seuls les ratios des gros traders (`ls_top_*`) manquent sur la majeure partie de 2022 (complets à partir du 2022-12-15, sauf un peu de mai-juin). Le ratio global est disponible dès février 2022, le ratio taker dès mi-mai 2022, l'open interest est complet. Voir le README.
