# CLAUDE.md

Projet ENSAE 2A (groupe de 3, rendu début décembre 2026) : prédire un short squeeze à 24 h sur les perpétuels USDT Binance (10 cryptos pour l'instant, 2023-01-01 au 2026-09-30) à partir du positionnement (funding, OI, ratios long/short, ratio taker). Hypothèses : H1 (shorts informés) contre H2 (shorts fragiles). Réponse par l'analyse descriptive (étapes 1 à 5 de `rapport.ipynb`). Le modèle de prévision (PR-AUC face à une base naïve et à un modèle « prix seuls ») et le backtest prévus au départ sont abandonnés : non demandés pour l'ENSAE.

La note porte sur les données, l'analyse descriptive, la démarche et la reproductibilité. La modélisation ne fait pas partie du barème ENSAE. `rapport.ipynb` doit tourner sans erreur depuis un environnement vierge.

## Conventions
- Environnement uv avec Python 3.12. Tout se lance par `uv run ...`. Après `uv add`, régénérer `requirements.txt` avec `uv export --no-hashes --no-dev > requirements.txt`.
- pyarrow est plafonné à <25 : Smart App Control bloque la DLL `_parquet` de pyarrow 25 sur le PC Windows d'Ethan. Il bloque parfois aussi `jupyter.exe` : exécuter le notebook avec `uv run python -m nbconvert --to notebook --execute --inplace rapport.ipynb`.
- Chemins : toujours passer par `src/config.py` (`raw_path`, `INTERIM_DIR`). `data/` n'est jamais commité.
- Données brutes : parquet, un fichier par archive Binance, timestamps en UTC.
- Pas de fuite d'information : la cible n'utilise que ce qui se passe après t, les variables que ce qui précède t (fenêtres glissantes passées). Validation temporelle avec un trou de 24 h. Jamais d'accuracy.
- Code et commentaires en français. Une branche et une PR par tâche.
- Style : écrire un code simple, qui se comprend en le lisant, comme l'écrirait un étudiant soigneux, plutôt qu'un code parfait. Des noms explicites plutôt que des commentaires ; peu de commentaires, seulement pour expliquer un choix qui ne se voit pas dans le code. Docstrings courtes (une ou deux lignes). Pas de vérifications défensives ni d'abstractions qui ne servent pas le projet.Dupliquer le moins possible. 


## Univers
- `SYMBOLS` (10) : les perpétuels USDT au plus gros open interest en décembre 2022, cotés avant cette date, encore cotés sous le même nom en septembre 2026 et avec une paire spot du même nom (`scripts/choisir_cryptos.py`). MATIC est exclu : renommé POL en 2024.
- `SYMBOLS_DESCRIPTIF` (BTC, ETH, SOL) : étapes 1 à 3 du rapport. Les étapes 4 et 5 utilisent les 10. Bybit sert de contrôle de fiabilité de Binance sur les 10.

## Cible
- **C1** : max High(t, t+24h] / Close_{t-1} − 1 ≥ max(3·σ_30j·√24, 5 %).
- **C2** : OI au pic ≤ (1 − δ)·OI_{t-1}. Valeur de base δ = 10 % (premier calcul, étape 3). À partir de l'étape 4 : δ = 1,5·σ_OI_30j (écart-type des variations d'OI sur 24 h des 30 derniers jours, `delta_sigma=1.5`), avec δ = 1,2·σ_OI_30j en contrôle.
- **C3** : ratio taker moyen entre t et le pic > 1.
- **Variantes** : k 2.5/4, plancher 3/8 %, δ 5/15 % ou 1,2/1,5 σ, horizon 12/48 h.

## Consignes d'évaluation (https://pythonds.linogaliana.fr/content/annexes/evaluation.html)
- Barème ENSAE sur 20 : collecte et nettoyage 4, analyse descriptive 4, démarche scientifique et reproductibilité 5, propreté du code 2, soutenance 5. La modélisation n'est pas notée en tant que telle.
- Si une cellule du notebook plante, la moyenne est perdue. Les données doivent être accessibles depuis une URL publique ou le S3 du SSPCloud, jamais dans Git. Le dépôt GitHub doit être public.
- Le notebook appelle des fonctions de `src/` : pas de cellules copiées-collées. Chaque statistique et chaque graphique sont commentés et reliés à la question, avec des axes et des légendes lisibles.
- Collecte : plusieurs sources (API, scraping ou fichiers bruts enrichis), avec un nettoyage explicite (valeurs manquantes, formats, renommage, types). Les données de compétitions Kaggle et l'API Twitter/X sont interdites.
- IA autorisée comme assistance, mais pas de « vibe coding ». Le groupe doit comprendre et retravailler tout le code, et l'usage abusif est sanctionné.
- Commits fréquents et PR. Éviter d'éditer le même notebook à plusieurs sur la même branche.

## Points de données connus
Sur BTC, seuls les ratios des gros traders (`ls_top_*`) manquent sur la majeure partie de 2022 (complets à partir du 2022-12-15, sauf un peu de mai-juin). Le ratio global est disponible dès février 2022, le ratio taker dès mi-mai 2022, l'open interest est complet. Voir le README.
