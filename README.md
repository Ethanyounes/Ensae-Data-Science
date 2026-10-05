# Prédire les short squeezes sur les perpétuels crypto

Projet de Data Science, ENSAE 2A.

**Question** : peut-on prédire un short squeeze 24 heures à l'avance à partir du positionnement des traders sur les perpétuels crypto (Binance USD-M) ?

- **H1, shorts informés** : une accumulation de shorts annonce une baisse, et le positionnement ne prédit pas de squeeze.
- **H2, shorts fragiles** : des shorts accumulés avec du levier rendent le marché vulnérable à un rachat forcé.

La réponse comporte deux volets :
- **statistique** : le positionnement bat-il une base naïve, et un modèle qui n'utilise que les prix ?
- **économique** : ce gain reste-t-il rentable une fois les frais payés ?

## Installation et reproduction

On utilise [uv](https://docs.astral.sh/uv/) avec Python 3.12.

```bash
uv sync                                    # crée .venv avec les versions de uv.lock
uv run python scripts/verif_donnees.py     # 1re vérification des données Binance
uv run jupyter lab                         # ouvrir rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, généré par `uv export --no-hashes --no-dev > requirements.txt`.

Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Le dossier se change avec la variable d'environnement `DATA_DIR`. Sur le SSPCloud, les données brutes sont copiées dans le bucket S3 (`S3_BUCKET`), avec la même arborescence.

## Structure

```
src/
  config.py      chemins (raw_path), univers de cryptos, période, URL      ← tout le monde importe d'ici
  collecte.py    Binance (checksum + parquet), Bybit, Fear & Greed, S3
  panel.py       agrégation horaire, nettoyage                              (A)
  cible.py       définition des squeezes                                    (B)
  variables.py   construction des variables                                 (C)
  modeles.py     base naïve, logit, LightGBM, validation temporelle
  strategie.py   backtest avec frais et funding
scripts/
  verif_donnees.py   couverture et fréquence des fichiers metrics et fundingRate
data/
  raw/binance/{metrics,fundingRate,klines,premiumIndexKlines,spot_klines}/<SYMBOLE>/*.parquet
  raw/bybit/{funding,open_interest}/<SYMBOLE>/
  raw/fng/
  interim/       panel horaire
  processed/     panel + cible + variables
rapport.ipynb    récit, graphiques, résultats
```

Exemple :

```python
from src.collecte import download_binance, load_raw
download_binance("BTCUSDT", "metrics", "2023-01-01", "2023-01-31")   # 5 min, fichiers quotidiens
download_binance("BTCUSDT", "klines", "2023-01", "2023-12")          # 1 h, fichiers mensuels
df = load_raw("binance", "metrics", "BTCUSDT")
```

## Les données

| Source | Contenu | Fréquence | Accès |
|---|---|---|---|
| Binance `metrics` | open interest, ratios long/short (gros traders, tous les comptes), ratio taker acheteurs/vendeurs | 5 min, fichiers **quotidiens** | data.binance.vision + `.CHECKSUM` |
| Binance `fundingRate` | funding et intervalle (8/4/1 h) | par paiement, fichiers mensuels | idem |
| Binance `klines` 1h | OHLCV du perpétuel | 1 h | idem |
| Binance `premiumIndexKlines` 1h | basis perpétuel/indice | 1 h | idem |
| Binance spot `klines` 1h | OHLCV spot | 1 h | idem |
| Bybit API v5 | funding, open interest | 1 h | REST paginé |
| alternative.me | Fear & Greed | quotidien | REST |

### Constats de la première vérification (5 octobre 2026)

- Les `metrics` sont bien à 5 minutes, soit 288 lignes par jour. Ils commencent le 2020-09-01 pour BTC et le 2021-12-01 pour SOL.
- **Les ratios long/short manquent sur une grande partie de 2022.** Sur BTCUSDT et SOLUSDT, les ratios gros traders et tous comptes sont vides du 2022-01-01 au 2022-12-16, sauf entre le 28 mai et le 20 juin environ. Le ratio taker est vide avant le 2022-05-14. L'open interest, lui, est complet. Conséquence : il faut soit démarrer l'échantillon fin 2022, soit garder 2022 pour les seules variables d'open interest et de funding. Ce choix est à documenter dans le rapport.
- Le funding est versé toutes les 8 h sur BTC et SOL pour les mois testés. La colonne `funding_interval_hours` sert à ramener chaque taux en horaire.
- Les fichiers spot récents (2025) ont des timestamps en microsecondes. `collecte._to_utc` détecte l'unité automatiquement.

## Répartition

| Période | A | B | C |
|---|---|---|---|
| Début octobre | dépôt, collecte Binance | Bybit et Fear & Greed | revue de littérature |
| Mi à fin octobre | panel horaire | cible | variables |
| Début novembre | analyse descriptive (en commun) | | |
| Mi-novembre | logit et base naïve | gradient boosting | stratégie et frais |
| Fin novembre | robustesse | rédaction | reproductibilité |

Méthode de travail : une branche et une pull request par tâche.
