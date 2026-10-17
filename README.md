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
uv run python -m scripts.telecharger_binance   # collecte Binance (relançable)
uv run python -m scripts.telecharger_bybit     # collecte Bybit (moins de 5 min)
uv run jupyter lab                         # ouvrir rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, généré par `uv export --no-hashes --no-dev > requirements.txt`.

Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Le dossier se change avec la variable d'environnement `DATA_DIR`. Sur le SSPCloud, les données brutes sont copiées dans le bucket S3 (`S3_BUCKET`), avec la même arborescence.

## Structure

```
src/
  config.py      chemins (raw_path), univers de cryptos, période, URL      ← tout le monde importe d'ici
  collecte_binance.py  archives Binance : téléchargement, checksum, parquet
  collecte_bybit.py    API Bybit v5 : funding, open interest, bougies
  -- à venir --
  panel.py       agrégation horaire, nettoyage                              (A)
  cible.py       définition des squeezes                                    (B)
  variables.py   construction des variables                                 (C)
  modeles.py     base naïve, logit, LightGBM, validation temporelle
  strategie.py   backtest avec frais et funding
scripts/
  telecharger_binance.py   télécharge toutes les archives Binance (SYMBOLS × types)
  telecharger_bybit.py     télécharge toutes les séries Bybit
data/
  raw/binance/{metrics,fundingRate,klines,premiumIndexKlines,spot_klines}/<SYMBOLE>/*.parquet
  raw/bybit/{funding,open_interest,klines}/<SYMBOLE>/<SYMBOLE>-<série>.parquet
  raw/fng/
  interim/       panel horaire
  processed/     panel + cible + variables
rapport.ipynb    récit, graphiques, résultats
```

Exemple :

```python
from src.collecte_binance import download_binance, load_binance
download_binance("BTCUSDT", "metrics", "2023-01-01", "2023-01-31")   # 5 min, fichiers quotidiens
download_binance("BTCUSDT", "klines", "2023-01-01", "2023-12-31")    # 1 h, fichiers mensuels
df = load_binance("BTCUSDT", "metrics")
```

## Les données

| Source | Contenu | Fréquence | Accès |
|---|---|---|---|
| Binance `metrics` | open interest, ratios long/short (gros traders, tous les comptes), ratio taker acheteurs/vendeurs | 5 min, fichiers **quotidiens** | data.binance.vision + `.CHECKSUM` |
| Binance `fundingRate` | funding et intervalle (8/4/1 h) | par paiement, fichiers mensuels | idem |
| Binance `klines` 1h | OHLCV du perpétuel | 1 h | idem |
| Binance `premiumIndexKlines` 1h | basis perpétuel/indice | 1 h | idem |
| Binance spot `klines` 1h | OHLCV spot | 1 h | idem |
| Bybit API v5 | funding, open interest, bougies (high, close, volumes) | par paiement ; 1 h | REST paginé, sans clé |
| alternative.me | Fear & Greed | quotidien | REST |

### Constats de la première vérification (5 octobre 2026)

- Les `metrics` sont bien à 5 minutes, soit 288 lignes par jour. Ils commencent le 2020-09-01 pour BTC et le 2021-12-01 pour SOL.
- **Les ratios long/short manquent sur une grande partie de 2022.** Sur BTCUSDT et SOLUSDT, les ratios gros traders et tous comptes sont vides du 2022-01-01 au 2022-12-16, sauf entre le 28 mai et le 20 juin environ. Le ratio taker est vide avant le 2022-05-14. L'open interest, lui, est complet. Conséquence : il faut soit démarrer l'échantillon fin 2022, soit garder 2022 pour les seules variables d'open interest et de funding. Ce choix est à documenter dans le rapport.
- Le funding est versé toutes les 8 h sur BTC et SOL pour les mois testés. La colonne `funding_interval_hours` sert à ramener chaque taux en horaire.
- Les fichiers spot récents (2025) ont des timestamps en microsecondes. `collecte_binance._to_utc` détecte l'unité automatiquement.

### Collecte Bybit (12 octobre 2026)

Bybit sert de source de robustesse : vérifier que les résultats ne sont pas propres aux clients de Binance. Trois séries, pour BTCUSDT, ETHUSDT et SOLUSDT, du 2022-01-01 au 2026-09-30 :

| Série | Point d'accès | Colonnes gardées | Lignes par crypto | Usage |
|---|---|---|---|---|
| `funding` | `/v5/market/funding/history` | `calc_time`, `last_funding_rate`, `funding_interval_hours` | 5 202 (SOL : 5 562) | prédicteur |
| `open_interest` | `/v5/market/open-interest`, pas 1 h | `create_time`, `open_interest`, `single_open_interest` | 41 616 | C2 et prédicteur |
| `klines` | `/v5/market/kline`, intervalle 60 | `open_time`, `high`, `close`, `volume`, `quote_volume` | 41 616 | C1 et prédicteurs de prix |

Méthode (`src/collecte_bybit.py`) :
- L'API renvoie au plus 200 points par requête (1 000 pour les bougies), du plus récent au plus ancien. La boucle remonte donc le temps : la fin de chaque requête est placée juste avant le point le plus ancien de la page précédente.
- Bybit renvoie ses erreurs avec un code HTTP 200 : on contrôle `retCode`. Les erreurs passagères du serveur (« svc error ») sont retentées jusqu'à 5 fois.
- Nettoyage : conversion du texte en nombres, des millisecondes en dates UTC, suppression des doublons entre pages, tri chronologique. Les colonnes portent les mêmes noms que chez Binance pour réutiliser le code du panel.
- L'open interest est pris au pas horaire plutôt qu'à 5 minutes (moins de 200 requêtes par crypto contre environ 2 000), ce qui suffit pour une vérification.
- Un parquet par crypto et par série. Relancer le script ne retélécharge pas les fichiers déjà présents.

Constats :
- Aucune heure manquante : 41 616 lignes, soit 1 734 jours × 24 h, pour l'open interest et les bougies des trois cryptos.
- **Le funding de SOL est passé toutes les 2 h du 10 novembre au 20 décembre 2022**, après la faillite de FTX, contre 8 h le reste du temps. La colonne `funding_interval_hours`, calculée comme l'écart entre deux paiements, permet de ramener chaque taux en horaire.
- `open_interest` compte les deux côtés du marché et vaut le double de `single_open_interest`. Pour comparer au `sum_open_interest` de Binance, utiliser `single_open_interest`. Pour C2, qui mesure une variation relative, les deux donnent le même résultat.
- Bybit ne fournit pas le volume acheteur agressif dans ses bougies : la condition C3 ne peut pas être refaite sur Bybit.

## Répartition

| Période | A | B | C |
|---|---|---|---|
| Début octobre | dépôt, collecte Binance | Bybit et Fear & Greed | revue de littérature |
| Mi à fin octobre | panel horaire | cible | variables |
| Début novembre | analyse descriptive (en commun) | | |
| Mi-novembre | logit et base naïve | gradient boosting | stratégie et frais |
| Fin novembre | robustesse | rédaction | reproductibilité |

Méthode de travail : une branche et une pull request par tâche.
