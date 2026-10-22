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
uv run python -m scripts.telecharger_fng       # indice Fear & Greed (une requête)
uv run python -m scripts.construire_panel      # panel horaire -> data/interim/panel.parquet
uv run jupyter lab                             # ouvrir rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, généré par `uv export --no-hashes --no-dev > requirements.txt`.

Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Le dossier local se change avec la variable d'environnement `DATA_DIR`.

**Données en ligne.** Les données brutes et le panel sont déposés sur le stockage S3 du SSPCloud, dans un dossier public en lecture seule : `https://minio.lab.sspcloud.fr/ethany1/diffusion/ensae-squeeze/`, avec la même arborescence que `data/`. Le notebook les lit sans identifiants (`src.stockage.lire_public`). Pour en déposer de nouvelles, il faut des identifiants SSPCloud dans un fichier `.env` (jamais commité) et lancer Python avec `uv run --env-file .env python ...` (`src.stockage.envoyer`).

## Structure

```
src/
  config.py            chemins, univers de cryptos, période, URL, S3     ← tout le monde importe d'ici
  collecte_binance.py  archives Binance : téléchargement, checksum, parquet
  collecte_bybit.py    API Bybit v5 : funding, open interest, bougies
  collecte_fng.py      indice Fear & Greed (alternative.me), quotidien
  stockage.py          envoi sur le S3 du SSPCloud et lecture publique
  panel.py             nettoyage et panel horaire (une ligne par crypto et par heure t)
  -- à venir --
  cible.py       définition des squeezes                                    (B)
  variables.py   construction des variables                                 (C)
  modeles.py     base naïve, logit, LightGBM, validation temporelle
  strategie.py   backtest avec frais et funding
scripts/
  telecharger_binance.py   télécharge toutes les archives Binance (SYMBOLS × types)
  telecharger_bybit.py     télécharge toutes les séries Bybit
  telecharger_fng.py       télécharge l'indice Fear & Greed
  construire_panel.py      construit data/interim/panel.parquet et affiche les manquants
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
| Bybit API v5 | funding, open interest, bougies OHLCV | par paiement ; 1 h | REST paginé, sans clé |
| alternative.me | Fear & Greed | quotidien | REST |

### Constats sur les données Binance (BTCUSDT, 2022-2026)

- Les `metrics` sont à 5 minutes, soit 288 relevés par jour. Sur BTC, 133 relevés manquent sur 499 392 attendus (un trou de 10 h 30 et trois trous de 15 à 20 min) : 9 heures du panel n'ont aucun relevé.
- **Seuls les ratios des gros traders manquent sur la majeure partie de 2022**, pas tous les ratios. Part de relevés manquants par mois en 2022 :

  | Variable | Janv. | Févr.-avr. | Mai | Juin | Juil.-nov. | Déc. |
  |---|---|---|---|---|---|---|
  | `ls_top_comptes`, `ls_top_positions` (gros traders) | 97 % | 100 % | 86 % | 21 % | 100 % | 44 % (complets à partir du 2022-12-15) |
  | `ls_global` (tous les comptes) | 60 % | 0 % | 0 % | 0 % | 0 % | 0 % |
  | `taker_ratio` | 97 % | 100 % | 29 % | 0 % | 0 % | 0 % |

  Le ratio global est donc disponible dès février 2022 et le ratio taker dès la mi-mai 2022. L'open interest est complet. Conséquence : on peut garder 2022 pour l'open interest, le funding, le ratio global et (après mi-mai) le ratio taker, et traiter à part les ratios gros traders, disponibles seulement à partir de mi-décembre 2022. Ce choix est à documenter dans le rapport. À vérifier sur les autres cryptos.
- Quelques open interest valent 0 (463 relevés sur BTC), ce qui est impossible : ils sont remplacés par des valeurs manquantes dans le panel.
- Le ratio taker atteint des valeurs extrêmes (jusqu'à 63 sur 5 minutes, pour une médiane de 0,99). Elles sont gardées : elles sont possibles sur 5 minutes de faible volume.
- Côté bougies : le perpétuel est complet (41 616 heures), il manque 1 heure au spot et 3 jours à l'indice de prime (`basis`).
- Le funding est versé toutes les 8 h sur BTC et SOL pour les mois testés. La colonne `funding_interval_hours` sert à ramener chaque taux en horaire.
- Les fichiers spot récents (2025) ont des timestamps en microsecondes. `collecte_binance._to_utc` détecte l'unité automatiquement.

### Collecte Bybit (12 octobre 2026)

Bybit sert de source de robustesse : vérifier que les résultats ne sont pas propres aux clients de Binance. Trois séries, pour BTCUSDT, ETHUSDT et SOLUSDT, du 2022-01-01 au 2026-09-30 :

| Série | Point d'accès | Colonnes gardées | Lignes par crypto | Usage |
|---|---|---|---|---|
| `funding` | `/v5/market/funding/history` | `calc_time`, `last_funding_rate`, `funding_interval_hours` | 5 202 (SOL : 5 562) | prédicteur |
| `open_interest` | `/v5/market/open-interest`, pas 1 h | `create_time`, `open_interest`, `single_open_interest` | 41 616 | C2 et prédicteur |
| `klines` | `/v5/market/kline`, intervalle 60 | `open_time`, `open`, `high`, `low`, `close`, `volume`, `quote_volume` | 41 616 | C1 et prédicteurs de prix |

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
- Une crypto absente de Bybit, ou cotée sous un autre nom (`1000SHIBUSDT` chez Binance, `SHIB1000USDT` chez Bybit), est signalée puis sautée par `scripts/telecharger_bybit.py`.

### Indice Fear & Greed (alternative.me)

Une seule requête (`limit=0`) renvoie tout l'historique depuis le 2018-02-01 : 3 180 jours au 20 octobre 2026. Tout l'historique est gardé, pas seulement la période d'étude, car les moyennes glissantes ont besoin des jours qui précèdent `START`.

Constats et points de vigilance pour le panel :
- L'horodatage est en **secondes** (millisecondes chez Binance et Bybit).
- **4 jours manquent** : du 2018-04-14 au 2018-04-16, et le 2024-10-26. Seul le dernier est dans la période d'étude : il sera comblé par la valeur de la veille.
- **Risque de fuite d'information** : la valeur datée du jour J (00:00 UTC) est publiée vers minuit UTC, sans garantie qu'elle soit connue à 00:00 précisément. Dans le panel, l'indice sera donc **décalé d'un jour** : à l'heure t du jour J, on utilise la valeur de J−1.
- `fng_classe` (« Fear », « Greed »…) se déduit de `fng` : elle reste dans les données brutes mais ne sert pas de variable.

## Convention de temps : que représente l'heure t ?

Le panel a une ligne par crypto et par heure t (UTC). **La ligne t contient uniquement ce qui est connu à l'instant t.** Toutes les sources sont ramenées à cette règle avant la fusion, et toute la suite du projet (cible, variables, validation) s'appuie dessus.

### Pourquoi c'est nécessaire

Les sources n'horodatent pas de la même façon. Chez Binance, une bougie `open_time = 14:00` couvre 14:00 à 15:00 : son `close` et son `high` ne sont connus qu'à **15:00**. Un relevé `metrics` daté de 14:05 est connu dès 14:05. Si l'on fusionnait les sources sur leur horodatage brut, la ligne « 14:00 » contiendrait le prix de 15:00 à côté de l'open interest de 14:00. Un modèle pourrait alors « voir » une heure dans le futur, ce qui est une fuite d'information.

### La règle : chaque ligne est datée par la fin de l'heure

La ligne t = 15:00 contient :

| Source | Ce qu'on met dans la ligne t | Transformation |
|---|---|---|
| `klines`, `spot_klines` | la bougie 14:00-15:00 : `close` est le prix à 15:00, `high` et `low` les extrêmes entre 14:00 et 15:00 | t = `open_time` + 1 h |
| `premiumIndexKlines` | la basis de la bougie 14:00-15:00 | t = `open_time` + 1 h |
| `metrics` : open interest | le dernier relevé dont l'heure est ≤ 15:00 | regroupement des relevés de 14:05 à 15:00 inclus, dernière valeur |
| `metrics` : ratios long/short et taker | la moyenne des relevés de 14:05 à 15:00 inclus | regroupement fermé à droite, daté par sa borne droite |
| `fundingRate` | le dernier paiement dont l'heure (arrondie à l'heure) est ≤ 15:00 | report de la dernière valeur connue |
| Fear & Greed | la valeur de la veille (la valeur du jour J n'est utilisée qu'à partir de J+1 00:00) | date de publication + 1 jour, puis dernière valeur connue |

Hypothèse sur les `metrics` : un relevé daté de 14:05 est supposé connu à 14:05, c'est-à-dire que `create_time` marque la fin de la mesure. Si l'on découvre le contraire, il suffit de décaler les relevés de 5 minutes.

### Conséquences pour la cible et les variables

- **Variables** : elles n'utilisent que les lignes ≤ t (fenêtres glissantes passées). La ligne t elle-même est autorisée, puisqu'elle ne contient que du connu à t.
- **Cible** : elle n'utilise que les lignes de t+1 h à t+24 h. La bougie de la ligne t+1 h couvre t à t+1 h : c'est bien la première heure après t.
- **Lien avec les notations de la définition des squeezes** : `Close_{t−1}` et `OI_{t−1}` désignent le dernier prix et le dernier OI connus avant le début de la fenêtre. Ce sont le `close` et l'open interest **de la ligne t**. `max High(t, t+24h]` est le maximum des `high` des lignes t+1 h à t+24 h.
- **Grille complète** : le panel contient toutes les heures de `START` à `END`. Une heure sans données apparaît comme une ligne de valeurs manquantes au lieu de disparaître.

### Mise en œuvre (pandas)

- bougies : `open_time + pd.Timedelta("1h")` ;
- relevés à 5 minutes : `resample("1h", closed="right", label="right")` (fermé à droite, daté par la borne droite) ;
- funding et Fear & Greed : `pd.merge_asof(..., direction="backward")` (dernière valeur dont la date est ≤ t).

## Répartition

| Période | A | B | C |
|---|---|---|---|
| Début octobre | dépôt, collecte Binance | Bybit et Fear & Greed | revue de littérature |
| Mi à fin octobre | panel horaire | cible | variables |
| Début novembre | analyse descriptive (en commun) | | |
| Mi-novembre | logit et base naïve | gradient boosting | stratégie et frais |
| Fin novembre | robustesse | rédaction | reproductibilité |

Méthode de travail : une branche et une pull request par tâche.
