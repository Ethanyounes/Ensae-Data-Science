# Prédire les short squeezes sur les perpétuels crypto

Projet de Data Science, ENSAE 2A.

1. [Question et hypothèses](#question-et-hypothèses)
2. [Installation et reproduction](#installation-et-reproduction)
3. [Organisation du dépôt](#organisation-du-dépôt)
4. [Données](#données)
5. [Convention de temps](#convention-de-temps)
6. [Nettoyage et panel horaire](#nettoyage-et-panel-horaire)
7. [Analyse descriptive](#analyse-descriptive)
8. [Répartition du travail](#répartition-du-travail)

## Question et hypothèses

**Question** : peut-on prédire un short squeeze 24 heures à l'avance à partir du positionnement des traders sur les perpétuels crypto (Binance USD-M) ?

- **H1, shorts informés** : une accumulation de shorts annonce une baisse, et le positionnement ne prédit pas de squeeze.
- **H2, shorts fragiles** : des shorts accumulés avec du levier rendent le marché vulnérable à un rachat forcé.

**Cible.** Il y a un short squeeze à l'heure t si les trois conditions suivantes sont réunies dans les 24 h qui suivent (valeurs de base) :
- **C1, forte hausse** : `max High(t, t+24h] / Close_{t−1} − 1 ≥ max(3·σ_30j·√24, 5 %)` ;
- **C2, rachat des shorts** : l'open interest au pic est inférieur d'au moins 10 % à `OI_{t−1}` ;
- **C3, achats agressifs** : le ratio taker moyen entre t et le pic est supérieur à 1.

Variantes testées : k = 2,5 ou 4 ; plancher de 3 ou 8 % ; δ = 5 ou 15 % ; horizon de 12 ou 48 h.

**La réponse comporte deux volets :**
- **statistique** : le positionnement bat-il une base naïve, et un modèle qui n'utilise que les prix (PR-AUC) ?
- **économique** : ce gain reste-t-il rentable une fois les frais et le funding payés (backtest) ?

## Installation et reproduction

On utilise [uv](https://docs.astral.sh/uv/) avec Python 3.12.

```bash
uv sync                                        # crée .venv avec les versions de uv.lock
uv run python -m scripts.telecharger_binance   # collecte Binance (relançable)
uv run python -m scripts.telecharger_bybit     # collecte Bybit (moins de 5 min)
uv run python -m scripts.telecharger_fng       # indice Fear & Greed (une requête)
uv run python -m scripts.construire_panel      # panels horaires -> data/interim/
uv run jupyter lab                             # ouvrir rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, généré par `uv export --no-hashes --no-dev > requirements.txt`.

**Données locales.** Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Le dossier local se change avec la variable d'environnement `DATA_DIR`.

**Données en ligne.** Les données brutes et les panels sont déposés sur le stockage S3 du SSPCloud, dans un dossier public en lecture seule, avec la même arborescence que `data/` : `https://minio.lab.sspcloud.fr/ethany1/diffusion/ensae-squeeze/`.
- Le notebook les lit sans identifiants (`src.stockage.lire_public`).
- Pour en déposer de nouvelles (`src.stockage.envoyer`), il faut des identifiants SSPCloud dans un fichier `.env`, jamais commité, et lancer Python avec `uv run --env-file .env python ...`.

## Organisation du dépôt

```
src/
  config.py            chemins, cryptos, période, URL, S3            ← tout le monde importe d'ici
  collecte_binance.py  archives Binance : téléchargement, checksum, parquet
  collecte_bybit.py    API Bybit v5 : funding, open interest, bougies
  collecte_fng.py      indice Fear & Greed (alternative.me), quotidien
  stockage.py          envoi sur le S3 du SSPCloud et lecture publique
  panel_binance.py     nettoyage et panel horaire (une ligne par crypto et par heure t)
  panel_bybit.py       panel horaire Bybit et fusion avec le panel Binance
  -- à venir --
  cible.py             définition des squeezes                            (B)
  variables.py         construction des variables                         (C)
  descriptif.py        statistiques et graphiques descriptifs             (en commun)
  modeles.py           base naïve, logit, LightGBM, validation temporelle
  strategie.py         backtest avec frais et funding
scripts/
  telecharger_binance.py   télécharge toutes les archives Binance (SYMBOLS × types)
  telecharger_bybit.py     télécharge toutes les séries Bybit
  telecharger_fng.py       télécharge l'indice Fear & Greed
  construire_panel.py      construit les panels Binance, Bybit et leur fusion, et affiche les manquants
data/
  raw/binance/{metrics,fundingRate,klines,premiumIndexKlines,spot_klines}/<SYMBOLE>/*.parquet
  raw/bybit/{funding,open_interest,klines}/<SYMBOLE>/<SYMBOLE>-<série>.parquet
  raw/fng/
  interim/       panels horaires
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

## Données

### Sources

| Source | Contenu | Fréquence | Accès |
|---|---|---|---|
| Binance `metrics` | open interest, ratios long/short (gros traders, tous les comptes), ratio taker acheteurs/vendeurs | 5 min, fichiers **quotidiens** | data.binance.vision + `.CHECKSUM` |
| Binance `fundingRate` | funding et intervalle (8/4/1 h) | par paiement, fichiers mensuels | idem |
| Binance `klines` 1h | OHLCV du perpétuel | 1 h | idem |
| Binance `premiumIndexKlines` 1h | basis perpétuel/indice | 1 h | idem |
| Binance spot `klines` 1h | OHLCV spot | 1 h | idem |
| Bybit API v5 | funding, open interest, bougies OHLCV | par paiement ; 1 h | REST paginé, sans clé |
| alternative.me | Fear & Greed | quotidien | REST |

### Période d'étude : 2023-01-01 au 2026-09-30

Sur BTC en 2022, **seuls les ratios des gros traders manquent sur la majeure partie de l'année**. Part de relevés manquants par mois :

| Variable | Janv. | Févr.-avr. | Mai | Juin | Juil.-nov. | Déc. |
|---|---|---|---|---|---|---|
| `ls_top_comptes`, `ls_top_positions` (gros traders) | 97 % | 100 % | 86 % | 21 % | 100 % | 44 % (complets à partir du 2022-12-15) |
| `ls_global` (tous les comptes) | 60 % | 0 % | 0 % | 0 % | 0 % | 0 % |
| `taker_ratio` | 97 % | 100 % | 29 % | 0 % | 0 % | 0 % |

Le ratio global est disponible dès février 2022, le ratio taker dès la mi-mai 2022, et l'open interest est complet. À vérifier sur les autres cryptos.

**Décision.** La période d'étude commence le 2023-01-01 (`START` dans `config.py`), pour avoir toutes les variables de positionnement, ratios des gros traders compris, sur toute la période.
- Le panel commence 30 jours plus tôt (`CHAUFFE_JOURS`, soit le 2022-12-02). Ces jours de chauffe servent aux fenêtres glissantes, comme la volatilité sur 30 jours de la cible, mais n'entrent pas dans l'analyse.
- Les fichiers plus anciens de 2022 restent dans `data/raw/` sans être utilisés.

### Binance : constats (BTCUSDT, 2022-2026)

- **Relevés `metrics`** : 288 par jour (un toutes les 5 minutes). 133 manquent sur 499 392 attendus (un trou de 10 h 30 et trois trous de 15 à 20 min) : 9 heures du panel n'ont aucun relevé.
- **Open interest nul** : 463 relevés valent 0, ce qui est impossible. Ils sont remplacés par des valeurs manquantes dans le panel.
- **Ratio taker extrême** : jusqu'à 63 sur 5 minutes, pour une médiane de 0,99. Ces valeurs sont gardées, car elles sont possibles sur 5 minutes de faible volume.
- **Bougies** : le perpétuel est complet (41 616 heures), il manque 1 heure au spot et 3 jours à l'indice de prime (`basis`).
- **Funding** : versé toutes les 8 h sur BTC et SOL pour les mois testés. La colonne `funding_interval_hours` sert à ramener chaque taux en horaire.
- **Format des dates** : les fichiers spot récents (2025) sont en microsecondes. `collecte_binance.to_utc` détecte l'unité automatiquement.

### Bybit : collecte et constats (12 octobre 2026)

Bybit sert à vérifier que les résultats ne sont pas propres aux clients de Binance. Les fichiers actuels de BTCUSDT, ETHUSDT et SOLUSDT couvrent le 2022-01-01 au 2026-09-30 : ils ont été téléchargés avant le passage de `START` à 2023. Un nouveau téléchargement commence à `DEBUT_DONNEES`.

| Série | Point d'accès | Colonnes gardées | Lignes par crypto | Usage |
|---|---|---|---|---|
| `funding` | `/v5/market/funding/history` | `calc_time`, `last_funding_rate`, `funding_interval_hours` | 5 202 (SOL : 5 562) | prédicteur |
| `open_interest` | `/v5/market/open-interest`, pas 1 h | `create_time`, `open_interest`, `single_open_interest` | 41 616 | C2 et prédicteur |
| `klines` | `/v5/market/kline`, intervalle 60 | `open_time`, `open`, `high`, `low`, `close`, `volume`, `quote_volume` | 41 616 | C1 et prédicteurs de prix |

**Méthode** (`src/collecte_bybit.py`) :
- **Pagination** : l'API renvoie au plus 200 points par requête (1 000 pour les bougies), du plus récent au plus ancien. La boucle remonte donc le temps : chaque requête s'arrête juste avant le point le plus ancien de la page précédente.
- **Erreurs** : Bybit renvoie ses erreurs avec un code HTTP 200, on contrôle donc `retCode`. Les erreurs passagères du serveur (« svc error ») sont retentées jusqu'à 5 fois.
- **Nettoyage** : texte converti en nombres, millisecondes en dates UTC, doublons entre pages supprimés, tri chronologique. Les colonnes portent les mêmes noms que chez Binance pour réutiliser le code du panel.
- **Open interest horaire** plutôt qu'à 5 minutes : moins de 200 requêtes par crypto contre environ 2 000, ce qui suffit pour une vérification.
- **Fichiers** : un parquet par crypto et par série. Relancer le script ne retélécharge pas les fichiers déjà présents.

**Constats** :
- **Aucune heure manquante** : 41 616 lignes (1 734 jours × 24 h) pour l'open interest et les bougies des trois cryptos.
- **Funding de SOL toutes les 2 h du 10 novembre au 20 décembre 2022**, après la faillite de FTX, contre 8 h le reste du temps. `funding_interval_hours`, calculée comme l'écart entre deux paiements, permet de ramener chaque taux en horaire.
- **Deux open interest** : `open_interest` compte les deux côtés du marché et vaut le double de `single_open_interest`. Pour comparer au `sum_open_interest` de Binance, on utilise `single_open_interest`. Pour C2, qui mesure une variation relative, les deux donnent le même résultat.
- **Pas de volume acheteur agressif** dans les bougies Bybit : la condition C3 ne peut pas être refaite sur Bybit.
- **Contrats absents ou renommés** : une crypto absente de Bybit est signalée puis sautée par `scripts/telecharger_bybit.py`. Un contrat coté sous un autre nom est téléchargé sous son nom Bybit (voir la [table de correspondance](#panel-bybit-et-fusion-srcpanel_bybitpy)).

### Indice Fear & Greed (alternative.me)

Une seule requête (`limit=0`) renvoie tout l'historique depuis le 2018-02-01 : 3 180 jours au 20 octobre 2026. Tout l'historique est gardé, car les moyennes glissantes ont besoin des jours qui précèdent `START`.

- **Horodatage en secondes** (millisecondes chez Binance et Bybit).
- **4 jours manquent** : du 2018-04-14 au 2018-04-16, et le 2024-10-26. Seul le dernier est dans la période d'étude : il sera comblé par la valeur de la veille.
- **Risque de fuite d'information** : la valeur du jour J (datée 00:00 UTC) est publiée vers minuit UTC, sans garantie qu'elle soit connue à 00:00 précisément. Dans le panel, l'indice sera donc **décalé d'un jour** : à l'heure t du jour J, on utilise la valeur de J−1.
- **`fng_classe`** (« Fear », « Greed »…) se déduit de `fng` : elle reste dans les données brutes mais ne sert pas de variable.

## Convention de temps

Le panel a une ligne par crypto et par heure t (UTC). **La ligne t contient uniquement ce qui est connu à l'instant t.** Toutes les sources sont ramenées à cette règle avant la fusion, et toute la suite du projet (cible, variables, validation) s'appuie dessus.

### Pourquoi c'est nécessaire

Les sources n'horodatent pas de la même façon. Chez Binance, une bougie `open_time = 14:00` couvre 14:00 à 15:00 : son `close` et son `high` ne sont connus qu'à **15:00**. Un relevé `metrics` daté de 14:05 est connu dès 14:05. Si l'on fusionnait les sources sur leur horodatage brut, la ligne « 14:00 » contiendrait le prix de 15:00 à côté de l'open interest de 14:00. Un modèle pourrait alors « voir » une heure dans le futur : c'est une fuite d'information.

### La règle : chaque ligne est datée par la fin de l'heure

La ligne t = 15:00 contient :

| Source | Contenu de la ligne t | Transformation |
|---|---|---|
| `klines`, `spot_klines` | la bougie 14:00-15:00 : `close` est le prix à 15:00, `high` et `low` les extrêmes entre 14:00 et 15:00 | t = `open_time` + 1 h |
| `premiumIndexKlines` | la basis de la bougie 14:00-15:00 | t = `open_time` + 1 h |
| `metrics` : open interest | le dernier relevé dont l'heure est ≤ 15:00 | regroupement des relevés de 14:05 à 15:00 inclus, dernière valeur |
| `metrics` : ratios long/short et taker | la moyenne des relevés de 14:05 à 15:00 inclus | regroupement fermé à droite, daté par sa borne droite |
| `fundingRate` | le dernier paiement dont l'heure (arrondie à l'heure) est ≤ 15:00 | report de la dernière valeur connue |
| Fear & Greed | la valeur de la veille (celle du jour J n'est utilisée qu'à partir de J+1 00:00) | date de publication + 1 jour, puis dernière valeur connue |

Hypothèse sur les `metrics` : un relevé daté de 14:05 est supposé connu à 14:05, c'est-à-dire que `create_time` marque la fin de la mesure. Si l'on découvre le contraire, il suffit de décaler les relevés de 5 minutes.

### Conséquences pour la cible et les variables

- **Variables** : elles n'utilisent que les lignes ≤ t (fenêtres glissantes passées). La ligne t elle-même est autorisée, puisqu'elle ne contient que ce qui est connu à t.
- **Cible** : elle n'utilise que les lignes de t+1 h à t+24 h. La bougie de la ligne t+1 h couvre t à t+1 h : c'est bien la première heure après t.
- **Notations de la cible** : `Close_{t−1}` et `OI_{t−1}` sont le dernier prix et le dernier open interest connus avant le début de la fenêtre, c'est-à-dire le `close` et l'open interest **de la ligne t**. `max High(t, t+24h]` est le maximum des `high` des lignes t+1 h à t+24 h.
- **Chauffe** : on calcule la cible et les variables sur tout le panel, puis on ne garde que les heures à partir de `START` (`panel.loc[START:]`). Si une variable utilise une fenêtre de plus de 30 jours, il faut augmenter `CHAUFFE_JOURS`.
- **Grille complète** : le panel contient toutes les heures de `DEBUT_DONNEES` à `END`. Une heure sans données apparaît comme une ligne de valeurs manquantes au lieu de disparaître.

### Mise en œuvre (pandas)

- bougies : `open_time + pd.Timedelta("1h")` ;
- relevés à 5 minutes : `resample("1h", closed="right", label="right")` (fermé à droite, daté par la borne droite) ;
- funding et Fear & Greed : `pd.merge_asof(..., direction="backward")` (dernière valeur dont la date est ≤ t).

## Nettoyage et panel horaire

### Panel Binance (`src/panel_binance.py`)

Le panel (`data/interim/panel.parquet`) a une ligne par crypto et par heure t, du 2022-12-02 01:00 (chauffe comprise) au 2026-10-01 00:00, soit 33 576 heures par crypto.

| Choix | Décision | Raison |
|---|---|---|
| Période | 2023-01-01 au 2026-09-30, plus 30 jours de chauffe | voir [Période d'étude](#période-détude--2023-01-01-au-2026-09-30) |
| Pas de temps | horaire, ligne t = ce qui est connu à t | pas de temps des bougies et de la cible ; pas de fuite d'information (voir [Convention de temps](#convention-de-temps)) |
| `metrics` (5 min) → 1 h | open interest : dernier relevé de l'heure ; ratios : moyenne des relevés | l'open interest est un niveau, les ratios des mesures sur 5 minutes |
| Open interest nul | remplacé par une valeur manquante | impossible : erreur d'enregistrement |
| Ratio taker extrême (jusqu'à 63) | gardé | possible sur 5 minutes de faible volume, notamment pendant un squeeze |
| Valeurs manquantes | jamais comblées ; une heure absente reste une ligne vide | ne pas inventer de données ; `n_releves` (12 = heure complète) indique la fiabilité de l'heure |
| Funding | heure de paiement arrondie (16:00:00.007 → 16:00), taux ramené à une heure (`funding_horaire`), chaque heure reçoit le dernier paiement connu | rattacher les paiements à la grille et comparer des intervalles de 8, 4 ou 1 h |
| Dates | toutes en UTC, en millisecondes | le spot récent est en microsecondes : sans harmonisation, les jointures échouent |
| Noms | `oi`, `oi_usd`, `ls_top_comptes`, `ls_top_positions`, `ls_global`, `taker_ratio` ; préfixe `spot_` pour le spot | les noms Binance sont trompeurs (`sum_...` pour un ratio) |
| Colonnes retirées | `ignore`, `close_time`, volumes de l'indice de prime (on garde son close sous le nom `basis`) | toujours vides ou nulles |

**Valeurs manquantes** (BTC, ETH, SOL, chauffe comprise) :

| Colonnes | Manquants | Cause |
|---|---|---|
| prix, volumes, funding | 0 % | |
| open interest, ratio global, ratio taker | 0,03 % | un trou de relevés chez Binance |
| basis | 0,14 % | 3 jours absents |
| ratios des gros traders | 0,9 % | presque entièrement pendant la chauffe (2 au 14 décembre 2022) |

Le funding est versé toutes les 8 h pour les trois cryptos sur toute la période. Entre 15 et 19 heures par crypto ont moins de 12 relevés.

Pas encore dans le panel : l'indice Fear & Greed (à rattacher avec un jour de décalage).

### Panel Bybit et fusion (`src/panel_bybit.py`)

Trois fichiers dans `data/interim/` :

| Fichier | Contenu |
|---|---|
| `panel.parquet` | panel Binance (référence de l'analyse) |
| `panel_bybit.parquet` | panel Bybit, mêmes heures t, mêmes noms de colonnes |
| `panel_binance_bybit.parquet` | panel Binance complété des colonnes Bybit, suffixe `_bybit` (`close_bybit`, `oi_bybit`…), fusion sur la crypto et l'heure t |

**Choix :**
- **Même convention de temps que Binance**, vérifiée sur les données. Les bougies Bybit sont datées par leur début (prix identiques à Binance à 0,01 % près pour une même `open_time`), donc t = `open_time` + 1 h. L'open interest Bybit est un relevé instantané : ses variations sont corrélées à celles de Binance à la même heure (0,36 à 0,44), mais plus du tout avec une heure de décalage (0,07). Il est donc connu à son heure.
- **Open interest** : `single_open_interest` (un seul côté du marché, comme Binance), renommé `oi`.
- **Funding** : même traitement que Binance (`clean_funding`).
- **Crypto absente de Bybit** : absente du panel Bybit ; ses colonnes `_bybit` restent vides dans la fusion.

**Table de correspondance des contrats** (`BYBIT_NOMS` et `nom_bybit` dans `config.py`). Sur les 659 perpétuels USDT de Binance (relevé du 2026-11-02 sur les API des deux plateformes), 464 existent sous le même nom chez Bybit et 5 sous un autre nom :

| Binance | Bybit | Facteur |
|---|---|---|
| `1000SATSUSDT` | `10000SATSUSDT` | 10 |
| `1000SHIBUSDT` | `SHIB1000USDT` | 1 |
| `TAGUSDT` | `1000TAGUSDT` | 1000 |
| `TOSHIUSDT` | `1000TOSHIUSDT` | 1000 |
| `TURBOUSDT` | `1000TURBOUSDT` | 1000 |

Le facteur est la quantité de crypto d'un contrat Bybit divisée par celle d'un contrat Binance. Dans le panel Bybit, les prix sont divisés par ce facteur, le volume en crypto et l'open interest multipliés par lui, pour être exprimés dans l'unité Binance. Les fichiers bruts Bybit sont rangés sous le nom Binance du contrat.

**Premières comparaisons** (2023-01 à 2026-09) :

| | Corrélation des rendements horaires | Corrélation des variations d'OI | Corrélation du funding | Écart de prix moyen |
|---|---|---|---|---|
| BTC | 0,9996 | 0,44 | 0,78 | 0,013 % |
| ETH | 0,9997 | 0,36 | 0,78 | 0,013 % |
| SOL | 0,9996 | 0,37 | 0,93 | 0,018 % |

Les prix sont quasiment identiques : les deux sources sont bien alignées dans le temps. Le positionnement (open interest, funding) est commun aux deux plateformes mais pas identique, car chaque plateforme a ses propres clients.

## Analyse descriptive

**Objectif.** Montrer que la cible est rare et groupée dans le temps, décrire les variables de positionnement, puis voir si le positionnement diffère déjà avant un squeeze. Cette dernière étape donne une première réponse à H1 contre H2, sans modèle. Le code ira dans `src/descriptif.py`, appelé par `rapport.ipynb`. Chaque graphique y est commenté et relié à la question.

| Étape | Question | Données nécessaires |
|---|---|---|
| 1. Données et univers | Les données sont-elles fiables, et les cryptos comparables ? | panel actuel |
| 2. Variables de positionnement | Comment se comportent funding, open interest et ratios ? | panel actuel |
| 3. La cible | Combien de squeezes, quand, et de quelle forme ? | `cible.py` |
| 4. Positionnement et squeezes | Le positionnement diffère-t-il avant un squeeze ? (H1/H2) | `cible.py` |
| 5. Hétérogénéité et robustesse | Les constats tiennent-ils partout ? | `cible.py`, Bybit, Fear & Greed |

### Étape 1 : données et univers

- **Carte des valeurs manquantes** par crypto et par mois (`rapport_qualite` étendu aux 25 cryptos) : repérer les cryptos cotées après janvier 2023 et les trous de relevés.
- **Cohérence du ratio taker** : recalculer le ratio à 1 h depuis les bougies (`taker_buy_volume / (volume − taker_buy_volume)`) et le comparer à `taker_ratio`, issu des relevés à 5 min. Ce contrôle croise deux sources Binance et donne une seconde mesure possible pour C3.
- **Tableau par crypto** : volume moyen en USDT, open interest moyen en USD, volatilité annualisée, rendement total, funding moyen annualisé, part des heures à funding négatif.
- **Distribution des rendements horaires** (kurtosis, QQ-plot) : queues épaisses et volatilités très différentes d'une crypto à l'autre, d'où un seuil C1 exprimé en σ.
- **Volume du perpétuel rapporté au volume spot** (`quote_volume / spot_quote_volume`) : part de la spéculation à levier, par crypto et dans le temps.

### Étape 2 : variables de positionnement

- **Distributions** : funding (forte masse attendue au taux par défaut de Binance, 0,01 % par 8 h), ratios long/short, ratio taker (valeurs extrêmes).
- **Séries 2023-2026**, avec les grandes phases du marché. Les niveaux changent d'une crypto à l'autre et dans le temps, d'où des z-scores glissants par crypto plutôt que des niveaux bruts.
- **Corrélations entre variables** (heatmap) : les trois ratios long/short sont sans doute très corrélés, ce qui posera un problème de multicolinéarité dans le logit.
- **Persistance** (autocorrélation) : le funding et l'open interest évoluent lentement, le ratio taker très vite.

### Étape 3 : la cible

- **Fréquence des squeezes** : globale, par crypto, par année. Elle fixe la base naïve et justifie la PR-AUC (classes déséquilibrées, jamais d'accuracy).
- **Entonnoir C1 → C1+C2 → C1+C2+C3** : parmi les fortes hausses, part de celles où l'open interest baisse (rachat des shorts) plutôt que monte (nouveaux longs). Cela vérifie que la définition isole bien le mécanisme visé.
- **Épisodes plutôt qu'heures** : avec la fenêtre de 24 h, un même squeeze marque jusqu'à 24 heures t de suite. On compte les épisodes distincts et leur durée.
- **Simultanéité** : part des squeezes qui touchent plusieurs cryptos à la même heure (effet de marché, souvent emmené par BTC).
- **Calendrier** : heure UTC, jour de la semaine, distance au prochain paiement de funding (00 h, 08 h, 16 h).
- **Profil d'un squeeze** : ampleur du pic, délai avant le pic, baisse de l'open interest.
- **Sensibilité** : fréquence des squeezes pour chaque variante (k, plancher, δ, horizon).
- **Exemples** : un ou deux squeezes connus en graphique (prix, open interest, funding, ratio long/short sur ±3 jours).

### Étape 4 : positionnement et squeezes, première réponse à H1/H2

- **Taux de squeeze par décile** de chaque variable (funding, variation d'open interest sur 24 h et 7 jours, ratios long/short, basis, volume perpétuel/spot). Selon H2, le taux monte quand les shorts sont nombreux : funding négatif, ratio bas, open interest en hausse.
- **Rendement à 24 h par décile** (quantiles 5 %, 50 % et 95 %). Selon H1, le rendement baisse quand les shorts sont nombreux. Les deux hypothèses ne s'excluent pas : H1 porte sur le centre de la distribution, H2 sur sa queue droite.
- **Étude d'événement** : moyenne de chaque variable de t−72 h à t+24 h autour des squeezes, comparée aux hausses C1 sans baisse d'open interest et aux heures ordinaires.
- **Croisement** : taux de squeeze selon le funding (négatif ou non) et l'open interest (en hausse ou non), en heatmap.
- **Variables de prix seules** (rendement passé, volatilité, drawdown) : même exercice, plus leur corrélation avec le positionnement. Cela prépare la comparaison avec le modèle « prix seuls » et dit si les deux apportent la même information.

### Étape 5 : hétérogénéité et robustesse

- **Par crypto** (BTC et ETH contre petites cryptos), **par année** et **par niveau de Fear & Greed**.
- **Binance contre Bybit** sur BTC, ETH et SOL : retrouve-t-on les mêmes faits ? C3 ne peut pas être refaite sur Bybit.

### Précautions

- **Période** : tout calculer sur `panel.loc[START:]`, sans la chauffe.
- **Tests statistiques** : les fenêtres de 24 h se chevauchent, donc la cible est très autocorrélée. Un t-test ou un χ² naïf surestime la significativité. On s'en tient aux graphiques et aux intervalles, ou on raisonne en épisodes.
- **Pas de regard sur la période de test** : si l'analyse sert à choisir des seuils ou des variables, elle se fait seulement sur la période d'entraînement.

### Priorités

Les étapes 1 et 2 se font dès maintenant ; les étapes 3 à 5 attendent `cible.py`. Si le temps manque, quatre graphiques relient le plus directement les données à la question :
1. la fréquence de la cible, comptée en épisodes ;
2. l'entonnoir C1 → C3 ;
3. le taux de squeeze par décile des variables de positionnement ;
4. l'étude d'événement autour des squeezes.

## Répartition du travail

| Période | A | B | C |
|---|---|---|---|
| Début octobre | dépôt, collecte Binance | Bybit et Fear & Greed | revue de littérature |
| Mi à fin octobre | panel horaire | cible | variables |
| Début novembre | analyse descriptive (en commun) | | |
| Mi-novembre | logit et base naïve | gradient boosting | stratégie et frais |
| Fin novembre | robustesse | rédaction | reproductibilité |

Méthode de travail : une branche et une pull request par tâche.
