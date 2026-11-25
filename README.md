# Le positionnement des traders annonce-t-il les short squeezes ?

Projet de Python pour la data science, ENSAE 2A, 2026 : Emmanuelle Chocron, Maxime Duvivier, Ethan Younes.

1. [Question et hypothèses](#question-et-hypothèses)
2. [Installation et reproduction](#installation-et-reproduction)
3. [Organisation du dépôt](#organisation-du-dépôt)
4. [Données](#données)
5. [Convention de temps](#convention-de-temps)
6. [Nettoyage et panel horaire](#nettoyage-et-panel-horaire)
7. [Cible](#cible-srcciblepy)
8. [Analyse descriptive](#analyse-descriptive)
9. [Répartition du travail](#répartition-du-travail)

## Question et hypothèses

**Question** : le positionnement des traders sur les perpétuels crypto (Binance USD-M) annonce-t-il les short squeezes, 24 heures à l'avance ?

- **H1, shorts informés** : une accumulation de shorts annonce une baisse, et le positionnement n'annonce pas de squeeze.
- **H2, shorts fragiles** : des shorts accumulés avec du levier rendent le marché vulnérable à un rachat forcé.

**Cible.** Il y a un short squeeze à l'heure t si les trois conditions suivantes sont réunies dans les 24 h qui suivent (valeurs de base) :
- **C1, forte hausse** : `max High(t, t+24h] / Close_{t−1} − 1 ≥ max(3·σ_30j·√24, 5 %)` ;
- **C2, rachat des shorts** : l'open interest au pic est inférieur d'au moins 10 % à `OI_{t−1}` ;
- **C3, achats agressifs** : le ratio taker moyen entre t et le pic est supérieur à 1.

Variantes testées : k = 2,5 ou 4 ; plancher de 3 ou 8 % ; δ = 5 ou 15 %, ou 1,2 ou 1,5 écart-type des variations d'open interest sur 24 h des 30 derniers jours ; horizon de 12 ou 48 h (le seuil devient alors `k·σ_30j·√horizon`). Le calcul est détaillé dans [Cible](#cible-srcciblepy).

**Cible retenue à partir de l'étape 4** : δ = 1,5 σ, avec δ = 1,2 σ en contrôle. Exprimer C2 en écarts-types, comme C1, lui donne la même sévérité pour chaque crypto, ce que ne fait pas un seuil fixe (voir la section 3 du rapport).

**Démarche.** Les consignes de l'ENSAE ne demandent pas de modèle : on répond à la question par l'analyse descriptive. Le positionnement change-t-il de façon visible avant les squeezes, et ces signaux tiennent-ils partout ? Le projet prévoyait au départ un modèle de prévision (PR-AUC face à une base naïve et à un modèle « prix seuls ») et un backtest après frais et funding. Ils restent des prolongements possibles, non réalisés.

## Installation et reproduction

On utilise [uv](https://docs.astral.sh/uv/) avec Python 3.12.

```bash
uv sync                                        # crée .venv avec les versions de uv.lock
uv run python -m scripts.choisir_cryptos       # facultatif : retrouve la liste SYMBOLS de config.py (20 min)
uv run python -m scripts.telecharger_binance   # collecte Binance (relançable)
uv run python -m scripts.telecharger_bybit     # collecte Bybit (moins de 5 min)
uv run python -m scripts.telecharger_fng       # indice Fear & Greed (une requête)
uv run python -m scripts.construire_panel      # panels horaires -> data/interim/
uv run python -m scripts.indicateur_shortsqueeze  # cible -> data/processed/ (facultatif : le rapport la recalcule)
uv run jupyter lab                             # ouvrir rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, généré par `uv export --no-hashes --no-dev > requirements.txt`.

**Données locales.** Les données ne sont pas versionnées (`data/` est dans `.gitignore`). Le dossier local se change avec la variable d'environnement `DATA_DIR`.

**Données en ligne.** Les données brutes et les panels sont déposés sur le stockage S3 du SSPCloud, dans un dossier public en lecture seule, avec la même arborescence que `data/` : `https://minio.lab.sspcloud.fr/ethany1/diffusion/ensae-squeeze/`.
- Le notebook les lit sans identifiants : `src.stockage.lire_donnees` lit chaque fichier en local s'il existe, sinon au même chemin sur le S3.
- Pour en déposer de nouvelles, il faut des identifiants SSPCloud dans un fichier `.env`, jamais commité : `uv run --env-file .env python -m scripts.envoyer_s3`. Le script n'envoie que les fichiers bruts absents du S3, et renvoie toujours les panels de `data/interim/`. Les identifiants SSPCloud sont temporaires : les régénérer depuis son compte s'ils sont refusés.

## Organisation du dépôt

```
src/
  config.py            chemins, cryptos, période, URL, S3            ← tout le monde importe d'ici
  collecte_binance.py  archives Binance : téléchargement, checksum, parquet
  collecte_bybit.py    API Bybit v5 : funding, open interest, bougies
  collecte_fng.py      indice Fear & Greed (alternative.me), quotidien
  stockage.py          envoi sur le S3 du SSPCloud et lecture publique
  panel_binance.py     nettoyage et panel horaire (une ligne par crypto et par heure t)
  panel_bybit.py       panel horaire Bybit, contrôles de cohérence et fusion avec le panel Binance
  panel_fng.py         nettoyage de l'indice Fear & Greed et rattachement au panel horaire
  cible.py             cible : conditions C1, C2, C3 et squeeze, seuils en paramètres
  descriptif.py        statistiques et graphiques descriptifs (étapes 1 à 3)
scripts/
  choisir_cryptos.py       applique la règle de choix des cryptos (open interest de décembre 2022)
  telecharger_binance.py   télécharge toutes les archives Binance (SYMBOLS × types)
  telecharger_bybit.py     télécharge toutes les séries Bybit
  telecharger_fng.py       télécharge l'indice Fear & Greed
  construire_panel.py      construit les panels Binance, Bybit et leur fusion, et affiche manquants et contrôles
  envoyer_s3.py            dépose les données brutes manquantes et les panels sur le S3 du SSPCloud
  indicateur_shortsqueeze.py  calcule la cible et enregistre data/processed/panel_cible.parquet
  descriptif.ipynb         exploration des premiers squeezes (brouillon de l'étape 3)
data/
  raw/binance/{metrics,fundingRate,klines,premiumIndexKlines,spot_klines}/<SYMBOLE>/*.parquet
  raw/bybit/{funding,open_interest,klines}/<SYMBOLE>/<SYMBOLE>-<série>.parquet
  raw/fng/
  interim/       panels horaires
  processed/     panel + cible (panel_cible.parquet) + variables
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

### Univers : 10 cryptos

Règle (`scripts/choisir_cryptos.py`) : parmi les 154 perpétuels USDT cotés avant décembre 2022, on classe les contrats par open interest moyen en USDT de décembre 2022, juste avant la période d'étude, et on garde les 10 premiers qui existent encore sous le même nom en septembre 2026 avec une paire spot du même nom. Classer avant la période évite de choisir les cryptos qui ont monté ensuite (biais du survivant).

| Rang | Contrat | Open interest déc. 2022 (M USDT) |
|---|---|---|
| 1-2 | BTCUSDT, ETHUSDT | 1 880, 1 470 |
| 3-7 | BNBUSDT, DOGEUSDT, LTCUSDT, XRPUSDT, SOLUSDT | 190, 140, 124, 108, 86 |
| 8-10 | DOTUSDT, LINKUSDT, ADAUSDT | 75, 65, 64 |

- **MATICUSDT** (70 M USDT, entre DOT et LINK) est exclu : renommé POL en 2024, il n'a plus de données sous ce nom.
- **L'open interest plutôt que le volume** : en décembre 2022, AXS et MASK avaient des volumes gonflés par des hausses spéculatives ponctuelles. L'open interest mesure les positions à levier ouvertes, au cœur de la question.
- **Usage** : les étapes 1 à 3 du rapport décrivent en détail BTC, ETH et SOL (`SYMBOLS_DESCRIPTIF`) ; les étapes 4 et 5 utilisent les 10. Bybit est collecté pour les 10, comme contrôle de fiabilité.

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

- **Horodatage en secondes** (millisecondes chez Binance et Bybit) : il est ramené en millisecondes dans le panel.
- **4 jours manquent** : du 2018-04-14 au 2018-04-16, et le 2024-10-26. Seul le dernier est dans la période d'étude (voir [son traitement dans le panel](#indice-fear--greed-dans-le-panel-srcpanel_fngpy)).
- **Risque de fuite d'information** : la valeur du jour J (datée 00:00 UTC) est publiée vers minuit UTC, sans garantie qu'elle soit connue à 00:00 précisément. Dans le panel, l'indice est donc **décalé d'un jour** : à l'heure t du jour J, on utilise la valeur de J−1.
- **`fng_classe`** (« Fear », « Greed »…) se déduit de `fng` : elle reste dans les données brutes mais ne sert pas de variable.

## Convention de temps

Le panel a une ligne par crypto et par heure t (UTC). **La ligne t contient uniquement ce qui est connu à l'instant t.** Toutes les sources sont ramenées à cette règle avant la fusion, et toute la suite du projet (cible, variables, analyse) s'appuie dessus.

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
| Fear & Greed | chaque heure reçoit la valeur de la veille (`fng`) et la date de cette valeur (`fng_date`) | pas de fuite d'information (voir [plus bas](#indice-fear--greed-dans-le-panel-srcpanel_fngpy)) |
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
| Fear & Greed | 0 % | le jour absent (2024-10-26) reçoit la dernière valeur connue |

Le funding est versé toutes les 8 h pour les trois cryptos sur toute la période. Entre 15 et 19 heures par crypto ont moins de 12 relevés.

### Indice Fear & Greed dans le panel (`src/panel_fng.py`)

L'indice est ajouté au panel Binance par `ajouter_fng`, appelée dans `build_panel_crypto`.
- **Nettoyage** : on garde la date et la valeur (`fng_classe` se déduit de `fng`). Une valeur hors de [0, 100] ou une date en double serait supprimée ; `controles_fng` n'en trouve aucune.
- **Décalage d'un jour** : la valeur datée du jour J devient connue à J+1 00:00 (`t_fng` = date + 1 jour). Chaque heure t reçoit ensuite la dernière valeur connue (`merge_asof`).
- **Jour manquant** : le 2024-10-26 n'a pas de valeur, donc les heures du 2024-10-27 reçoivent celle du 2024-10-25. Ce report n'invente pas de donnée : c'est la dernière valeur connue à t, comme pour le funding. `fng_date` indique de quel jour date la valeur utilisée.

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
- **Doublons** : une bougie ou un relevé d'open interest en double sur la même heure est supprimé.
- **Open interest nul ou négatif** : remplacé par une valeur manquante, comme chez Binance.
- **Crypto absente de Bybit** : absente du panel Bybit ; ses colonnes `_bybit` restent vides dans la fusion.

**Contrôles de cohérence** (`controles_bybit`, chauffe comprise). La fonction signale sans rien corriger. Résultat identique pour BTC, ETH et SOL, sur 33 576 heures chacune :

| Contrôle | Heures en défaut |
|---|---|
| bougie absente | 0 |
| open interest absent | 1 : la dernière heure de la grille (2026-10-01 00:00), après la fin du téléchargement |
| funding absent | 0 |
| ouverture ou clôture hors de l'intervalle [plus bas, plus haut] | 0 |
| prix nul ou négatif | 0 |
| volume nul | 0 |
| open interest identique 3 heures de suite (série figée) | 0 |

**Valeurs extrêmes** (`extremes_bybit`, période d'étude). Elles sont gardées : ce sont de vrais événements de marché, et un squeeze en est justement un.

| | Plus forte baisse horaire de l'OI | Plus forte variation horaire du prix | Funding horaire min / max | Heures à funding négatif |
|---|---|---|---|---|
| BTC | −25,5 % (2025-10-10 22:00) | 6,2 % (2023-04-26 20:00) | −0,0057 % / 0,0141 % | 16,1 % |
| ETH | −26,4 % (2025-10-10 22:00) | 11,9 % (2025-02-03 02:00) | −0,0147 % / 0,0121 % | 18,1 % |
| SOL | −22,7 % (2023-06-30 14:00) | 12,9 % (2023-07-14 01:00) | −0,1079 % / 0,0173 % | 26,9 % |

- **La baisse d'OI du 2025-10-10** sur BTC et ETH correspond au krach du 10 octobre 2025, une vague de liquidations de longs pendant une chute des prix. Une baisse d'OI seule ne signale donc pas un squeeze : il faut aussi la hausse (C1) et les achats agressifs (C3).
- **Le funding le plus négatif de SOL** (2023-01-04 08:00) tombe pendant sa hausse de 10 à 23,7 $ entre le 1er et le 14 janvier 2023 : les shorts payaient cher pour rester en position pendant que le prix montait. C'est un bon exemple de squeeze à montrer dans l'analyse descriptive.

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

## Cible (`src/cible.py`)

`ajouter_cible(panel, k=3, plancher=0.05, delta=0.10, horizon=24)` ajoute la cible au panel, crypto par crypto. Chaque variante change un seul paramètre, par exemple `ajouter_cible(panel, delta=0.05)`. Avec `delta_sigma=1.5`, C2 exige une baisse de 1,5 `sigma_oi_30j` au lieu de `delta`.

| Colonne | Contenu |
|---|---|
| `sigma_30j` | écart-type des rendements horaires (en log) de t−30 jours (exclu) à t |
| `sigma_oi_30j` | écart-type des variations relatives de l'open interest sur `horizon` heures, sur les 30 derniers jours |
| `date_pic`, `delai_pic_h` | heure du plus haut des `horizon` heures suivantes, et son délai après t |
| `hausse_pic` | (`high` au pic) / (`close` de la ligne t) − 1 |
| `variation_oi` | (open interest au pic) / (open interest de la ligne t) − 1 |
| `ratio_achats_ventes` | achats agressifs / ventes agressives, de t+1 h au pic (bougies) |
| `c1`, `c2`, `c3` | chaque condition séparément, pour l'entonnoir |
| `shortsqueeze` | C1 et C2 et C3 |

- **Calcul sans boucle** : le pic vient d'une fenêtre glissante sur les `high` des heures suivantes, et les volumes de t au pic de la différence de deux sommes cumulées. Le calcul prend moins d'une seconde, contre plusieurs dizaines de minutes avec une boucle sur les heures, ce qui permet de tester les variantes.
- **Sur le panel complet** : σ a besoin des 30 jours de chauffe, on garde ensuite la période d'étude (`descriptif.periode_etude`).
- **Fin de série** : les dernières heures n'ont pas assez de futur, leur cible est inconnue (`<NA>`) et non fausse.
- **Vérification** : sur les 13 820 heures où la hausse dépasse 5 %, les grandeurs intermédiaires sont identiques à celles de la première version (une boucle sur les heures). Seules 10 heures changent de classement, parce que C1 compare désormais la hausse simple au seuil, comme dans la définition, et non la hausse en log.
- **Épisodes** (`descriptif.tableau_episodes`) : deux heures positives d'une même crypto à moins d'un horizon d'écart appartiennent au même épisode, car leurs fenêtres se chevauchent. Compter les heures consécutives coupait un même squeeze en plusieurs épisodes dès qu'une heure manquait. Regrouper les heures par pic ne marche pas non plus : pendant une hausse qui dure, chaque heure voit un pic différent, une heure plus tard. Chaque épisode est décrit par son heure à la plus forte hausse.

`scripts/indicateur_shortsqueeze.py` enregistre le panel avec la cible dans `data/processed/panel_cible.parquet`. Le rapport n'en a pas besoin : il recalcule la cible.

## Analyse descriptive

**Objectif.** Montrer que la cible est rare et groupée dans le temps, décrire les variables de positionnement, puis voir si le positionnement diffère déjà avant un squeeze. Cette dernière étape, complétée par l'étape 5, donne la réponse du projet à H1 contre H2, sans modèle. Le code est dans `src/descriptif.py`, appelé par `rapport.ipynb`. Chaque graphique y est commenté et relié à la question.

| Étape | Question | Données nécessaires |
|---|---|---|
| 1. Données et univers | Les données sont-elles fiables, et les cryptos comparables ? | panel actuel |
| 2. Variables de positionnement | Comment se comportent funding, open interest et ratios ? | panel actuel |
| 3. La cible | Combien de squeezes, quand, et de quelle forme ? | `cible.py` |
| 4. Positionnement et squeezes | Le positionnement diffère-t-il avant un squeeze ? (H1/H2) | `cible.py` |
| 5. Hétérogénéité et robustesse | Les constats tiennent-ils partout ? | `cible.py`, Bybit, Fear & Greed |

### Étape 1 : données et univers

- **Carte des valeurs manquantes** par crypto et par mois (`rapport_qualite`, sur BTC, ETH et SOL ; les 10 cryptos sont contrôlées au début de l'étape 4) : repérer les cryptos cotées après janvier 2023 et les trous de relevés.
- **Cohérence du ratio taker** : recalculer le ratio à 1 h depuis les bougies (`taker_buy_volume / (volume − taker_buy_volume)`) et le comparer à `taker_ratio`, issu des relevés à 5 min. Ce contrôle croise deux sources Binance et donne une seconde mesure possible pour C3.
- **Tableau par crypto** : volume moyen en USDT, open interest moyen en USD, volatilité annualisée, rendement total, funding moyen annualisé, part des heures à funding négatif.
- **Distribution des rendements horaires** (kurtosis, QQ-plot) : queues épaisses et volatilités très différentes d'une crypto à l'autre, d'où un seuil C1 exprimé en σ.
- **Volume du perpétuel rapporté au volume spot** (`quote_volume / spot_quote_volume`) : part de la spéculation à levier, par crypto et dans le temps.

### Étape 2 : variables de positionnement

- **Distributions** : funding (forte masse attendue au taux par défaut de Binance, 0,01 % par 8 h), ratios long/short, ratio taker (valeurs extrêmes).
- **Séries 2023-2026**, avec les grandes phases du marché. Les niveaux changent d'une crypto à l'autre et dans le temps, d'où des z-scores glissants par crypto plutôt que des niveaux bruts.
- **Corrélations entre variables** (heatmap) : les trois ratios long/short sont sans doute très corrélés, ce qui poserait un problème de multicolinéarité dans une régression.
- **Persistance** (autocorrélation) : le funding et l'open interest évoluent lentement, le ratio taker très vite.

### Étape 3 : la cible

- **Fréquence des squeezes** : globale, par crypto, par année. Elle fixerait la base naïve d'un modèle de prévision (PR-AUC, classes déséquilibrées, jamais d'accuracy).
- **Entonnoir C1 → C1+C2 → C1+C2+C3** : parmi les fortes hausses, part de celles où l'open interest baisse (rachat des shorts) plutôt que monte (nouveaux longs). Cela vérifie que la définition isole bien le mécanisme visé.
- **Épisodes plutôt qu'heures** : avec la fenêtre de 24 h, un même squeeze marque jusqu'à 24 heures t de suite. On compte les épisodes distincts et leur durée.
- **Simultanéité** : part des squeezes qui touchent plusieurs cryptos à la même heure (effet de marché, souvent emmené par BTC).
- **Calendrier** : heure UTC, jour de la semaine, distance au prochain paiement de funding (00 h, 08 h, 16 h).
- **Profil d'un squeeze** : ampleur du pic, délai avant le pic, baisse de l'open interest.
- **Sensibilité** : fréquence des squeezes pour chaque variante (k, plancher, δ, horizon).
- **Exemples** : un ou deux squeezes connus en graphique (prix, open interest, funding, ratio long/short sur ±3 jours). Candidat : SOL du 1er au 14 janvier 2023 (voir les [valeurs extrêmes Bybit](#panel-bybit-et-fusion-srcpanel_bybitpy)).

**Fait** (section 3 de `rapport.ipynb`). Principaux constats, avec les valeurs de base :
- **16 épisodes** (BTC 4, ETH 4, SOL 8), soit 0,13 à 0,31 % des heures : ce serait la base naïve d'un modèle de prévision.
- **13 épisodes sur 16 en 2023**, aucun en 2025 et un seul en 2026. Les comparaisons dans le temps reposent donc sur peu d'épisodes récents.
- **Entonnoir** : plus de la moitié des fortes hausses (C1) se font avec un open interest en hausse, donc de nouveaux longs. C2 ne garde que 6 à 13 % des heures C1. C3 n'enlève presque rien.
- **Simultanéité** : 7 épisodes sur 16 tombent dans trois mouvements de marché (13-14 janvier 2023, mars 2023 après SVB, fin février 2024).
- **Calendrier** : aucun effet visible de l'heure, et trop peu d'épisodes pour conclure sur le jour de la semaine. Retiré du rapport pour rester à 12 graphiques.
- **Sensibilité** : δ est le seuil décisif (avec δ = 5 %, environ trois fois plus d'épisodes ; avec δ = 15 %, aucun pour BTC). Le plancher de 5 % ne joue presque jamais. L'étape 4 utilise δ = 1,5 écart-type des variations d'open interest, de même sévérité pour chaque crypto, avec δ = 1,2 σ en contrôle.

### Étape 4 : positionnement et squeezes, première réponse à H1/H2

Sur les 10 cryptos, avec δ = 1,5 σ.

- **Contrôle des données avec Bybit** : accord des prix, de l'open interest et du funding entre les deux plateformes, et, pour chaque épisode, baisse de l'open interest Bybit entre t et le pic. Une baisse vue sur les deux plateformes est un vrai rachat de positions, pas une erreur de relevé.
- **Taux de squeeze par décile** de chaque variable (funding, variation d'open interest sur 24 h et 7 jours, ratios long/short, basis, volume perpétuel/spot). Selon H2, le taux monte quand les shorts sont nombreux : funding négatif, ratio bas, open interest en hausse.
- **Rendement à 24 h par décile** (quantiles 5 %, 50 % et 95 %). Selon H1, le rendement baisse quand les shorts sont nombreux. Les deux hypothèses ne s'excluent pas : H1 porte sur le centre de la distribution, H2 sur sa queue droite.
- **Étude d'événement** : moyenne de chaque variable de t−72 h à t+24 h autour des squeezes, comparée aux hausses C1 sans baisse d'open interest et aux heures ordinaires.
- **Croisement** : taux de squeeze selon le funding (négatif ou non) et l'open interest (en hausse ou non), en heatmap.
- **Variables de prix seules** (rendement passé, volatilité, drawdown) : même exercice, plus leur corrélation avec le positionnement. Cela dit si le positionnement et le prix apportent la même information.

**Fait** (section 4 de `rapport.ipynb`, variables dans `descriptif.variables_etape_4`). Principaux constats :
- **86 épisodes, mais 34 événements de marché distincts** (épisodes de cryptos différentes dont les pics sont à moins de 24 h), dont 37 épisodes en 2025-2026 grâce aux nouvelles cryptos.
- **Étude d'événement** : avant un squeeze, le prix baisse, l'open interest monte (+2,5 %), le funding est négatif et les gros traders se placent à la baisse, un à trois jours avant le pic ; puis l'open interest chute de 7 % pendant la hausse. Les fortes hausses sans rachat ont le profil inverse (open interest et funding en hausse). Avec les bandes de confiance à 95 % (`bande_etude_evenement`, bootstrap par semaines du pic), seul le funding se sépare nettement des heures ordinaires avant le pic (de 72 h à 8 h avant) ; la baisse de l'open interest n'est nette que dans les 9 à 12 dernières heures.
- **Déciles** : le ratio des gros traders en positions donne le gradient le plus net (28 % des épisodes dans son premier décile) et n'est presque pas corrélé au prix. Le funding négatif et l'open interest en hausse, réunis, multiplient le taux de squeeze par presque quatre (`rapports_croisement` : 3,9, intervalle [1,6 ; 8,7]).
- **Le prix compte beaucoup** : 42 % des épisodes commencent après les pires rendements sur 7 jours. Une partie du signal du positionnement est donc déjà dans le prix.
- **Entonnoir avec la cible finale** : l'open interest monte pendant 74 % des fortes hausses des sept petites cryptos (57 % pour BTC, ETH et SOL). C3 n'écarte que 2 % des heures C1 + C2 des trois grosses, mais 49 % de celles des sept autres.
- **H1 n'est pas soutenue** (rendement médian inchangé), **H2 l'est**. Le lien avec la volatilité est en partie mécanique (C1 exprimée en σ).
- **Limite** : ces constats portent sur toute la période 2023-2026 et ont servi à choisir δ = 1,5 σ et la variable clé. L'étape 5 vérifie qu'ils tiennent sur chaque sous-période.

### Étape 5 : hétérogénéité et robustesse

- **Par crypto** (BTC et ETH contre petites cryptos), **par année** et **par niveau de Fear & Greed**.
- **Binance contre Bybit** sur les 10 cryptos : retrouve-t-on les mêmes faits ? C3 ne peut pas être refaite sur Bybit.
- **δ = 1,2 σ** : les constats de l'étape 4 tiennent-ils avec une cible moins stricte, qui compte plus d'épisodes en 2025-2026 ?

**Fait** (section 5 de `rapport.ipynb`). Les trois signaux de H2 de l'étape 4 : gros traders placés à la baisse (ratio en positions, déciles 1-2), funding négatif, forte hausse de l'open interest sur 24 h (déciles 9-10). Le contrôle δ = 1,2 σ est à la fin de l'étape 4.
- **Par crypto et dans le temps** (`effet_des_signaux`) : le funding négatif (taux de squeeze × 1,8 à 2,7) et la hausse de l'open interest (× 1,6 à 3,7) vont dans le même sens pour les grosses et les petites cryptos, en 2023-2024 comme en 2025-2026. Le ratio des gros traders est le plus fort (× 5,3 à 6,7) mais **s'inverse en 2025-2026 en niveau brut** (× 0,3) : son niveau a monté et les seuils de 2023-2026 ne sont presque plus atteints. Avec des déciles recalculés dans chaque période (`effet_selon_deciles`), il tient (× 4,4 puis × 3,0). Il faut donc le mesurer en relatif (z-scores glissants).
- **Intervalles de confiance à 95 %** (`intervalle_bootstrap`) : bootstrap par blocs d'une semaine, toutes cryptos ensemble (2 000 tirages), car les heures d'un même squeeze et les squeezes simultanés ne sont pas indépendants. L'intervalle du ratio des gros traders reste au-dessus de 1 dans tous les groupes (en relatif) ; ceux du funding et de l'open interest ne le sont que dans deux groupes sur quatre chacun.
- **Fear & Greed** (`squeezes_par_classe_fng`) : 65 % des épisodes commencent en phase de peur, qui ne couvre que 36 % des heures. Mais l'indice suit surtout le funding (corrélation 0,50) et le prix.
- **Binance contre Bybit** (`cible_bybit`, C1 + C2 seulement) : autant d'épisodes sur les deux plateformes, 6 sur 10 se retrouvent sur l'autre, et l'étude d'événement donne des profils presque superposés : le mécanisme n'est pas propre à Binance.

### Graphiques du rapport : 12 au maximum

Le rapport se limite à 12 graphiques, chacun commenté ; le reste est présenté en tableaux. Les étapes 1 à 4 en utilisent 9, l'étape 5 en utilise 3.

| Étape | Graphiques |
|---|---|
| 1. Données et univers | aucun : valeurs manquantes, ratio taker, queues des rendements et volume perpétuel / spot sont en tableaux |
| 2. Positionnement | séries 2023-2026 ; corrélations entre variables |
| 3. La cible | QQ-plots des rendements et de l'open interest (seuils de C1 et C2 en σ) ; deux exemples de squeeze, BTC et SOL |
| 4. Positionnement et squeezes | épisodes par année ; entonnoir (BTC, ETH, SOL contre les 7 autres) ; étude d'événement ; part des épisodes par décile (positionnement et prix) ; rendement à 24 h par décile (H1). Le taux de squeeze par décile, en heures, est en tableau |
| 5. Hétérogénéité et robustesse | effet des signaux par groupe (cryptos, périodes) ; squeezes par classe de Fear & Greed ; étude d'événement Binance contre Bybit |

### Précautions

- **Période** : tout calculer sur `panel.loc[START:]`, sans la chauffe.
- **Tests statistiques** : les fenêtres de 24 h se chevauchent, donc la cible est très autocorrélée. Un t-test ou un χ² naïf surestime la significativité. On s'en tient aux graphiques et aux intervalles (bootstrap par semaines, étape 5), ou on raisonne en épisodes.
- **Choix faits sur toute la période** : δ = 1,5 σ et la variable clé ont été choisis en regardant 2023-2026. Pour une analyse descriptive, ce n'est pas un problème, et l'étape 5 vérifie que les signaux tiennent sur chaque sous-période. Un modèle de prévision devrait en revanche refaire ces choix sur sa seule période d'entraînement (par exemple `part_episodes_par_decile` sur `panel.loc[:fin_entrainement]`).

### Priorités

Les cinq étapes sont faites. Si le temps manque, quatre graphiques relient le plus directement les données à la question :
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
| Mi-novembre | ~~logit et base naïve~~ | ~~gradient boosting~~ | ~~stratégie et frais~~ |
| Fin novembre | robustesse | rédaction | reproductibilité |

La modélisation prévue mi-novembre a été abandonnée : elle n'est pas demandée pour l'ENSAE, et le temps a été consacré à l'analyse descriptive (étapes 3 à 5) et à la réduction du rapport à 12 graphiques.

Méthode de travail : une branche et une pull request par tâche.
