# Le positionnement des traders annonce-t-il les short squeezes ?

Projet de Python pour la data science, ENSAE 2A, 2026 : Emmanuelle Chocron, Maxime Duvivier, Ethan Younes.

Le rapport complet, avec le code, les graphiques et leur commentaire, est le notebook [`rapport.ipynb`](rapport.ipynb).

## Question

Sur les plateformes crypto, l'essentiel des échanges porte sur des **contrats perpétuels** : des contrats à terme sans échéance qui permettent de parier à la hausse ou à la baisse avec un effet de levier. Un **short squeeze** est une forte hausse du prix qui force les vendeurs à découvert (*shorts*) à racheter leurs positions, ce qui accélère encore la hausse.

**Question** : sur les perpétuels USDT de Binance, le positionnement des traders (funding, open interest, ratios long/short, ratio taker) annonce-t-il un short squeeze dans les 24 heures qui suivent ?

- **H1, shorts informés** : une accumulation de shorts annonce une baisse, et le positionnement n'annonce pas de squeeze.
- **H2, shorts fragiles** : des shorts accumulés avec du levier rendent le marché vulnérable à un rachat forcé.

On répond par l'analyse descriptive, sans modèle de prévision (non demandé pour l'ENSAE).

## Réponse en bref

- **H2 est soutenue, pas H1.** Le rendement médian des 24 heures suivantes ne baisse pas quand les shorts sont nombreux. En revanche, avant un squeeze, des shorts s'accumulent pendant une baisse du prix en payant le funding, puis sont rachetés pendant la hausse.
- **Trois signaux** : un funding négatif (nettement plus bas que d'habitude un à trois jours avant le pic), une forte hausse de l'open interest et des gros traders placés à la baisse. Selon le signal, le groupe de cryptos et la période, les squeezes sont 1,6 à 6,7 fois plus fréquents quand le signal est présent. Un funding négatif et un open interest en hausse réunis multiplient le taux de squeeze par presque quatre (intervalle de confiance à 95 % : [1,6 ; 8,7]).
- **Robustesse** : les signaux vont dans le même sens pour les grosses et les petites cryptos, en 2023-2024 comme en 2025-2026, et le mécanisme se retrouve sur Bybit. Le ratio des gros traders doit toutefois être mesuré par rapport à son niveau récent.
- **Limites** : 86 épisodes, mais seulement 34 événements de marché distincts, car un squeeze touche souvent plusieurs cryptos à la fois. Les intervalles de confiance sont donc larges.

## Données

| Source | Contenu | Fréquence | Accès |
|---|---|---|---|
| **Binance**, archives publiques ([data.binance.vision](https://data.binance.vision)) | perpétuel : bougies, open interest, ratios long/short, ratio taker, funding, basis ; spot : bougies | 5 min, 1 h, 8 h | un fichier zip par jour ou par mois, avec sa somme de contrôle |
| **Bybit**, API publique v5 | bougies, open interest, funding | 1 h, 8 h | requêtes paginées, sans clé |
| **alternative.me**, API Fear & Greed | indice de sentiment du marché, de 0 (peur extrême) à 100 (avidité extrême) | quotidienne | une requête |

Binance fournit les données principales. Bybit sert de contrôle : les constats sont-ils propres aux clients de Binance ? Le Fear & Greed enrichit l'analyse : les squeezes dépendent-ils du sentiment du marché ?

**Univers : 10 cryptos** (`scripts/choisir_cryptos.py`). Parmi les 154 perpétuels USDT cotés avant décembre 2022, on garde les 10 au plus gros open interest moyen de décembre 2022 qui existent encore sous le même nom en septembre 2026, avec une paire spot du même nom : BTC, ETH, BNB, DOGE, LTC, XRP, SOL, DOT, LINK, ADA. Classer avant la période d'étude évite de choisir les cryptos qui ont monté ensuite. MATIC, 9e, est exclu : il a été renommé POL en 2024.

**Période : du 2023-01-01 au 2026-09-30**, en données horaires. Avant mi-décembre 2022, les ratios des gros traders manquent chez Binance. Le panel commence 30 jours plus tôt : ces jours de chauffe servent aux fenêtres glissantes (volatilité sur 30 jours de la cible) mais n'entrent pas dans l'analyse.

**Stockage.** Les données ne sont pas dans Git (`data/` est dans `.gitignore`). Les fichiers bruts et les panels sont déposés sur le stockage S3 du SSPCloud, dans un dossier public en lecture : `https://minio.lab.sspcloud.fr/ethany1/diffusion/ensae-squeeze/`, avec la même arborescence que `data/`. Le notebook lit chaque fichier en local s'il existe, sinon sur le S3 (`src.stockage.lire_donnees`), sans identifiants.

## Méthode

### Panel horaire et convention de temps

Les sources sont nettoyées (noms, types, dates UTC, valeurs impossibles, doublons) puis réunies en un panel d'une ligne par crypto et par heure t : 33 576 heures par crypto, chauffe comprise, et toutes les heures de la période, même sans données. Les valeurs manquantes ne sont jamais comblées.

**La ligne t ne contient que ce qui est connu à t**, pour qu'aucune variable ne voie le futur :

| Source | Contenu de la ligne t = 15:00 |
|---|---|
| bougies (perpétuel, spot, indice de prime) | la bougie 14:00-15:00 (t = `open_time` + 1 h) |
| relevés à 5 min : open interest | le dernier relevé de 14:05 à 15:00 |
| relevés à 5 min : ratios long/short et taker | la moyenne des relevés de 14:05 à 15:00 |
| funding | le dernier paiement effectué à 15:00 au plus tard |
| Fear & Greed | la valeur de la veille (celle du jour J n'est utilisée qu'à partir de J+1 00:00) |

### Cible (`src/cible.py`)

Une heure t est **positive** s'il y a un short squeeze dans les 24 heures qui suivent, c'est-à-dire si trois conditions sont réunies entre t et le **pic** (l'heure du plus haut des 24 heures suivantes) :

- **C1, forte hausse** : `max High(t, t+24h] / Close_t − 1 ≥ max(3·σ_30j·√24, 5 %)`, σ_30j étant l'écart-type des rendements horaires des 30 derniers jours ;
- **C2, rachat des shorts** : l'open interest baisse d'au moins δ entre t et le pic ;
- **C3, achats agressifs** : entre t et le pic, le volume des achats agressifs dépasse celui des ventes agressives.

`Close_t` et l'open interest de départ sont ceux de la ligne t, les derniers connus avant la fenêtre. L'étape 3 part de δ = 10 %. À partir de l'étape 4, δ vaut 1,5 écart-type des variations d'open interest sur 24 h des 30 derniers jours, ce qui donne à C2 la même sévérité pour chaque crypto. δ = 1,2 σ sert de contrôle. Les variantes testées sont k = 2,5 ou 4, un plancher de 3 ou 8 %, δ = 5 ou 15 % et un horizon de 12 ou 48 h.

Deux heures positives d'une même crypto à moins de 24 h d'écart appartiennent au même **épisode**, car leurs fenêtres se chevauchent.

### Analyse descriptive (`src/descriptif.py`)

1. **Données** : collecte, nettoyage, contrôle du panel et comparaison des 10 cryptos.
2. **Variables de positionnement** : distributions, évolution, corrélations et persistance (BTC, ETH, SOL).
3. **La cible** : fréquence, entonnoir C1 → C2 → C3, simultanéité, profil des épisodes et sensibilité aux seuils (BTC, ETH, SOL).
4. **Positionnement et squeezes** (10 cryptos) : étude d'événement autour des pics, taux de squeeze par décile de chaque variable, rendement à 24 h par décile (test de H1), croisement du funding et de l'open interest.
5. **Hétérogénéité et robustesse** : par crypto, par période, selon le Fear & Greed, et Binance contre Bybit.

Les fenêtres de 24 h se chevauchent et les squeezes de plusieurs cryptos sont souvent simultanés : les heures ne sont pas indépendantes. Les intervalles de confiance sont donc calculés par bootstrap en tirant des semaines entières.

## Reproduction

Avec [uv](https://docs.astral.sh/uv/) (Python 3.12) :

```bash
uv sync
uv run python -m nbconvert --to notebook --execute --inplace rapport.ipynb
```

Sans uv : `pip install -r requirements.txt`, puis ouvrir `rapport.ipynb` dans Jupyter. Le notebook lit les données sur le S3 et tourne en deux minutes environ.

Pour tout refaire depuis les sources (long : près de 16 000 archives Binance), mettre `RECONSTRUIRE = True` au début du notebook, ou lancer les scripts :

```bash
uv run python -m scripts.choisir_cryptos       # facultatif : retrouve la liste SYMBOLS de src/config.py (20 min)
uv run python -m scripts.telecharger_binance   # archives Binance (relançable)
uv run python -m scripts.telecharger_bybit     # séries Bybit
uv run python -m scripts.telecharger_fng       # indice Fear & Greed
uv run python -m scripts.construire_panel      # panels horaires -> data/interim/
```

Le dossier des données se change avec la variable d'environnement `DATA_DIR`. Pour déposer les données sur le S3, il faut des identifiants SSPCloud dans un fichier `.env`, jamais commité : `uv run --env-file .env python -m scripts.envoyer_s3`.

## Organisation du dépôt

```
rapport.ipynb            le rapport : récit, résultats et graphiques
src/
  config.py              chemins, cryptos, période, adresses
  collecte_binance.py    archives Binance : téléchargement, somme de contrôle, parquet
  collecte_bybit.py      API Bybit : funding, open interest, bougies
  collecte_fng.py        indice Fear & Greed
  panel_binance.py       nettoyage et panel horaire Binance
  panel_bybit.py         panel horaire Bybit, contrôles et fusion avec Binance
  panel_fng.py           nettoyage du Fear & Greed et rattachement au panel
  stockage.py            envoi sur le S3 et lecture publique
  cible.py               conditions C1, C2, C3 et squeeze
  descriptif.py          statistiques et graphiques du rapport
scripts/                 choix des cryptos, téléchargements, construction des panels, envoi sur le S3
data/                    non versionné
  raw/                   fichiers téléchargés, jamais modifiés
  interim/               panels horaires (Binance, Bybit, fusion)
```

## Travail en groupe

Une branche et une pull request par tâche.

| Période | A | B | C |
|---|---|---|---|
| Début octobre | dépôt, collecte Binance | Bybit et Fear & Greed | revue de littérature |
| Mi à fin octobre | panel horaire | cible | variables |
| Novembre | analyse descriptive (en commun) | | |
| Fin novembre | robustesse | rédaction | reproductibilité |
