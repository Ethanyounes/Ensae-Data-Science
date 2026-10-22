"""Configuration du projet : chemins des données, période d'étude et univers de cryptos.

Tous les modules importent leurs chemins et leurs paramètres d'ici, pour qu'un
changement (de dossier, de période ou de cryptos) se fasse en une seule ligne.
"""

import os
from pathlib import Path

# Racine du dépôt, calculée à partir de ce fichier (src/config.py) et non du
# dossier courant : le notebook et les scripts trouvent ainsi les mêmes fichiers.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Dossier des données, remplaçable par la variable d'environnement DATA_DIR
# (utile sur le SSPCloud). Path(...) car os.environ renvoie une chaîne.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"              # fichiers téléchargés, jamais modifiés
INTERIM_DIR = DATA_DIR / "interim"      # panel horaire nettoyé et fusionné
PROCESSED_DIR = DATA_DIR / "processed"  # panel + cible + variables

# Période d'étude (dates incluses, UTC)
START = "2022-01-01"
END = "2026-09-30"

# Univers de cryptos (contrats perpétuels USDT de Binance).
# À compléter jusqu'à 25 une fois le critère de sélection choisi en groupe.
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def raw_path(source: str, kind: str, symbol: str) -> Path:
    """Dossier des données brutes d'une source, d'un type et d'une crypto.

    Exemple : raw_path("binance", "metrics", "BTCUSDT")
    -> data/raw/binance/metrics/BTCUSDT
    Le dossier n'est pas créé ici : c'est à la fonction qui écrit de le faire.
    """
    return RAW_DIR / source / kind / symbol


# Adresses des sources (sans "/" final)
BINANCE_BASE_URL = "https://data.binance.vision/data"
BYBIT_BASE_URL = "https://api.bybit.com"
FNG_URL = "https://api.alternative.me/fng/"


# Stockage S3 du SSPCloud. Le dossier diffusion/ d'un bucket est lisible publiquement :
# le notebook y lit les données sans identifiants. L'écriture, elle, demande les clés
# du fichier .env (jamais commité).
# "or" plutôt qu'une valeur par défaut : la variable peut exister mais être vide
_endpoint = os.environ.get("AWS_S3_ENDPOINT") or "minio.lab.sspcloud.fr"
S3_ENDPOINT = "https://" + _endpoint.removeprefix("https://")
S3_BUCKET = "ethany1"                  # identifiant SSPCloud du bucket de référence du groupe
S3_PREFIX = "diffusion/ensae-squeeze"  # dossier du projet dans le bucket
S3_PUBLIC_URL = f"{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}"
