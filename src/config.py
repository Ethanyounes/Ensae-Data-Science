"""Chemins, période d'étude, cryptos et adresses utilisés par tout le projet."""

import os
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent  # racine du dépôt, d'où que l'on lance le code

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"              # fichiers téléchargés, jamais modifiés
INTERIM_DIR = DATA_DIR / "interim"      # panel horaire
PROCESSED_DIR = DATA_DIR / "processed"  # panel + cible + variables

# Début en 2023 : avant mi-décembre 2022, les ratios des gros traders manquent (voir README)
START = "2023-01-01"
END = "2026-09-30"

# Période de chauffe : les jours avant START dont ont besoin les fenêtres glissantes
# (sigma sur 30 jours dans la cible). Ces heures sont dans le panel mais pas dans l'analyse.
CHAUFFE_JOURS = 30
DEBUT_DONNEES = str(date.fromisoformat(START) - timedelta(days=CHAUFFE_JOURS))

# Les 10 perpétuels USDT au plus gros open interest en décembre 2022 (règle : scripts/choisir_cryptos.py)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "DOGEUSDT", "LTCUSDT",
           "XRPUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT", "ADAUSDT"]
# Les étapes 1 à 3 du rapport décrivent en détail les 3 plus grosses ; les étapes 4 et 5 utilisent les 10
SYMBOLS_DESCRIPTIF = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def raw_path(source: str, kind: str, symbol: str) -> Path:
    """Dossier des données brutes, par exemple data/raw/binance/metrics/BTCUSDT."""
    return RAW_DIR / source / kind / symbol


# Contrats dont le nom diffère chez Bybit (relevé le 2026-11-02 sur les API des deux plateformes).
# Le facteur est la quantité de crypto d'un contrat Bybit divisée par celle d'un contrat Binance :
# 1000SATSUSDT vaut 1 000 SATS, 10000SATSUSDT en vaut 10 000, donc son prix est 10 fois plus haut.
BYBIT_NOMS = {
    "1000SATSUSDT": ("10000SATSUSDT", 10),
    "1000SHIBUSDT": ("SHIB1000USDT", 1),
    "TAGUSDT": ("1000TAGUSDT", 1000),
    "TOSHIUSDT": ("1000TOSHIUSDT", 1000),
    "TURBOUSDT": ("1000TURBOUSDT", 1000),
}


def nom_bybit(symbol: str) -> tuple[str, int]:
    """Nom Bybit d'un contrat Binance et facteur de quantité (le même nom et 1 le plus souvent)."""
    return BYBIT_NOMS.get(symbol, (symbol, 1))


BINANCE_BASE_URL = "https://data.binance.vision/data"
BYBIT_BASE_URL = "https://api.bybit.com"
FNG_URL = "https://api.alternative.me/fng/"

# Stockage S3 du SSPCloud : le dossier diffusion/ est lisible par tous, sans identifiants
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
S3_BUCKET = "ethany1"
S3_PREFIX = "diffusion/ensae-squeeze"
S3_PUBLIC_URL = f"{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}"
