"""Chemins, période d'étude, cryptos et adresses utilisés par tout le projet."""

import os
from datetime import date, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"          # fichiers téléchargés et jamais modifiés
INTERIM_DIR = DATA_DIR / "interim"  # panels horaires

# Début en 2023 : avant mi-décembre 2022, les ratios des gros traders manquent (voir README)
START = "2023-01-01"
END = "2026-09-30"

# Jours avant START dont ont besoin les fenêtres glissantes (σ sur 30 jours de la cible) :
# ils sont dans le panel mais pas dans l'analyse
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


BINANCE_BASE_URL = "https://data.binance.vision/data"
BYBIT_BASE_URL = "https://api.bybit.com"
FNG_URL = "https://api.alternative.me/fng/"

# Stockage S3 du SSPCloud : le dossier diffusion/ est lisible sans identifiants
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
S3_BUCKET = "ethany1"
S3_PREFIX = "diffusion/ensae-squeeze"
S3_PUBLIC_URL = f"{S3_ENDPOINT}/{S3_BUCKET}/{S3_PREFIX}"
