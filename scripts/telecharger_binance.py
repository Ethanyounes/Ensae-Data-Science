"""Télécharge toutes les archives Binance du projet dans data/raw/binance/.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_binance
Le script peut être relancé sans risque : les fichiers déjà présents sont sautés.
"""

from src.collecte_binance import BINANCE_KINDS, download_binance
from src.config import END, START, SYMBOLS

if __name__ == "__main__":
    for symbol in SYMBOLS:
        for kind in BINANCE_KINDS:
            bilan = download_binance(symbol, kind, START, END)
            print(f"{symbol:9} {kind:19} {bilan}")
