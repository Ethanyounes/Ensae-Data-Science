"""Télécharge les archives Binance de toutes les cryptos.

uv run python -m scripts.telecharger_binance
"""

from src.collecte_binance import BINANCE_KINDS, download_binance
from src.config import DEBUT_DONNEES, END, SYMBOLS

if __name__ == "__main__":
    for symbol in SYMBOLS:
        for kind in BINANCE_KINDS:
            bilan = download_binance(symbol, kind, DEBUT_DONNEES, END)
            print(f"{symbol:9} {kind:19} {bilan}")
