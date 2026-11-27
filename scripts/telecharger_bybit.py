"""Télécharge les séries Bybit de toutes les cryptos.

uv run python -m scripts.telecharger_bybit
"""

import pandas as pd

from src.collecte_bybit import BYBIT_KINDS, bybit_file, download_bybit, save_bybit
from src.config import SYMBOLS

if __name__ == "__main__":
    for symbol in SYMBOLS:
        for kind in BYBIT_KINDS:
            if bybit_file(kind, symbol).exists():
                df = pd.read_parquet(bybit_file(kind, symbol))
            else:
                try:
                    df = download_bybit(kind, symbol)
                except RuntimeError as erreur:
                    print(f"{symbol:9} {kind:14} ignoré : {erreur}")
                    continue
                save_bybit(df, kind, symbol)

            if df.empty:
                print(f"{symbol:9} {kind:14} aucune donnée sur la période")
            else:
                print(f"{symbol:9} {kind:14} {len(df):6} lignes, {df.iloc[0, 0]} -> {df.iloc[-1, 0]}")
