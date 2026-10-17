"""Télécharge les séries Bybit du projet dans data/raw/bybit/.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_bybit
Le script peut être relancé sans risque : les séries déjà présentes sont relues, pas retéléchargées.
"""

import pandas as pd

from src.collecte_bybit import BYBIT_KINDS, bybit_file, download_bybit, save_bybit
from src.config import SYMBOLS

if __name__ == "__main__":
    for symbol in SYMBOLS:
        for kind in BYBIT_KINDS:
            if bybit_file(kind, symbol).exists():  # reprise : déjà téléchargé
                df = pd.read_parquet(bybit_file(kind, symbol))
            else:
                df = download_bybit(kind, symbol)
                save_bybit(df, kind, symbol)
            print(f"{symbol:9} {kind:14} {len(df):6} lignes, "
                  f"{df.iloc[0, 0]} -> {df.iloc[-1, 0]}")
