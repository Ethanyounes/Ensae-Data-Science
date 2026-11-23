"""Calcule la cible (valeurs de base) et enregistre data/processed/panel_cible.parquet.

uv run python -m scripts.indicateur_shortsqueeze
"""

import pandas as pd

from src.cible import PANEL_CIBLE_FILE, ajouter_cible
from src.panel_binance import PANEL_FILE

if __name__ == "__main__":
    panel_cible = ajouter_cible(pd.read_parquet(PANEL_FILE))
    PANEL_CIBLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel_cible.to_parquet(PANEL_CIBLE_FILE)
    print(f"Panel cible -> {PANEL_CIBLE_FILE}")
    print(panel_cible.groupby("symbol")[["c1", "c2", "c3", "shortsqueeze"]].sum())
