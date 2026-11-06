"""Construit les panels Binance, Bybit et leur fusion dans data/interim/, et affiche les manquants.

uv run python -m scripts.construire_panel
"""

from src.config import SYMBOLS
from src.panel_binance import PANEL_FILE, build_panel, rapport_qualite
from src.panel_bybit import PANEL_FUSION_FILE, build_panel_bybit, fusionner

if __name__ == "__main__":
    panel_binance = build_panel(SYMBOLS)
    print(f"Panel Binance : {len(panel_binance)} lignes, {panel_binance['symbol'].nunique()} cryptos -> {PANEL_FILE}")
    print("\nValeurs manquantes (%) :")
    print(rapport_qualite(panel_binance).T.to_string())

    panel_bybit = build_panel_bybit(SYMBOLS)
    print(f"\nPanel Bybit : {len(panel_bybit)} lignes, {panel_bybit['symbol'].nunique()} cryptos")
    print(rapport_qualite(panel_bybit).T.to_string())

    fusion = fusionner(panel_binance, panel_bybit)
    print(f"\nFusion : {len(fusion)} lignes, {fusion.shape[1]} colonnes -> {PANEL_FUSION_FILE}")
