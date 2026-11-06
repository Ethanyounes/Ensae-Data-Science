"""Construit les panels Binance, Bybit et leur fusion dans data/interim/, et affiche les manquants.

uv run python -m scripts.construire_panel
"""

import pandas as pd

from src.collecte_fng import FNG_FILE
from src.config import DEBUT_DONNEES, END, START, SYMBOLS
from src.panel_binance import PANEL_FILE, build_panel, rapport_qualite
from src.panel_bybit import (PANEL_FUSION_FILE, build_panel_bybit, controles_bybit,
                             extremes_bybit, fusionner)
from src.panel_fng import controles_fng

if __name__ == "__main__":
    print("Contrôles de l'indice Fear & Greed (période du panel, chauffe comprise) :")
    print(controles_fng(pd.read_parquet(FNG_FILE), DEBUT_DONNEES, END).to_string())
    print()

    panel_binance = build_panel(SYMBOLS)
    print(f"Panel Binance : {len(panel_binance)} lignes, {panel_binance['symbol'].nunique()} cryptos -> {PANEL_FILE}")
    print("\nValeurs manquantes (%) :")
    print(rapport_qualite(panel_binance).T.to_string())

    panel_bybit = build_panel_bybit(SYMBOLS)
    print(f"\nPanel Bybit : {len(panel_bybit)} lignes, {panel_bybit['symbol'].nunique()} cryptos")
    print(rapport_qualite(panel_bybit).T.to_string())
    print("\nContrôles de cohérence Bybit (nombre d'heures en défaut) :")
    print(controles_bybit(panel_bybit).to_string())
    print("\nValeurs extrêmes Bybit sur la période d'étude (gardées) :")
    print(extremes_bybit(panel_bybit.loc[START:]).to_string())

    fusion = fusionner(panel_binance, panel_bybit)
    print(f"\nFusion : {len(fusion)} lignes, {fusion.shape[1]} colonnes -> {PANEL_FUSION_FILE}")
