"""Panel horaire Bybit (même convention de temps que Binance) et fusion avec le panel Binance."""

import pandas as pd

from src.collecte_bybit import bybit_file
from src.config import INTERIM_DIR, nom_bybit
from src.panel_binance import clean_funding, grille_horaire

PANEL_BYBIT_FILE = INTERIM_DIR / "panel_bybit.parquet"
PANEL_FUSION_FILE = INTERIM_DIR / "panel_binance_bybit.parquet"


def build_panel_bybit_crypto(symbol: str) -> pd.DataFrame:
    """Bougies, open interest et funding Bybit d'une crypto, sur la grille horaire."""
    facteur = nom_bybit(symbol)[1]

    bougies = pd.read_parquet(bybit_file("klines", symbol))
    bougies["t"] = (bougies["open_time"] + pd.Timedelta("1h")).dt.as_unit("ms")
    bougies = bougies.drop(columns="open_time").set_index("t")
    # Un contrat Bybit peut contenir facteur fois plus de crypto : on ramène à l'unité Binance
    bougies[["open", "high", "low", "close"]] /= facteur
    bougies["volume"] *= facteur

    # Relevé instantané, connu à son heure (vérifié en le comparant à l'open interest Binance).
    # single_open_interest compte un seul côté du marché, comme l'open interest de Binance.
    oi = pd.read_parquet(bybit_file("open_interest", symbol))
    oi["t"] = oi["create_time"].dt.as_unit("ms")
    oi = oi.set_index("t")[["single_open_interest"]].rename(columns={"single_open_interest": "oi"})
    oi["oi"] *= facteur

    panel = pd.DataFrame(index=grille_horaire()).join(bougies).join(oi)

    funding = clean_funding(pd.read_parquet(bybit_file("funding", symbol)))
    panel = pd.merge_asof(panel.reset_index(), funding, left_on="t", right_on="t_funding")
    panel = panel.set_index("t")

    panel.insert(0, "symbol", symbol)
    return panel


def build_panel_bybit(symbols: list[str]) -> pd.DataFrame:
    """Panel Bybit des cryptos téléchargées, enregistré dans data/interim/panel_bybit.parquet."""
    presentes = [s for s in symbols if bybit_file("klines", s).exists()]
    panel = pd.concat([build_panel_bybit_crypto(s) for s in presentes])
    PANEL_BYBIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_BYBIT_FILE)
    return panel


def fusionner(panel: pd.DataFrame, panel_bybit: pd.DataFrame) -> pd.DataFrame:
    """Panel Binance complété des colonnes Bybit (suffixe _bybit), enregistré dans data/interim/."""
    bybit = panel_bybit.drop(columns="symbol").add_suffix("_bybit")
    bybit["symbol"] = panel_bybit["symbol"]

    fusion = panel.reset_index().merge(bybit.reset_index(), on=["t", "symbol"], how="left")
    fusion = fusion.set_index("t")
    fusion.to_parquet(PANEL_FUSION_FILE)
    return fusion
