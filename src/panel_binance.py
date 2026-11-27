"""Nettoyage des données Binance et construction du panel horaire.

Une ligne par crypto et par heure t (UTC), qui ne contient que ce qui est connu à t
(voir « Convention de temps » dans le README).
"""

import pandas as pd

from src.collecte_binance import load_binance
from src.config import DEBUT_DONNEES, END, INTERIM_DIR
from src.panel_fng import ajouter_fng

PANEL_FILE = INTERIM_DIR / "panel.parquet"

METRICS_NOMS = {
    "sum_open_interest": "oi",                             # en crypto
    "sum_open_interest_value": "oi_usd",                   # en USDT
    "count_toptrader_long_short_ratio": "ls_top_comptes",  # gros traders, en nombre de comptes
    "sum_toptrader_long_short_ratio": "ls_top_positions",  # gros traders, pondéré par la taille des positions
    "count_long_short_ratio": "ls_global",                 # tous les comptes
    "sum_taker_long_short_vol_ratio": "taker_ratio",       # achats agressifs / ventes agressives
}


def grille_horaire() -> pd.DatetimeIndex:
    """Toutes les heures t, de DEBUT_DONNEES 01:00 (chauffe comprise) au lendemain de END 00:00."""
    debut = pd.Timestamp(DEBUT_DONNEES, tz="UTC") + pd.Timedelta("1h")
    fin = pd.Timestamp(END, tz="UTC") + pd.Timedelta("1D")
    return pd.date_range(debut, fin, freq="h", name="t", unit="ms")


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Relevés à 5 min -> une ligne par heure t (relevés de t-55 min à t)."""
    df = df.drop(columns="symbol").rename(columns=METRICS_NOMS)
    df["releve"] = 1

    # Un open interest nul est impossible : c'est une erreur d'enregistrement
    df["oi"] = df["oi"].mask(df["oi"] == 0)
    df["oi_usd"] = df["oi_usd"].mask(df["oi_usd"] == 0)

    heures = df.set_index("create_time").resample("1h", closed="right", label="right")
    horaire = heures.agg(
        oi=("oi", "last"),
        oi_usd=("oi_usd", "last"),
        ls_top_comptes=("ls_top_comptes", "mean"),
        ls_top_positions=("ls_top_positions", "mean"),
        ls_global=("ls_global", "mean"),
        taker_ratio=("taker_ratio", "mean"),
        n_releves=("releve", "count"),
    )
    horaire.index = horaire.index.as_unit("ms").rename("t")
    return horaire


def clean_klines(df: pd.DataFrame, prefixe: str = "") -> pd.DataFrame:
    """Bougies -> une ligne par heure t, t étant la fin de la bougie."""
    df = df.drop(columns=["close_time", "ignore"])
    df["t"] = (df["open_time"] + pd.Timedelta("1h")).dt.as_unit("ms")
    df = df.drop(columns="open_time").drop_duplicates("t").set_index("t")
    return df.add_prefix(prefixe)


def clean_basis(df: pd.DataFrame) -> pd.DataFrame:
    """premiumIndexKlines -> colonne basis (ses volumes sont toujours nuls)."""
    return clean_klines(df)[["close"]].rename(columns={"close": "basis"})


def clean_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Paiements de funding arrondis à l'heure, avec le taux ramené à une heure."""
    df = df.rename(columns={"calc_time": "t_funding", "last_funding_rate": "funding",
                            "funding_interval_hours": "funding_intervalle_h"})
    df["t_funding"] = df["t_funding"].dt.round("h").dt.as_unit("ms")
    df["funding_horaire"] = df["funding"] / df["funding_intervalle_h"]
    return df.drop_duplicates("t_funding").sort_values("t_funding")


def build_panel_crypto(symbol: str) -> pd.DataFrame:
    """Toutes les sources Binance d'une crypto, fusionnées sur la grille horaire."""
    panel = pd.DataFrame(index=grille_horaire())
    panel = panel.join(clean_klines(load_binance(symbol, "klines")))
    panel = panel.join(clean_klines(load_binance(symbol, "spot_klines"), "spot_"))
    panel = panel.join(clean_basis(load_binance(symbol, "premiumIndexKlines")))
    panel = panel.join(clean_metrics(load_binance(symbol, "metrics")))
    panel["n_releves"] = panel["n_releves"].fillna(0).astype("int64")

    # Chaque heure t reçoit le dernier paiement de funding effectué avant ou à t
    funding = clean_funding(load_binance(symbol, "fundingRate"))
    panel = pd.merge_asof(panel.reset_index(), funding, left_on="t", right_on="t_funding")
    panel = panel.set_index("t")

    panel = ajouter_fng(panel)

    panel.insert(0, "symbol", symbol)
    return panel


def build_panel(symbols: list[str]) -> pd.DataFrame:
    """Panel de toutes les cryptos, enregistré dans data/interim/panel.parquet."""
    panel = pd.concat([build_panel_crypto(s) for s in symbols])
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_FILE)
    return panel


def rapport_qualite(panel: pd.DataFrame) -> pd.DataFrame:
    """Part de valeurs manquantes (en %) par crypto et par colonne."""
    manquants = panel.drop(columns="symbol").isna().groupby(panel["symbol"]).mean()
    return (manquants * 100).round(2)


def controles_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Structure du panel, par crypto : heures présentes et attendues, doublons, heures sans bougie ou sans relevé."""
    resultats = {}
    for symbol, crypto in panel.groupby("symbol"):
        resultats[symbol] = {
            "Heures": len(crypto),
            "Heures attendues": len(grille_horaire()),
            "Heures en double": crypto.index.duplicated().sum(),
            "Heures sans bougie": crypto["close"].isna().sum(),
            "Heures sans relevé": (crypto["n_releves"] == 0).sum(),
            "Heures avec 1 à 11 relevés": crypto["n_releves"].between(1, 11).sum(),
        }
    return pd.DataFrame(resultats).T
