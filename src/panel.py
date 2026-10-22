"""Nettoyage des données brutes Binance et construction du panel horaire.

Convention (voir README, « Convention de temps ») : le panel a une ligne par crypto
et par heure t (UTC), et la ligne t ne contient que ce qui est connu à l'instant t.

Enchaînement : clean_metrics, clean_klines, clean_funding (une source chacune)
-> build_panel_crypto (fusion sur la grille horaire d'une crypto)
-> build_panel (toutes les cryptos, enregistré dans data/interim/panel.parquet).
"""

import pandas as pd

from src.collecte_binance import load_binance
from src.config import END, INTERIM_DIR, START

# Noms des colonnes metrics : Binance -> projet
METRICS_NOMS = {
    "sum_open_interest": "oi",                             # open interest, en crypto (BTC...)
    "sum_open_interest_value": "oi_usd",                   # open interest, en USDT
    "count_toptrader_long_short_ratio": "ls_top_comptes",  # gros traders : comptes long / comptes short
    "sum_toptrader_long_short_ratio": "ls_top_positions",  # gros traders : positions long / short (pondéré par la taille)
    "count_long_short_ratio": "ls_global",                 # tous les comptes : long / short
    "sum_taker_long_short_vol_ratio": "taker_ratio",       # volume acheteur agressif / vendeur agressif
}

PANEL_FILE = INTERIM_DIR / "panel.parquet"


def grille_horaire() -> pd.DatetimeIndex:
    """Toutes les heures t de la période d'étude.

    Première heure : START 01:00, fin de la première bougie (00:00-01:00).
    Dernière heure : le lendemain de END à 00:00, fin de la dernière bougie (23:00-24:00).
    """
    debut = pd.Timestamp(START, tz="UTC") + pd.Timedelta("1h")
    fin = pd.Timestamp(END, tz="UTC") + pd.Timedelta("1D")
    return pd.date_range(debut, fin, freq="h", name="t", unit="ms")


def clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Relevés metrics à 5 min -> une ligne par heure t.

    La ligne t regroupe les relevés de t-55 min à t inclus : open interest au dernier
    relevé, ratios en moyenne, et n_releves le nombre de relevés (12 si l'heure est complète).
    Les heures sans aucun relevé sont gardées, avec n_releves = 0 et des valeurs manquantes.
    """
    df = df.drop(columns="symbol").rename(columns=METRICS_NOMS)
    df["releve"] = 1  # compté avant le nettoyage : une ligne présente = un relevé reçu

    # Un open interest nul est impossible : défaut d'enregistrement chez Binance.
    # Les deux colonnes sont traitées séparément (leurs zéros ne tombent pas aux mêmes lignes).
    for col in ["oi", "oi_usd"]:
        df[col] = df[col].mask(df[col] == 0)
    # Les ratios extrêmes (taker_ratio jusqu'à 60) sont gardés : ils sont possibles sur
    # 5 min de faible volume, et c'est justement ce qu'on étudie. Pas de valeur comblée.

    horaire = df.set_index("create_time").resample("1h", closed="right", label="right").agg(
        oi=("oi", "last"),          # stock : son niveau à la fin de l'heure
        oi_usd=("oi_usd", "last"),
        ls_top_comptes=("ls_top_comptes", "mean"),      # mesures sur 5 min : moyenne de l'heure
        ls_top_positions=("ls_top_positions", "mean"),
        ls_global=("ls_global", "mean"),
        taker_ratio=("taker_ratio", "mean"),
        n_releves=("releve", "count"),
    )
    horaire.index = horaire.index.as_unit("ms").rename("t")
    return horaire


def clean_klines(df: pd.DataFrame, prefixe: str = "") -> pd.DataFrame:
    """Bougies horaires -> une ligne par heure t = fin de la bougie (open_time + 1 h).

    prefixe distingue les marchés dans le panel : "" pour le perpétuel, "spot_" pour le spot.
    """
    df = df.drop(columns=["close_time", "ignore"])
    df["t"] = (df["open_time"] + pd.Timedelta("1h")).dt.as_unit("ms")
    df = df.drop(columns="open_time").drop_duplicates("t").set_index("t").sort_index()
    return df.add_prefix(prefixe)


def clean_basis(df: pd.DataFrame) -> pd.DataFrame:
    """premiumIndexKlines -> basis (écart relatif perpétuel / indice) en fin d'heure t.

    Seul le close est gardé : les colonnes de volume sont toujours nulles pour cet indice.
    """
    return clean_klines(df)[["close"]].rename(columns={"close": "basis"})


def clean_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Paiements de funding, datés à l'heure pile, avec le taux ramené à une heure.

    Binance horodate le calcul avec quelques millisecondes de retard (16:00:00.007) :
    on arrondit à l'heure pour pouvoir rattacher chaque paiement à la grille.
    """
    df = df.rename(columns={"calc_time": "t_funding", "last_funding_rate": "funding",
                            "funding_interval_hours": "funding_intervalle_h"})
    df["t_funding"] = df["t_funding"].dt.round("h").dt.as_unit("ms")
    df["funding_horaire"] = df["funding"] / df["funding_intervalle_h"]
    return df.drop_duplicates("t_funding").sort_values("t_funding").reset_index(drop=True)


def build_panel_crypto(symbol: str) -> pd.DataFrame:
    """Panel horaire d'une crypto : toutes les sources Binance fusionnées sur la grille."""
    panel = pd.DataFrame(index=grille_horaire())

    # Jointures sur l'index t : la grille impose les lignes, une heure absente d'une
    # source devient une ligne de valeurs manquantes (left join, jamais de ligne en plus).
    panel = panel.join(clean_klines(load_binance(symbol, "klines")), how="left")
    panel = panel.join(clean_klines(load_binance(symbol, "spot_klines"), "spot_"), how="left")
    panel = panel.join(clean_basis(load_binance(symbol, "premiumIndexKlines")), how="left")
    panel = panel.join(clean_metrics(load_binance(symbol, "metrics")), how="left")
    panel["n_releves"] = panel["n_releves"].fillna(0).astype("int64")

    # Funding : chaque heure t reçoit le dernier paiement dont l'heure est <= t.
    panel = pd.merge_asof(panel, clean_funding(load_binance(symbol, "fundingRate")),
                          left_index=True, right_on="t_funding", direction="backward")
    panel.index = grille_horaire()  # merge_asof remplace l'index : on remet t

    if len(panel) != len(grille_horaire()):
        raise ValueError(f"{symbol} : la fusion a ajouté des lignes (doublons dans une source)")
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
    return (panel.drop(columns="symbol").isna().groupby(panel["symbol"]).mean() * 100).round(2)
