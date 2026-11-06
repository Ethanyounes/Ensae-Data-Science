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
    bougies = bougies.drop(columns="open_time").drop_duplicates("t").set_index("t")
    # Un contrat Bybit peut contenir facteur fois plus de crypto : on ramène à l'unité Binance
    bougies[["open", "high", "low", "close"]] /= facteur
    bougies["volume"] *= facteur

    # Relevé instantané, connu à son heure (vérifié en le comparant à l'open interest Binance).
    # single_open_interest compte un seul côté du marché, comme l'open interest de Binance.
    oi = pd.read_parquet(bybit_file("open_interest", symbol))
    oi["t"] = oi["create_time"].dt.as_unit("ms")
    oi = oi.drop_duplicates("t").set_index("t")[["single_open_interest"]]
    oi = oi.rename(columns={"single_open_interest": "oi"})
    oi["oi"] *= facteur
    # Un open interest nul ou négatif est impossible : erreur d'enregistrement (même règle que Binance)
    oi["oi"] = oi["oi"].mask(oi["oi"] <= 0)

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


def controles_bybit(panel: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'heures qui échouent à chaque contrôle de cohérence, par crypto.

    Rien n'est corrigé ici : la fonction signale. 0 partout signifie des données saines.
    """
    resultats = {}
    for symbol, p in panel.groupby("symbol"):
        # Bougie cohérente : l'ouverture et la clôture sont entre le plus bas et le plus haut
        hors_bornes = (p[["open", "close"]].max(axis=1) > p["high"]) | (
            p[["open", "close"]].min(axis=1) < p["low"])
        # Open interest identique trois heures de suite : série probablement non mise à jour
        inchange = p["oi"].diff().eq(0)
        fige = inchange & inchange.shift(1, fill_value=False)

        resultats[symbol] = {
            "heures": len(p),
            "sans_bougie": p["close"].isna().sum(),
            "sans_oi": p["oi"].isna().sum(),
            "sans_funding": p["funding"].isna().sum(),
            "prix_hors_bornes": hors_bornes.sum(),
            "prix_non_positif": (p["low"] <= 0).sum(),
            "volume_nul": (p["volume"] == 0).sum(),
            "oi_fige_3h": fige.sum(),
        }
    return pd.DataFrame(resultats).T.astype("int64")


def extremes_bybit(panel: pd.DataFrame) -> pd.DataFrame:
    """Plus fortes variations horaires et funding extrêmes par crypto, avec leur date.

    Ces valeurs sont gardées dans le panel : ce sont de vrais événements de marché,
    et un squeeze est justement un événement extrême. On les signale seulement.
    """
    resultats = {}
    for symbol, p in panel.groupby("symbol"):
        variation_oi = p["oi"].pct_change()
        rendement = p["close"].pct_change()
        resultats[symbol] = {
            "baisse_oi_max_pct": round(variation_oi.min() * 100, 1),
            "date_baisse_oi": variation_oi.idxmin(),
            "variation_prix_max_pct": round(rendement.abs().max() * 100, 1),
            "date_variation_prix": rendement.abs().idxmax(),
            "funding_horaire_min": p["funding_horaire"].min(),
            "funding_horaire_max": p["funding_horaire"].max(),
            # Part des heures où le dernier funding connu est négatif (les shorts paient les longs)
            "heures_funding_negatif_pct": round((p["funding"] < 0).mean() * 100, 1),
        }
    return pd.DataFrame(resultats).T


def fusionner(panel: pd.DataFrame, panel_bybit: pd.DataFrame) -> pd.DataFrame:
    """Panel Binance complété des colonnes Bybit (suffixe _bybit), enregistré dans data/interim/."""
    bybit = panel_bybit.drop(columns="symbol").add_suffix("_bybit")
    bybit["symbol"] = panel_bybit["symbol"]

    fusion = panel.reset_index().merge(bybit.reset_index(), on=["t", "symbol"], how="left")
    fusion = fusion.set_index("t")
    fusion.to_parquet(PANEL_FUSION_FILE)
    return fusion
