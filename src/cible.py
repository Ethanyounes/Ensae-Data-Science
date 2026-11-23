"""Cible : y a-t-il un short squeeze dans les heures qui suivent t ? (conditions C1, C2, C3 du README)

Le calcul se fait d'un coup sur toutes les heures d'une crypto, sans boucle, et les seuils
sont des paramètres pour pouvoir tester les variantes.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.config import PROCESSED_DIR

PANEL_CIBLE_FILE = PROCESSED_DIR / "panel_cible.parquet"

# Valeurs de base. Les variantes du README changent un seul de ces paramètres.
K_SIGMA = 3        # C1 : hausse d'au moins k écarts-types...
PLANCHER = 0.05    # ... et d'au moins 5 %
DELTA_OI = 0.10    # C2 : baisse de l'open interest d'au moins 10 %
HORIZON = 24       # en heures


def delai_du_pic(high: pd.Series, horizon: int) -> np.ndarray:
    """Nombre d'heures entre t et le plus haut des `horizon` heures suivantes (NaN en fin de série)."""
    n = len(high)
    delai = np.full(n, np.nan)
    # Fenêtre i : lignes i à i+horizon-1. La fenêtre i+1 couvre donc les heures t+1h à t+horizon de la ligne i.
    fenetres = sliding_window_view(high.fillna(-np.inf).to_numpy(), horizon)
    delai[: n - horizon] = fenetres[1:].argmax(axis=1) + 1
    return delai


def cible_crypto(crypto: pd.DataFrame, k: float = K_SIGMA, plancher: float = PLANCHER,
                 delta: float = DELTA_OI, horizon: int = HORIZON) -> pd.DataFrame:
    """Conditions C1, C2, C3 et grandeurs qui servent à les tester, pour chaque heure t d'une crypto."""
    close = crypto["close"]
    rendements = np.log(close / close.shift(1))
    sigma_30j = rendements.rolling("30D").std()  # rendements horaires de t-30j (exclu) à t

    delai = delai_du_pic(crypto["high"], horizon)
    connue = ~np.isnan(delai)  # les dernières heures de la série n'ont pas assez de futur
    # Numéro de ligne du pic (la ligne t elle-même quand le pic est inconnu, masquée plus bas)
    ligne_pic = np.arange(len(crypto)) + np.where(connue, delai, 0).astype(int)

    hausse_pic = crypto["high"].to_numpy()[ligne_pic] / close - 1
    variation_oi = crypto["oi"].to_numpy()[ligne_pic] / crypto["oi"] - 1

    # Somme des volumes de t+1h au pic = cumul au pic - cumul à t
    achats = crypto["taker_buy_volume"].fillna(0).cumsum()
    ventes = (crypto["volume"] - crypto["taker_buy_volume"]).fillna(0).cumsum()
    ratio_achats_ventes = (achats.to_numpy()[ligne_pic] - achats) / (ventes.to_numpy()[ligne_pic] - ventes)

    cible = pd.DataFrame({
        "sigma_30j": sigma_30j,
        "date_pic": crypto.index + pd.to_timedelta(delai, unit="h"),
        "delai_pic_h": delai,
        "hausse_pic": hausse_pic.where(connue),
        "variation_oi": variation_oi.where(connue),
        "ratio_achats_ventes": ratio_achats_ventes.where(connue),
    })
    # La volatilité est ramenée à l'horizon : σ horaire × √horizon
    cible["c1"] = cible["hausse_pic"] >= np.maximum(k * sigma_30j * np.sqrt(horizon), plancher)
    cible["c2"] = cible["variation_oi"] <= -delta
    cible["c3"] = cible["ratio_achats_ventes"] > 1
    cible["shortsqueeze"] = cible["c1"] & cible["c2"] & cible["c3"]
    # Sans futur, la cible est inconnue (NA) et non fausse
    for colonne in ["c1", "c2", "c3", "shortsqueeze"]:
        cible[colonne] = cible[colonne].astype("boolean").mask(~connue)
    return cible


def ajouter_cible(panel: pd.DataFrame, k: float = K_SIGMA, plancher: float = PLANCHER,
                  delta: float = DELTA_OI, horizon: int = HORIZON) -> pd.DataFrame:
    """Panel complété des colonnes de la cible, calculées crypto par crypto."""
    return pd.concat([crypto.join(cible_crypto(crypto, k, plancher, delta, horizon))
                      for _, crypto in panel.groupby("symbol")])


def episodes(l: list):
    """
    Identifie les épisodes de shortsquizzs consécutifs dans une liste de timestamps.
    Un épisode est une suite d'heures positives consécutives.
    """
    if len(l) == 0:
        return []
    longueur = len(l)
    j = 1
    res = []
    for i in range(1, longueur):
        if l[i] == l[i-1] + pd.Timedelta(hours=1):
            j += 1
        else:
            res.append((l[i-1] - pd.Timedelta(hours=j-1), j))
            j = 1
    res.append((l[-1] - pd.Timedelta(hours=j-1), j))
    return res
