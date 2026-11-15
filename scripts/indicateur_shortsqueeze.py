import numpy as np
import pandas as pd
from src.panel_binance import PANEL_FILE
from src.config import PROCESSED_DIR

PANEL_CIBLE_FILE = PROCESSED_DIR / "panel_cible.parquet"

panel = pd.read_parquet(PANEL_FILE)

def is_shortsquizz(t: pd.Timestamp, i: str):
    
    paneli = panel.loc[panel["symbol"] == i]
    
    if t + pd.Timedelta(hours=24) > paneli.index.max():
        return {"shortsqueeze": pd.NA, "date_pic": pd.NaT, "hausse_pic": np.nan,
                "variation_oi": np.nan, "ratio_achats_ventes": np.nan, "sigma_30j": np.nan}

    close = paneli["close"]
    high = paneli["high"]
    tau = high[t + pd.Timedelta(hours=1): t + pd.Timedelta(days=1)].idxmax() #heure où le maximum est atteint dans les 24h après t

    rendements = np.log(close / close.shift(1))
    sigma = rendements.loc[t + pd.Timedelta(hours=1) - pd.Timedelta(days=30) : t].std()
    
    r1 = np.log(high[tau] / close[t])
    c1 = r1 > np.log(1.05) and r1 > 3*sigma*np.sqrt(24)

    r2 = paneli.loc[tau, "oi"]/paneli.loc[t, "oi"] - 1
    c2 = r2 < -0.10

    v_achats = np.sum(paneli.loc[t + pd.Timedelta(hours=1):tau, "taker_buy_volume"])
    v_ventes = np.sum(paneli.loc[t + pd.Timedelta(hours=1):tau, "volume"] - paneli.loc[t + pd.Timedelta(hours=1):tau, "taker_buy_volume"])

    r3 = v_achats / v_ventes 
    c3 = r3 > 1

    res = {"shortsqueeze" : bool(c1 and c2 and c3), 
           "date_pic" : tau, "hausse_pic" : high.loc[tau]/close.loc[t] - 1, 
           "variation_oi" : r2, "ratio_achats_ventes" : r3, 
           "sigma_30j" : sigma }
    return res

def shortsquizzs(t1 : pd.Timestamp, t2 : pd.Timestamp, i):
    res = []
    for t in pd.date_range(t1, t2, freq="1h"):
        if is_shortsquizz(t, i)["shortsqueeze"]:
            res.append(t)
    return res

if __name__ == "__main__":

    panel_cibles = panel.copy()

    diagnostics = pd.DataFrame([is_shortsquizz(t, symbole)
                                for t, symbole in zip(panel.index, panel["symbol"])],
                                index=panel.index)

    panel_cibles = pd.concat([panel, diagnostics], axis=1)

    PANEL_CIBLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel_cibles.to_parquet(PANEL_CIBLE_FILE)

def episodes(l : list):
    """ 
    Identifie les épisodes de shortsquizzs consécutifs dans une liste de timestamps fournie par shortsquizzs.
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

"""  
dates_shortsquizz_BTC = shortsquizzs(pd.Timestamp("2023-01-01", tz = "UTC"), pd.Timestamp("2026-09-30", tz = "UTC"), "BTCUSDT")
episodes_BTC = episodes(dates_shortsquizz_BTC)

print(episodes_BTC)

"""






