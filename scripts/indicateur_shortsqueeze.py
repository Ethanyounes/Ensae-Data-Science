import numpy as np
import pandas as pd
from src.panel_binance import PANEL_FILE

panel = pd.read_parquet(PANEL_FILE)

def is_shortsquizz(t: pd.Timestamp, i: str):
    paneli = panel.loc[panel["symbol"] == i]
    close = paneli["close"]
    high = paneli["high"]
    tau = high[t + pd.Timedelta(hours=1): t + pd.Timedelta(days=1)].idxmax()

    rendements = np.log(close / close.shift(1))
    sigma = rendements.loc[t + pd.Timedelta(hours=1) - pd.Timedelta(days=30) : t].std()
    
    r = np.log(high[tau] / close[t])

    c1 = r > np.log(1.05) and r > 3*sigma*np.sqrt(24)

    if not c1: return False

    c2 = paneli.loc[tau, "oi"]/paneli.loc[t, "oi"] - 1 <= - 0.10

    if not c2: return False

    v_achats = np.sum(paneli.loc[t + pd.Timedelta(hours=1):tau, "taker_buy_volume"])
    v_ventes = np.sum(paneli.loc[t + pd.Timedelta(hours=1):tau, "volume"] - paneli.loc[t + pd.Timedelta(hours=1):tau, "taker_buy_volume"])
    c3 = v_achats / v_ventes > 1

    if c3: return True
    else: return False

def shortsquizzs(t1 : pd.Timestamp, t2 : pd.Timestamp, i):
    res = []
    for t in pd.date_range(t1, t2, freq="1h"):
        if is_shortsquizz(t, i):
            res.append(t)
    return res

panel_cible = panel.copy()

panel_cible["is_shortsquizz"] = [
    bool(is_shortsquizz(t, symbole))
    for t, symbole in zip(panel_cible.index, panel_cible["symbol"])
]
    
dates_shortsquizz_BTC = shortsquizzs(pd.Timestamp("2023-01-01", tz = "UTC"), pd.Timestamp("2026-09-30", tz = "UTC"), "BTCUSDT")
dates_compactes_BTC = pd.DatetimeIndex(dates_shortsquizz_BTC).normalize().unique()   

print(dates_compactes_BTC)

"""
#DatetimeIndex(['2023-01-13 00:00:00+00:00', '2023-01-14 00:00:00+00:00',
               '2023-01-20 00:00:00+00:00', '2023-10-15 00:00:00+00:00',
               '2023-10-16 00:00:00+00:00', '2024-02-27 00:00:00+00:00',
               '2024-02-28 00:00:00+00:00'],
              dtype='datetime64[us, UTC]', freq=None)

"""






