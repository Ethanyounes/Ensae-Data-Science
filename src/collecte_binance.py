"""Téléchargement des archives Binance (data.binance.vision) en fichiers parquet."""

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import BINANCE_BASE_URL, raw_path

# Types d'archives téléchargés, avec leur colonne de temps
BINANCE_KINDS = {
    "metrics": "create_time",
    "fundingRate": "calc_time",
    "klines": "open_time",
    "premiumIndexKlines": "open_time",
    "spot_klines": "open_time",
}
KLINE_KINDS = ["klines", "premiumIndexKlines", "spot_klines"]

# Les vieux fichiers de klines n'ont pas de ligne d'en-tête
BINANCE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

session = requests.Session()  # une seule connexion pour des milliers de fichiers


def binance_url(symbol: str, kind: str, period: str, interval: str = "1h") -> str:
    """URL d'une archive. period vaut "AAAA-MM-JJ" pour metrics, "AAAA-MM" sinon."""
    marche = "spot" if kind == "spot_klines" else "futures/um"
    frequence = "daily" if kind == "metrics" else "monthly"
    dossier = "klines" if kind == "spot_klines" else kind

    debut = f"{BINANCE_BASE_URL}/{marche}/{frequence}/{dossier}/{symbol}"
    if kind in KLINE_KINDS:
        return f"{debut}/{interval}/{symbol}-{interval}-{period}.zip"
    return f"{debut}/{symbol}-{kind}-{period}.zip"


def fetch_binance_file(symbol: str, kind: str, period: str, interval: str = "1h") -> bytes | None:
    """Télécharge une archive et vérifie son checksum. Renvoie None si elle n'existe pas."""
    url = binance_url(symbol, kind, period, interval)
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    checksum_attendu = session.get(url + ".CHECKSUM", timeout=60).text.split()[0]
    if hashlib.sha256(response.content).hexdigest() != checksum_attendu:
        raise ValueError(f"Fichier corrompu : {url}")
    return response.content


def to_utc(dates: pd.Series) -> pd.Series:
    """Dates Binance (texte, millisecondes ou microsecondes) -> dates UTC."""
    if not pd.api.types.is_numeric_dtype(dates):
        return pd.to_datetime(dates, utc=True)
    unite = "us" if dates.max() > 1e14 else "ms"  # le spot passe en microsecondes en 2025
    return pd.to_datetime(dates, unit=unite, utc=True)


def read_binance_zip(content: bytes, kind: str) -> pd.DataFrame:
    """Lit le CSV d'une archive et convertit sa colonne de temps en UTC."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        texte = z.read(z.namelist()[0]).decode()

    sans_entete = texte[0].isdigit()
    if sans_entete:
        df = pd.read_csv(io.StringIO(texte), header=None, names=BINANCE_KLINE_COLS)
    else:
        df = pd.read_csv(io.StringIO(texte))

    colonne_temps = BINANCE_KINDS[kind]
    df[colonne_temps] = to_utc(df[colonne_temps])
    return df


def binance_periods(kind: str, start: str, end: str) -> list[str]:
    """Jours (metrics) ou mois (autres types) entre start et end inclus."""
    if kind == "metrics":
        return list(pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"))
    premier_du_mois = pd.Timestamp(start).replace(day=1)
    return list(pd.date_range(premier_du_mois, end, freq="MS").strftime("%Y-%m"))


def binance_file(symbol: str, kind: str, period: str) -> Path:
    """Chemin du parquet d'une archive."""
    return raw_path("binance", kind, symbol) / f"{symbol}-{kind}-{period}.parquet"


def download_binance(symbol: str, kind: str, start: str, end: str, interval: str = "1h") -> dict:
    """Télécharge les archives manquantes entre start et end et renvoie un bilan."""
    bilan = {"telecharges": 0, "deja_presents": 0, "absents": 0}

    for period in tqdm(binance_periods(kind, start, end), desc=f"{symbol} {kind}"):
        fichier = binance_file(symbol, kind, period)
        if fichier.exists():
            bilan["deja_presents"] += 1
            continue

        content = fetch_binance_file(symbol, kind, period, interval)
        if content is None:
            bilan["absents"] += 1
            continue

        fichier.parent.mkdir(parents=True, exist_ok=True)
        read_binance_zip(content, kind).to_parquet(fichier, index=False)
        bilan["telecharges"] += 1

    return bilan


def load_binance(symbol: str, kind: str) -> pd.DataFrame:
    """Relit tous les parquets d'une crypto et d'un type, triés par date."""
    fichiers = sorted(raw_path("binance", kind, symbol).glob("*.parquet"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier {kind} pour {symbol} : lancer download_binance")

    df = pd.concat([pd.read_parquet(f) for f in fichiers], ignore_index=True)
    colonne_temps = BINANCE_KINDS[kind]
    return df.drop_duplicates(colonne_temps).sort_values(colonne_temps).reset_index(drop=True)
