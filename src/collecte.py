"""Collecte des données brutes : archives Binance (data.binance.vision)."""

import io
import zipfile

import pandas as pd
import requests

from src.config import BINANCE_BASE_URL

# Description de chaque type d'archive Binance :
#   marche    : "futures/um" (perpétuels USDT) ou "spot"
#   frequence : "daily" (un fichier par jour) ou "monthly" (un fichier par mois)
#   dossier   : nom du dossier chez Binance
#   intervalle: True si le type a un sous-dossier d'intervalle (klines)
#   temps     : colonne qui contient l'horodatage
BINANCE_KINDS = {
    "metrics": {"marche": "futures/um", "frequence": "daily",
                "dossier": "metrics", "intervalle": False, "temps": "create_time"},
    "fundingRate": {"marche": "futures/um", "frequence": "monthly",
                    "dossier": "fundingRate", "intervalle": False, "temps": "calc_time"},
    "klines": {"marche": "futures/um", "frequence": "monthly",
               "dossier": "klines", "intervalle": True, "temps": "open_time"},
    "premiumIndexKlines": {"marche": "futures/um", "frequence": "monthly",
                           "dossier": "premiumIndexKlines", "intervalle": True, "temps": "open_time"},
    "spot_klines": {"marche": "spot", "frequence": "monthly",
                    "dossier": "klines", "intervalle": True, "temps": "open_time"},
}

# Colonnes des klines : les fichiers anciens n'ont pas de ligne d'en-tête
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def binance_url(symbol: str, kind: str, period: str, interval: str = "1h") -> str:
    """URL d'une archive Binance.

    period : "AAAA-MM-JJ" pour un type quotidien, "AAAA-MM" pour un type mensuel.
    Exemple : binance_url("BTCUSDT", "klines", "2023-01")
    -> .../futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip
    """
    if kind not in BINANCE_KINDS:
        raise ValueError(f"Type inconnu : {kind!r}. Types possibles : {list(BINANCE_KINDS)}")
    info = BINANCE_KINDS[kind]

    # La longueur de period doit correspondre à la fréquence du type
    longueur_attendue = 10 if info["frequence"] == "daily" else 7
    if len(period) != longueur_attendue:
        raise ValueError(f"{kind} attend une période de {longueur_attendue} caractères, reçu {period!r}")

    dossier = f"{BINANCE_BASE_URL}/{info['marche']}/{info['frequence']}/{info['dossier']}/{symbol}"
    if info["intervalle"]:
        # Pour les klines, l'intervalle remplace le type dans le nom du fichier
        return f"{dossier}/{interval}/{symbol}-{interval}-{period}.zip"
    return f"{dossier}/{symbol}-{kind}-{period}.zip"


def fetch_binance_file(symbol: str, kind: str, period: str, interval: str = "1h") -> bytes | None:
    """Télécharge une archive Binance et renvoie son contenu brut (le zip, en octets).

    Renvoie None si le fichier n'existe pas (crypto pas encore cotée, jour absent).
    Lève une exception pour toute autre erreur HTTP.
    """
    url = binance_url(symbol, kind, period, interval)
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()  # les 404 sont traités plus haut
    return response.content


def _to_utc(ts: pd.Series) -> pd.Series:
    """Convertit une colonne de temps Binance en dates UTC.

    Trois formats coexistent : texte ("2023-01-01 00:05:00"), millisecondes
    (13 chiffres) et microsecondes (16 chiffres, spot à partir de 2025).
    """
    if not pd.api.types.is_numeric_dtype(ts):
        return pd.to_datetime(ts, utc=True)
    unite = "us" if ts.max() > 1e14 else "ms"
    return pd.to_datetime(ts, unit=unite, utc=True)


def read_binance_zip(content: bytes, kind: str) -> pd.DataFrame:
    """Lit le CSV contenu dans une archive Binance et convertit le temps en UTC."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        texte = z.read(z.namelist()[0]).decode()

    # Fichier sans en-tête si la première ligne commence par un chiffre (vieux klines)
    if texte[0].isdigit():
        df = pd.read_csv(io.StringIO(texte), header=None, names=KLINE_COLS)
    else:
        df = pd.read_csv(io.StringIO(texte))

    colonne_temps = BINANCE_KINDS[kind]["temps"]
    df[colonne_temps] = _to_utc(df[colonne_temps])
    return df
