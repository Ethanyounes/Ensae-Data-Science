"""Collecte des données brutes : archives Binance (data.binance.vision).

Enchaînement : binance_url -> fetch_binance_file (téléchargement + checksum)
-> read_binance_zip (zip -> DataFrame) -> download_binance (boucle + parquet).
load_binance relit ensuite tous les parquets d'une crypto.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import BINANCE_BASE_URL, raw_path

# Une seule connexion réutilisée pour toutes les requêtes : bien plus rapide
# que d'en ouvrir une nouvelle pour chacun des milliers de fichiers.
_SESSION = requests.Session()

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
BINANCE_KLINE_COLS = [
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
    Lève une exception pour toute autre erreur HTTP, ou si le checksum ne correspond pas.
    """
    url = binance_url(symbol, kind, period, interval)
    response = _SESSION.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()  # les 404 sont traités plus haut

    # Binance publie à côté de chaque zip son empreinte SHA-256 ("<empreinte>  <nom>").
    # On la compare à celle du fichier reçu pour garantir qu'il est intact.
    checksum = _SESSION.get(url + ".CHECKSUM", timeout=60)
    checksum.raise_for_status()
    attendu = checksum.text.split()[0]
    recu = hashlib.sha256(response.content).hexdigest()
    if recu != attendu:
        raise ValueError(f"Checksum incorrect pour {url} : fichier corrompu")
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
        df = pd.read_csv(io.StringIO(texte), header=None, names=BINANCE_KLINE_COLS)
    else:
        df = pd.read_csv(io.StringIO(texte))

    colonne_temps = BINANCE_KINDS[kind]["temps"]
    df[colonne_temps] = _to_utc(df[colonne_temps])
    return df


def binance_periods(kind: str, start: str, end: str) -> list[str]:
    """Liste des périodes à télécharger entre start et end (inclus), au format de Binance.

    Exemples : binance_periods("metrics", "2023-01-30", "2023-02-01")
    -> ["2023-01-30", "2023-01-31", "2023-02-01"]
    binance_periods("klines", "2023-01-15", "2023-03-10") -> ["2023-01", "2023-02", "2023-03"]
    """
    if BINANCE_KINDS[kind]["frequence"] == "daily":
        jours = pd.date_range(start, end, freq="D")
        return list(jours.strftime("%Y-%m-%d"))

    # "MS" = premier jour de chaque mois. On ramène start au 1er du mois,
    # sinon le mois de départ serait oublié dès que start n'est pas un 1er.
    debut_du_mois = pd.Timestamp(start).replace(day=1)
    mois = pd.date_range(debut_du_mois, end, freq="MS")
    return list(mois.strftime("%Y-%m"))


def binance_file(symbol: str, kind: str, period: str) -> Path:
    """Chemin du parquet d'une archive : data/raw/binance/<kind>/<SYMBOLE>/<SYMBOLE>-<kind>-<période>.parquet."""
    return raw_path("binance", kind, symbol) / f"{symbol}-{kind}-{period}.parquet"


def download_binance(symbol: str, kind: str, start: str, end: str, interval: str = "1h") -> dict:
    """Télécharge toutes les archives d'une crypto et d'un type entre start et end.

    Chaque archive devient un parquet. Un parquet déjà présent n'est pas retéléchargé :
    on peut donc relancer la fonction après une coupure, elle reprend où elle s'était arrêtée.
    Renvoie le bilan : nombre de fichiers téléchargés, déjà présents et absents chez Binance.
    """
    bilan = {"telecharges": 0, "deja_presents": 0, "absents": 0}

    for period in tqdm(binance_periods(kind, start, end), desc=f"{symbol} {kind}"):
        fichier = binance_file(symbol, kind, period)
        if fichier.exists():
            bilan["deja_presents"] += 1
            continue

        content = fetch_binance_file(symbol, kind, period, interval)
        if content is None:  # crypto pas encore cotée, ou mois en cours pas encore publié
            bilan["absents"] += 1
            continue

        df = read_binance_zip(content, kind)
        fichier.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(fichier, index=False)
        bilan["telecharges"] += 1

    return bilan


def load_binance(symbol: str, kind: str) -> pd.DataFrame:
    """Relit tous les parquets d'une crypto et d'un type en un seul DataFrame trié par date."""
    fichiers = sorted(raw_path("binance", kind, symbol).glob("*.parquet"))
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier pour {symbol} {kind} : lancez d'abord download_binance.")

    df = pd.concat([pd.read_parquet(f) for f in fichiers], ignore_index=True)

    # Deux archives voisines peuvent partager une ligne à leur frontière
    colonne_temps = BINANCE_KINDS[kind]["temps"]
    df = df.drop_duplicates(colonne_temps).sort_values(colonne_temps)
    return df.reset_index(drop=True)
