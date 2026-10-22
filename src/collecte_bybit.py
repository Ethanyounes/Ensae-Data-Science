"""Collecte Bybit (API publique v5) : funding, open interest et bougies 1 h.

Bybit sert de source de robustesse : vérifier que les résultats ne sont pas
propres aux clients de Binance. Pas de clé d'API nécessaire.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_bybit
"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import BYBIT_BASE_URL, END, START, raw_path

# Description de chaque série Bybit :
#   chemin : point d'accès de l'API
#   limite : nombre maximal de points par requête (fixé par Bybit)
#   debut, fin : noms des paramètres de dates (ils diffèrent selon le point d'accès)
#   extra  : paramètres propres à la série
BYBIT_KINDS = {
    "funding": {"chemin": "/v5/market/funding/history", "limite": 200,
                "debut": "startTime", "fin": "endTime", "extra": {}},
    # Pas 1 h : moins de 200 requêtes par crypto sur quatre ans (contre ~2 000 en 5 min)
    "open_interest": {"chemin": "/v5/market/open-interest", "limite": 200,
                      "debut": "startTime", "fin": "endTime", "extra": {"intervalTime": "1h"}},
    "klines": {"chemin": "/v5/market/kline", "limite": 1000,
               "debut": "start", "fin": "end", "extra": {"interval": "60"}},
}

# Les bougies arrivent sous forme de listes sans noms : ordre donné par la doc Bybit.
# Noms alignés sur ceux de Binance pour réutiliser le même code de panel.
BYBIT_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]

UNE_HEURE_MS = 3600 * 1000


def _to_ms(date: str) -> int:
    """Date "AAAA-MM-JJ" (UTC) -> millisecondes depuis 1970, le format de Bybit.

    L'API Bybit attend un nombre de millisecondes depuis le 1er janvier 1970, et non une date.
    """
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def _timestamp(ligne, kind: str) -> int:
    """Horodatage (ms) d'une ligne brute renvoyée par Bybit.

    Bybit ne place pas la date au même endroit selon la série : cette fonction la
    récupère sous forme d'entier quel que soit le format.
    """
    if kind == "klines":
        return int(ligne[0])  # première valeur de la liste
    if kind == "funding":
        return int(ligne["fundingRateTimestamp"])
    return int(ligne["timestamp"])


def fetch_bybit_page(kind: str, symbol: str, debut_ms: int, fin_ms: int,
                     essais: int = 5) -> list:
    """Une requête Bybit : au plus `limite` points entre debut_ms et fin_ms (inclus).

    Bybit renvoie les points du plus récent au plus ancien. Les erreurs passagères
    du serveur (« svc error », coupure réseau) sont retentées avec une attente croissante.
    """
    for essai in range(1, essais + 1):
        try:
            return _requete_bybit(kind, symbol, debut_ms, fin_ms)
        except (requests.RequestException, RuntimeError):
            if essai == essais:
                raise
            time.sleep(2 ** essai)  # 2, 4, 8, 16 s


def _requete_bybit(kind: str, symbol: str, debut_ms: int, fin_ms: int) -> list:
    """Envoie une seule requête à Bybit et renvoie la liste de points."""
    info = BYBIT_KINDS[kind]
    params = {
        "category": "linear",  # perpétuels USDT
        "symbol": symbol,
        "limit": info["limite"],
        info["debut"]: debut_ms,  # le funding exige toujours le début ET la fin
        info["fin"]: fin_ms,
        **info["extra"],
    }
    response = requests.get(BYBIT_BASE_URL + info["chemin"], params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Bybit signale ses erreurs (symbole inconnu, paramètre invalide...) avec un code HTTP 200
    if data["retCode"] != 0:
        raise RuntimeError(f"Bybit {kind} {symbol} : {data['retMsg']}")
    return data["result"]["list"]


def download_bybit(kind: str, symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
    """Télécharge une série Bybit sur [start, end] et renvoie un DataFrame trié en UTC.

    Comme les réponses vont du plus récent au plus ancien, la boucle remonte le
    temps : après chaque page, la nouvelle fin est juste avant le point le plus ancien reçu.
    """
    if kind not in BYBIT_KINDS:
        raise ValueError(f"Type inconnu : {kind!r}. Types possibles : {list(BYBIT_KINDS)}")

    debut_ms = _to_ms(start)
    fin_ms = _to_ms(end) + 24 * UNE_HEURE_MS - 1  # END est inclus : jusqu'à 23:59:59.999
    lignes = []
    while fin_ms >= debut_ms:
        page = fetch_bybit_page(kind, symbol, debut_ms, fin_ms)
        if not page:  # plus rien avant : contrat pas encore coté
            break
        lignes.extend(page)
        fin_ms = _timestamp(page[-1], kind) - 1  # page[-1] est le point le plus ancien
        time.sleep(0.1)  # rester loin de la limite de requêtes de Bybit

    return clean_bybit(lignes, kind)


def clean_bybit(lignes: list, kind: str) -> pd.DataFrame:
    """Met les lignes brutes au format du projet : noms, types numériques, dates UTC.

    Ce qui arrive de Bybit est une liste de textes : on la convertit en tableau pandas.
    """
    if kind == "klines":
        # Toutes les colonnes sont gardées : data/raw/ reste fidèle à la source,
        # le tri se fera dans le panel
        df = pd.DataFrame(lignes, columns=BYBIT_KLINE_COLS)
        colonne_temps = "open_time"
    elif kind == "funding":
        # La colonne symbol est retirée : la crypto est déjà dans le chemin du fichier
        df = pd.DataFrame(lignes, columns=["fundingRateTimestamp", "fundingRate"]).rename(
            columns={"fundingRateTimestamp": "calc_time", "fundingRate": "last_funding_rate"})
        colonne_temps = "calc_time"
    else:
        # openInterest compte les deux côtés, singleOpenInterest un seul (comme Binance)
        df = pd.DataFrame(lignes, columns=["timestamp", "openInterest", "singleOpenInterest"])
        df = df.rename(columns={"timestamp": "create_time", "openInterest": "open_interest",
                                "singleOpenInterest": "single_open_interest"})
        colonne_temps = "create_time"

    if df.empty:
        return df

    # Bybit renvoie tout en texte : conversion en nombres, et des ms en dates UTC
    for col in df.columns.drop(colonne_temps):
        df[col] = pd.to_numeric(df[col])
    df[colonne_temps] = pd.to_datetime(df[colonne_temps].astype("int64"), unit="ms", utc=True)

    # Deux pages peuvent partager un point à leur frontière
    df = df.drop_duplicates(colonne_temps).sort_values(colonne_temps).reset_index(drop=True)

    if kind == "funding":
        # Intervalle entre deux paiements (8, 4 ou 1 h), pour ramener le taux en horaire.
        # Le premier paiement n'a pas de précédent : on lui donne l'intervalle suivant.
        ecart = df["calc_time"].diff().dt.total_seconds() / 3600
        df["funding_interval_hours"] = ecart.round().bfill().astype("int64")
    return df


def bybit_file(kind: str, symbol: str) -> Path:
    """Chemin du parquet d'une série : data/raw/bybit/<kind>/<SYMBOLE>/<SYMBOLE>-<kind>.parquet."""
    return raw_path("bybit", kind, symbol) / f"{symbol}-{kind}.parquet"


def save_bybit(df: pd.DataFrame, kind: str, symbol: str) -> None:
    """Écrit la série en parquet (le dossier est créé si besoin).

    Le parquet est plus léger qu'un CSV et conserve les types d'une lecture à l'autre.
    """
    fichier = bybit_file(kind, symbol)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fichier, index=False)
