"""Collecte Bybit  : funding, open interest et bougies 1 h.

Bybit sert de source de robustesse : vérifier que les résultats ne sont pas
propres aux clients de Binance.

Collecte de l'indice Fear & Greed (alternative.me), quotidien : tester si les
squeezes se concentrent dans les phases de peur.

"""

import time
from pathlib import Path

import pandas as pd
import requests

from src.config import BYBIT_BASE_URL, END, FNG_URL, RAW_DIR, START, SYMBOLS, raw_path

# On crée un dictionnaire qui décrit les 3 séries à télécharger 
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
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]

UNE_HEURE_MS = 3600 * 1000


#ByBit qui est une API attend un nombre de millisecondes depuis le 1er janvier 1970 et non une date 
def _to_ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)

#Bybit ne donne pas la date au même endroit selon la série donc cette fonction permet 
#de récupérer la date sous form d'entier quel que soit le format
def _timestamp(ligne, kind: str) -> int:
    if kind == "klines":
        return int(ligne[0])  # première valeur de la liste
    if kind == "funding":
        return int(ligne["fundingRateTimestamp"])
    return int(ligne["timestamp"])

#Une requête Bybit : au plus `limite` points entre debut_ms et fin_ms (inclus).
#Bybit renvoie les points du plus récent au plus ancien. Les erreurs passagères
#du serveur comme « svc error », coupure réseau sont retentées et le temps d'attente augmente.
def fetch_bybit_page(kind: str, symbol: str, debut_ms: int, fin_ms: int,
                     essais: int = 5) -> list:
    
    for essai in range(1, essais + 1):
        try:
            return _requete_bybit(kind, symbol, debut_ms, fin_ms)
        except (requests.RequestException, RuntimeError):
            if essai == essais:
                raise
            time.sleep(2 ** essai)  # 2, 4, 8, 16 s

#Pour envoyer la requête à l'API Bybit
def _requete_bybit(kind: str, symbol: str, debut_ms: int, fin_ms: int) -> list:
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

#Pour télécharger une série Bybit sur [start, end] et renvoie un DataFrame trié en UTC.
#Comme les réponses vont du plus récent au plus ancien, la boucle remonte le
#temps : après chaque page, la nouvelle fin est juste avant le point le plus ancien reçu.    
def download_bybit(kind: str, symbol: str, start: str = START, end: str = END) -> pd.DataFrame:
   
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

# Pour nettoyer les données. Ce qui arrive à Bybit est une liste de textes.
#On la convertit avec cette fonction en un tableau pandas.
def clean_bybit(lignes: list, kind: str) -> pd.DataFrame:
    #Met les lignes brutes au format du projet : noms, types numériques, dates UTC.
    if kind == "klines":
        df = pd.DataFrame(lignes, columns=KLINE_COLS)
        df = df[["open_time", "high", "close", "volume", "quote_volume"]]
        colonne_temps = "open_time"
    elif kind == "funding":
        # La colonne symbol est retirée : la crypto est déjà dans le chemin du fichier
        df = pd.DataFrame(lignes, columns=["fundingRateTimestamp", "fundingRate"]).rename(
            columns={"fundingRateTimestamp": "calc_time", "fundingRate": "last_funding_rate"})
        colonne_temps = "calc_time"
    else:
        # openInterest compte les deux côtés, singleOpenInterest un seul 
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

    # Suppression des points qui apparaissent 2 fois à la frontière entre 2 pages
    df = df.drop_duplicates(colonne_temps).sort_values(colonne_temps).reset_index(drop=True)

    if kind == "funding":
        # Intervalle entre deux paiements (8, 4 ou 1 h), pour ramener le taux en horaire.
        # Le premier paiement n'a pas de précédent : on lui donne l'intervalle suivant.
        ecart = df["calc_time"].diff().dt.total_seconds() / 3600
        df["funding_interval_hours"] = ecart.round().bfill().astype("int64")
    return df

#Renvoie le chemin d'une série Bybite par exemple data/raw/bybit/klines/BTCUSDT/BTCUSDT-klines.parquet
def bybit_file(kind: str, symbol: str) -> Path:
    return raw_path("bybit", kind, symbol) / f"{symbol}-{kind}.parquet"

#Enregistre le fichier au format parquet qui est plus léger qu'un CSV et qui conserve les types après chaque lecture
def save_bybit(df: pd.DataFrame, kind: str, symbol: str) -> None:
    fichier = bybit_file(kind, symbol)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(fichier, index=False)

#Chemin de l'indice Fear & Greed : data/raw/fng/fear_greed.parquet (une seule série, pas de crypto)
FNG_FILE = RAW_DIR / "fng" / "fear_greed.parquet"


#Télécharge l'indice Fear & Greed. Pas de pagination : limit=0 renvoie tout l'historique
#(depuis février 2018) en une seule requête. On garde tout, et pas seulement START-END :
#les fenêtres glissantes (moyenne sur 30 jours...) ont besoin des jours avant START.
def download_fear_greed() -> pd.DataFrame:
    response = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Comme Bybit, l'API signale ses erreurs dans le contenu de la réponse
    if data["metadata"]["error"]:
        raise RuntimeError(f"Fear & Greed : {data['metadata']['error']}")
    return clean_fear_greed(data["data"])


#Met les données au format du projet. Chaque ligne arrive sous la forme
#{"value": "57", "value_classification": "Greed", "timestamp": "1791849600"}, tout en texte.
def clean_fear_greed(lignes: list) -> pd.DataFrame:
    # Les colonnes sont choisies explicitement : la ligne du jour contient en plus
    # "time_until_update", inutile
    df = pd.DataFrame(lignes, columns=["timestamp", "value", "value_classification"])
    df = df.rename(columns={"timestamp": "date", "value": "fng",
                            "value_classification": "fng_classe"})
    # Attention : ici le timestamp est en SECONDES (Bybit et Binance sont en millisecondes)
    df["date"] = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True)
    df["fng"] = pd.to_numeric(df["fng"])  # de 0 (peur extrême) à 100 (avidité extrême)
    # Les données arrivent du plus récent au plus ancien : on remet dans l'ordre
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


# Ci-dessous, le programme principal. N'est exécuté que si pn a lancé ce fichier directement.
if __name__ == "__main__":
    for symbol in SYMBOLS:
        for kind in BYBIT_KINDS:
            if bybit_file(kind, symbol).exists():  # reprise : déjà téléchargé
                df = pd.read_parquet(bybit_file(kind, symbol))
            else:
                df = download_bybit(kind, symbol)
                save_bybit(df, kind, symbol)
            print(f"{symbol:9} {kind:14} {len(df):6} lignes, "
                  f"{df.iloc[0, 0]} -> {df.iloc[-1, 0]}")

    if FNG_FILE.exists():
        fng = pd.read_parquet(FNG_FILE)
    else:
        fng = download_fear_greed()
        FNG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fng.to_parquet(FNG_FILE, index=False)
    print(f"{'Fear & Greed':24} {len(fng):6} lignes, "
          f"{fng['date'].iloc[0]} -> {fng['date'].iloc[-1]}")
