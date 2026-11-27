"""Choix des cryptos : les perpétuels USDT au plus gros open interest en décembre 2022.

uv run python -m scripts.choisir_cryptos   (environ 20 min pour que ça tourne)

Règle : contrat coté avant décembre 2022, classé par open interest moyen (en USDT) de décembre 2022,
gardé s'il existe encore sous le même nom en septembre 2026 et s'il a une paire spot du même nom.
Le classement se fait juste avant la période d'étude, pour ne pas choisir les cryptos qui ont monté ensuite.
"""

import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from src.collecte_binance import fetch_binance_file, read_binance_zip

LISTE_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ESPACE_XML = "{http://s3.amazonaws.com/doc/2006-03-01/}"
JOURS_DECEMBRE = ["2022-12-01", "2022-12-08", "2022-12-15", "2022-12-22", "2022-12-29"]
NOMBRE = 10


def perpetuels_usdt() -> list[str]:
    """Tous les perpétuels USDT qui ont des archives sur data.binance.vision, cotés ou non aujourd'hui."""
    symboles, marqueur = [], ""
    while True:  # la liste arrive par pages de 1 000
        reponse = requests.get(LISTE_URL, timeout=60, params={
            "delimiter": "/", "prefix": "data/futures/um/monthly/klines/", "marker": marqueur})
        racine = ET.fromstring(reponse.content)
        dossiers = [p.find(f"{ESPACE_XML}Prefix").text for p in racine.findall(f"{ESPACE_XML}CommonPrefixes")]
        symboles += [d.rstrip("/").split("/")[-1] for d in dossiers]
        if racine.find(f"{ESPACE_XML}IsTruncated").text != "true":
            break
        marqueur = dossiers[-1]
    # Les contrats trimestriels ont une date dans leur nom (BTCUSDT_230331) : on ne garde que les perpétuels
    return [s for s in symboles if re.fullmatch(r"[A-Z0-9]+USDT", s)]


def open_interest_decembre(symbole: str) -> float:
    """Open interest moyen en USDT sur cinq jours de décembre 2022 (NaN si le contrat n'existait pas)."""
    valeurs = []
    for jour in JOURS_DECEMBRE:
        contenu = fetch_binance_file(symbole, "metrics", jour)
        if contenu is not None:
            valeurs.append(read_binance_zip(contenu, "metrics")["sum_open_interest_value"].mean())
    return sum(valeurs) / len(valeurs) if valeurs else float("nan")


def donnees_completes(symbole: str) -> bool:
    """Le perpétuel et la paire spot du même nom ont des données en décembre 2022 et en septembre 2026."""
    return all(fetch_binance_file(symbole, kind, mois) is not None
               for kind in ["klines", "spot_klines"] for mois in ["2022-12", "2026-09"])


if __name__ == "__main__":
    cotes_avant = [s for s in perpetuels_usdt() if fetch_binance_file(s, "klines", "2022-11", "1d") is not None]
    print(f"{len(cotes_avant)} perpétuels USDT cotés avant décembre 2022")

    classement = pd.Series({s: open_interest_decembre(s) for s in cotes_avant}).sort_values(ascending=False)
    retenus = []
    for symbole, open_interest in classement.items():
        complet = donnees_completes(symbole)
        decision = "retenu" if complet else "exclu : données incomplètes"
        print(f"{symbole:12} {open_interest / 1e6:8.1f} M USDT  {decision}")
        if complet:
            retenus.append(symbole)
        if len(retenus) == NOMBRE:
            break
    print("\nSYMBOLS =", retenus)
