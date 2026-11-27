"""Collecte de l'indice Fear & Greed (alternative.me), quotidien.

Sert à tester si les squeezes se concentrent dans les phases de peur.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_fng
"""

import pandas as pd
import requests

from src.config import FNG_URL, RAW_DIR

FNG_FILE = RAW_DIR / "fng" / "fear_greed.parquet"


def download_fear_greed() -> pd.DataFrame:
    """Télécharge tout l'historique de l'indice (depuis février 2018) : limit=0 le renvoie en une seule requête."""
    response = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data["metadata"]["error"]:
        raise RuntimeError(f"Fear & Greed : {data['metadata']['error']}")
    return clean_fear_greed(data["data"])


def clean_fear_greed(lignes: list) -> pd.DataFrame:
    """Met les données au format du projet.
    Chaque ligne arrive sous la forme :
    {"value": "57", "value_classification": "Greed", "timestamp": "1791849600"}, tout en texte.
    """
    df = pd.DataFrame(lignes, columns=["timestamp", "value", "value_classification"])
    df = df.rename(columns={"timestamp": "date", "value": "fng",
                            "value_classification": "fng_classe"})
    df["date"] = pd.to_datetime(df["date"].astype("int64"), unit="s", utc=True)
    df["fng"] = pd.to_numeric(df["fng"])  # de 0 (peur extrême) à 100 (avidité extrême)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
