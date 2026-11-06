"""Collecte de l'indice Fear & Greed (alternative.me), quotidien.

Sert à tester si les squeezes se concentrent dans les phases de peur.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_fng
"""

import pandas as pd
import requests

from src.config import FNG_URL, RAW_DIR

# Chemin de l'indice Fear & Greed : data/raw/fng/fear_greed.parquet (une seule série, pas de crypto)
FNG_FILE = RAW_DIR / "fng" / "fear_greed.parquet"


def download_fear_greed() -> pd.DataFrame:
    """Télécharge l'indice Fear & Greed.

    Pas de pagination : limit=0 renvoie tout l'historique (depuis février 2018) en une
    seule requête. On garde tout, et pas seulement START-END : les fenêtres glissantes
    (moyenne sur 30 jours...) ont besoin des jours avant START.
    """
    response = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    # Comme Bybit, l'API signale ses erreurs dans le contenu de la réponse
    if data["metadata"]["error"]:
        raise RuntimeError(f"Fear & Greed : {data['metadata']['error']}")
    return clean_fear_greed(data["data"])


def clean_fear_greed(lignes: list) -> pd.DataFrame:
    """Met les données au format du projet.

    Chaque ligne arrive sous la forme
    {"value": "57", "value_classification": "Greed", "timestamp": "1791849600"}, tout en texte.
    """
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
