"""Nettoyage de l'indice Fear & Greed et rattachement au panel horaire.

L'indice est quotidien : chaque heure t reçoit la dernière valeur publiée avant t.
"""

import pandas as pd

from src.collecte_fng import FNG_FILE


def clean_fng(df: pd.DataFrame) -> pd.DataFrame:
    """Indice quotidien, avec l'heure t_fng à partir de laquelle chaque valeur est utilisée.
    La valeur du jour J est publiée vers minuit, sans garantie qu'elle soit connue à 00:00 :
    on ne l'utilise qu'à partir de J+1 00:00.
    """
   
    df = df[["date", "fng"]].rename(columns={"date": "fng_date"})
    df["fng_date"] = df["fng_date"].dt.as_unit("ms")
    df["fng"] = df["fng"].where(df["fng"].between(0, 100))
    df = df.drop_duplicates("fng_date").sort_values("fng_date")
    df["t_fng"] = (df["fng_date"] + pd.Timedelta("1D")).dt.as_unit("ms")
    return df.dropna(subset="fng")


def ajouter_fng(panel: pd.DataFrame) -> pd.DataFrame:
    """Panel d'une crypto complété de la dernière valeur de l'indice utilisable à chaque heure t.
    Un jour manquant (2024-10-26) reprend la valeur précédente ; fng_date dit de quel jour elle date.
    """
    fng = clean_fng(pd.read_parquet(FNG_FILE))
    panel = pd.merge_asof(panel.reset_index(), fng, left_on="t", right_on="t_fng")
    return panel.drop(columns="t_fng").set_index("t")


def controles_fng(df: pd.DataFrame, debut: str, fin: str) -> pd.Series:
    """Contrôles de l'indice brut : couverture, jours manquants, valeurs impossibles et doublons."""
    jours = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    manquants = jours.difference(df["date"])
    periode = pd.date_range(debut, fin, freq="D", tz="UTC")
    return pd.Series({
        "jours": len(df),
        "premier_jour": df["date"].min().date(),
        "dernier_jour": df["date"].max().date(),
        "jours_manquants": len(manquants),
        "jours_manquants_periode": len(manquants.intersection(periode)),
        "valeurs_hors_0_100": (~df["fng"].between(0, 100)).sum(),
        "dates_en_double": df["date"].duplicated().sum(),
    })
