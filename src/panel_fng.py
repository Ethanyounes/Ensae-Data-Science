"""Ici, on nettoie l'indice Fear & Greed et on le rattache au panel horaire.

L'indice est quotidien : chaque heure t reçoit la dernière valeur publiée avant t.
"""

import pandas as pd

from src.collecte_fng import FNG_FILE

#On trie les colonnes pour ne garder que les utiles, on harmonise l'unité de la date et 
#on écarte une valeur si elle est impossible et on supptime les doublons.
def clean_fng(df: pd.DataFrame) -> pd.DataFrame:
    #Les valeurs sont quotidiennes avec l'heure à partir de laquelle chacune est connue (t_fng).
    #La valeur datée du jour J (00:00 UTC) est publiée vers minuit, sans garantie qu'elle
    #soit connue à 00:00 précisément : on ne l'utilise qu'à partir de J+1 00:00.
    
    # fng_classe n'est pas gardée : elle se déduit de fng
    df = df[["date", "fng"]].rename(columns={"date": "fng_date"})
    df["fng_date"] = df["fng_date"].dt.as_unit("ms")
    # Un indice hors de [0, 100] est impossible : erreur d'enregistrement
    df["fng"] = df["fng"].where(df["fng"].between(0, 100))
    df = df.drop_duplicates("fng_date").sort_values("fng_date")
    # as_unit("ms") après l'addition : le décalage repasse la date en microsecondes
    df["t_fng"] = (df["fng_date"] + pd.Timedelta("1D")).dt.as_unit("ms")
    return df.dropna(subset="fng")

#Le problème : le panel a une ligne par heure, alors que l'indice n'a qu'une valeur par jour. Il faut donner une valeur à chacune des 24 heures.
#On ajoute à un panel d'une crypto la dernière valeur connue à t.
#Un jour manquant (2024-10-26) n'a pas de ligne : les heures du lendemain reçoivent
#donc la valeur précédente, et fng_date indique de quel jour date la valeur utilisée.

def ajouter_fng(panel: pd.DataFrame) -> pd.DataFrame:
    fng = clean_fng(pd.read_parquet(FNG_FILE))
    #l'indice est quotidien et le panel horaire, donc chaque heure reçoit la 
    # dernière valeur connue (merge_asof). Cette valeur est celle de la veille, par précaution contre la fuite d'information.
    panel = pd.merge_asof(panel.reset_index(), fng, left_on="t", right_on="t_fng")
    return panel.drop(columns="t_fng").set_index("t")

#On code ici une fonction de contrôles de l'indice brut :couverture, jours manquants, valeurs impossibles.
def controles_fng(df: pd.DataFrame, debut: str, fin: str) -> pd.Series:
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
