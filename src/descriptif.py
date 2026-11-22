"""Statistiques et graphiques descriptifs appelés par rapport.ipynb."""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import START
from src.panel_binance import PANEL_FILE
from src.stockage import lire_public


def charger_panel() -> pd.DataFrame:
    """Panel horaire sur la période d'étude, lu en local s'il existe, sinon sur le S3."""
    if PANEL_FILE.exists():
        panel = pd.read_parquet(PANEL_FILE)
    else:
        panel = lire_public("interim/panel.parquet")
    # La ligne START 00:00 contient la bougie de 23 h la veille : elle est exclue, comme dans grille_horaire.
    return panel.loc[panel.index > pd.Timestamp(START, tz="UTC")]


def mois_de_la_bougie(panel: pd.DataFrame) -> pd.Index:
    """Mois de chaque ligne au format AAAA-MM, d'après le début de sa bougie."""
    # La ligne 2026-10-01 00:00 couvre la dernière heure de septembre : elle reste en septembre.
    return (panel.index - pd.Timedelta("1h")).strftime("%Y-%m")


def manquants_par_mois(panel: pd.DataFrame) -> pd.DataFrame:
    """Part des heures (en %) où au moins une colonne manque, par crypto et par mois."""
    heure_incomplete = panel.drop(columns="symbol").isna().any(axis=1)
    return heure_incomplete.groupby([panel["symbol"], mois_de_la_bougie(panel)]).mean().unstack() * 100


def graphique_manquants(manquants: pd.DataFrame) -> None:
    """Carte crypto × mois de la part d'heures incomplètes."""
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(manquants) + 1.5))
    image = ax.imshow(manquants, cmap="Blues", vmin=0, aspect="auto")

    ax.set_yticks(range(len(manquants)), manquants.index)
    tous_les_6_mois = range(0, manquants.shape[1], 6)
    ax.set_xticks(tous_les_6_mois, manquants.columns[tous_les_6_mois])
    fig.colorbar(image, ax=ax, label="Heures incomplètes (%)")
    ax.set_title("Heures où au moins une variable manque, par mois")
    plt.show()
