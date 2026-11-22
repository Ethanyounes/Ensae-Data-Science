"""Statistiques et graphiques descriptifs appelés par rapport.ipynb."""

from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import START
from src.panel_binance import PANEL_FILE
from src.stockage import lire_public

COULEURS = {"BTCUSDT": "#2a78d6", "ETHUSDT": "#eb6834", "SOLUSDT": "#1baf7a"}
HEURES_PAR_AN = 24 * 365

NOMS_POSITIONNEMENT = {
    "funding_8h": "Funding (% par 8 h)",
    "variation_oi_24h": "Variation OI 24 h (%)",
    "ls_global": "L/S tous comptes",
    "ls_top_comptes": "L/S gros traders, comptes",
    "ls_top_positions": "L/S gros traders, positions",
    "taker_bougies": "Ratio taker",
    "basis": "Basis (%)",
}
# Valeur de référence tracée sur chaque histogramme (0,01 % par 8 h : taux de funding par défaut de Binance)
REPERES = {"funding_8h": 0.01, "variation_oi_24h": 0, "ls_global": 1, "ls_top_comptes": 1,
           "ls_top_positions": 1, "taker_bougies": 1, "basis": 0}


def charger_panel() -> pd.DataFrame:
    """Panel horaire sur la période d'étude, lu en local s'il existe, sinon sur le S3."""
    if PANEL_FILE.exists():
        panel = pd.read_parquet(PANEL_FILE)
    else:
        panel = lire_public("interim/panel.parquet")
    # La ligne START 00:00 contient la bougie de 23 h la veille : elle est exclue, comme dans grille_horaire.
    return panel.loc[panel.index > pd.Timestamp(START, tz="UTC")]


def rendements_horaires(crypto: pd.DataFrame) -> pd.Series:
    """Rendements logarithmiques horaires du perpétuel, pour une seule crypto."""
    return np.log(crypto["close"]).diff()


def taker_ratio_bougies(crypto: pd.DataFrame) -> pd.Series:
    """Ratio achats agressifs / ventes agressives de l'heure, calculé depuis les bougies."""
    ventes_agressives = crypto["volume"] - crypto["taker_buy_volume"]
    return crypto["taker_buy_volume"] / ventes_agressives


def mois_de_la_bougie(panel: pd.DataFrame) -> pd.Index:
    """Mois de chaque ligne au format AAAA-MM, d'après le début de sa bougie."""
    # La ligne 2026-10-01 00:00 couvre la dernière heure de septembre : elle reste en septembre.
    return (panel.index - pd.Timedelta("1h")).strftime("%Y-%m").rename("mois")


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


def comparer_taker(panel: pd.DataFrame) -> pd.DataFrame:
    """Compare, par crypto, le ratio taker des bougies à celui des relevés à 5 minutes."""
    lignes = {}
    for symbole, crypto in panel.groupby("symbol"):
        bougies = taker_ratio_bougies(crypto)
        releves = crypto["taker_ratio"]
        lignes[symbole] = {
            "Médiane bougies": bougies.median(),
            "Médiane relevés": releves.median(),
            "Médiane relevés / bougies": (releves / bougies).median(),
            "Corrélation des log": np.log(bougies).corr(np.log(releves)),
            "Même côté de 1 (%)": ((bougies > 1) == (releves > 1)).mean() * 100,
        }
    return pd.DataFrame(lignes).T


def graphique_taker(panel: pd.DataFrame) -> None:
    """Nuage du ratio taker des relevés contre celui des bougies, une case par crypto."""
    nb_cryptos = panel["symbol"].nunique()
    fig, axes = plt.subplots(1, nb_cryptos, figsize=(5 * nb_cryptos, 5), sharex=True, sharey=True)

    for ax, (symbole, crypto) in zip(axes, panel.groupby("symbol")):
        ax.scatter(taker_ratio_bougies(crypto), crypto["taker_ratio"], s=1, alpha=0.15, color=COULEURS[symbole])
        ax.axline((1, 1), slope=1, color="black", linestyle="--", linewidth=1, label="Égalité")
        ax.axhline(1, color="grey", linewidth=0.8)
        ax.axvline(1, color="grey", linewidth=0.8)
        # Les quelques ratios au-delà de 3 écraseraient le nuage
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_title(symbole)
        ax.set_xlabel("Ratio taker des bougies (1 h)")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Ratio taker des relevés (moyenne des 5 min)")
    fig.tight_layout()
    plt.show()


def tableau_cryptos(panel: pd.DataFrame) -> pd.DataFrame:
    """Taille, volatilité, rendement et funding de chaque crypto sur la période d'étude."""
    lignes = {}
    for symbole, crypto in panel.groupby("symbol"):
        rendements = rendements_horaires(crypto)
        lignes[symbole] = {
            "Volume quotidien moyen (Md USDT)": crypto["quote_volume"].mean() * 24 / 1e9,
            "Open interest moyen (Md USDT)": crypto["oi_usd"].mean() / 1e9,
            "Volatilité annualisée (%)": rendements.std() * np.sqrt(HEURES_PAR_AN) * 100,
            "Rendement total (%)": (crypto["close"].iloc[-1] / crypto["close"].iloc[0] - 1) * 100,
            "Funding moyen annualisé (%)": crypto["funding_horaire"].mean() * HEURES_PAR_AN * 100,
            "Heures à funding négatif (%)": (crypto["funding_horaire"] < 0).mean() * 100,
            "Volume perpétuel / volume spot": crypto["quote_volume"].sum() / crypto["spot_quote_volume"].sum(),
        }
    return pd.DataFrame(lignes).T


def queues_rendements(panel: pd.DataFrame) -> pd.DataFrame:
    """Asymétrie, kurtosis et part des rendements horaires au-delà de 4 écarts-types, face à une loi normale."""
    lignes = {}
    for symbole, crypto in panel.groupby("symbol"):
        rendements = rendements_horaires(crypto)
        centres_reduits = (rendements - rendements.mean()) / rendements.std()
        lignes[symbole] = {
            "Asymétrie": rendements.skew(),
            "Kurtosis (excès)": rendements.kurt(),
            "Au-delà de 4 σ (%)": (centres_reduits.abs() > 4).mean() * 100,
        }
    lignes["Loi normale"] = {
        "Asymétrie": 0,
        "Kurtosis (excès)": 0,
        "Au-delà de 4 σ (%)": 2 * (1 - NormalDist().cdf(4)) * 100,
    }
    return pd.DataFrame(lignes).T


def graphique_qqplot(panel: pd.DataFrame) -> None:
    """QQ-plot des rendements horaires centrés réduits contre une loi normale."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for symbole, crypto in panel.groupby("symbol"):
        rendements = rendements_horaires(crypto).dropna()
        centres_reduits = np.sort((rendements - rendements.mean()) / rendements.std())
        probabilites = (np.arange(len(centres_reduits)) + 0.5) / len(centres_reduits)
        quantiles_normaux = [NormalDist().inv_cdf(p) for p in probabilites]
        ax.plot(quantiles_normaux, centres_reduits, ".", markersize=3, color=COULEURS[symbole], label=symbole)

    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1, label="Loi normale")
    ax.set_xlabel("Quantiles d'une loi normale")
    ax.set_ylabel("Quantiles des rendements horaires (en écarts-types)")
    ax.set_title("QQ-plot des rendements horaires")
    ax.legend()
    plt.show()


def ratio_perp_spot_mensuel(panel: pd.DataFrame) -> pd.DataFrame:
    """Volume mensuel du perpétuel rapporté au volume spot, une colonne par crypto."""
    volumes = panel.groupby([panel["symbol"], mois_de_la_bougie(panel)])[["quote_volume", "spot_quote_volume"]].sum()
    ratio = volumes["quote_volume"] / volumes["spot_quote_volume"]
    return ratio.unstack("symbol")


def graphique_perp_spot(ratio: pd.DataFrame) -> None:
    """Évolution du rapport volume perpétuel / volume spot."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for symbole in ratio.columns:
        ax.plot(pd.to_datetime(ratio.index), ratio[symbole], linewidth=2, color=COULEURS[symbole], label=symbole)

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Volume perpétuel / volume spot")
    ax.set_title("Le perpétuel échange plusieurs fois le volume du spot")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.show()


def positionnement(panel: pd.DataFrame) -> pd.DataFrame:
    """Variables de positionnement de chaque heure, dans des unités lisibles, avec la colonne symbol."""
    variables = panel[["symbol", "ls_global", "ls_top_comptes", "ls_top_positions"]].copy()
    variables["funding_8h"] = panel["funding_horaire"] * 8 * 100
    variables["variation_oi_24h"] = panel.groupby("symbol")["oi"].pct_change(24) * 100
    variables["taker_bougies"] = taker_ratio_bougies(panel)
    variables["basis"] = panel["basis"] * 100
    return variables[["symbol", *NOMS_POSITIONNEMENT]]


def resume_positionnement(variables: pd.DataFrame) -> pd.DataFrame:
    """Minimum, quantiles à 1 et 99 %, médiane et maximum de chaque variable, par crypto."""
    lignes = {}
    for nom, libelle in NOMS_POSITIONNEMENT.items():
        for symbole, crypto in variables.groupby("symbol"):
            serie = crypto[nom]
            lignes[(libelle, symbole)] = {
                "Min": serie.min(),
                "1 %": serie.quantile(0.01),
                "Médiane": serie.median(),
                "99 %": serie.quantile(0.99),
                "Max": serie.max(),
            }
    return pd.DataFrame(lignes).T


def position_du_funding(panel: pd.DataFrame) -> pd.DataFrame:
    """Part des heures (en %) où le funding est sous, au ou au-dessus du taux par défaut, par crypto et par année."""
    funding_8h = panel["funding_horaire"] * 8
    au_taux_par_defaut = np.isclose(funding_8h, 0.0001)
    position = np.select([au_taux_par_defaut, funding_8h < 0.0001], ["Au taux par défaut", "En dessous"], "Au-dessus")
    position = pd.Series(position, index=panel.index)
    annee = mois_de_la_bougie(panel).str[:4].rename("année")
    parts = position.groupby([panel["symbol"], annee]).value_counts(normalize=True).unstack(fill_value=0) * 100
    return parts[["En dessous", "Au taux par défaut", "Au-dessus"]]


def graphique_distributions(variables: pd.DataFrame) -> None:
    """Histogramme de chaque variable, une courbe par crypto, entre les quantiles à 1 et 99 %."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for ax, (nom, libelle) in zip(axes.flat, NOMS_POSITIONNEMENT.items()):
        # Sans couper les 1 % extrêmes de chaque côté, quelques valeurs aberrantes écraseraient l'histogramme
        bornes = variables[nom].quantile([0.01, 0.99])
        for symbole, crypto in variables.groupby("symbol"):
            ax.hist(crypto[nom], bins=60, range=tuple(bornes), density=True, histtype="step",
                    linewidth=1.5, color=COULEURS[symbole], label=symbole)
        ax.axvline(REPERES[nom], color="black", linestyle="--", linewidth=1)
        ax.set_title(libelle)
        ax.set_yticks([])

    axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=12)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution des variables de positionnement (densité, tirets : valeur de référence)")
    fig.tight_layout()
    plt.show()
