"""Statistiques et graphiques descriptifs appelés par rapport.ipynb."""

from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cible import HORIZON, ajouter_cible
from src.config import START, SYMBOLS
from src.panel_binance import PANEL_FILE
from src.panel_bybit import PANEL_FUSION_FILE
from src.stockage import lire_public

COULEURS = {"BTCUSDT": "#2a78d6", "ETHUSDT": "#eb6834", "SOLUSDT": "#1baf7a", "BNBUSDT": "#e0b400",
            "DOGEUSDT": "#a0522d", "LTCUSDT": "#7f7f7f", "XRPUSDT": "#17becf", "DOTUSDT": "#d6338a",
            "LINKUSDT": "#3f3fbf", "ADAUSDT": "#8fbf3f"}
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
EVENEMENTS = {"2024-01-10": "ETF bitcoin au comptant", "2024-11-05": "Élection américaine",
              "2025-10-10": "Krach du 10 octobre"}


def lire_interim(fichier: Path) -> pd.DataFrame:
    """Parquet de data/interim/, lu en local s'il existe, sinon sur le S3."""
    if fichier.exists():
        return pd.read_parquet(fichier)
    return lire_public(f"interim/{fichier.name}")


def charger_panel(avec_chauffe: bool = False, symboles: list[str] = SYMBOLS) -> pd.DataFrame:
    """Panel horaire des cryptos demandées. Sans la chauffe par défaut."""
    panel = lire_interim(PANEL_FILE)
    panel = panel[panel["symbol"].isin(symboles)]
    # La cible a besoin de la chauffe : σ est calculé sur les 30 jours qui précèdent t
    if avec_chauffe:
        return panel
    return periode_etude(panel)


def periode_etude(panel: pd.DataFrame) -> pd.DataFrame:
    """Lignes de la période d'étude, sans la chauffe."""
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


def graphique_qqplot(panel: pd.DataFrame, variable=rendements_horaires, nom: str = "rendements horaires") -> None:
    """QQ-plot d'une variable centrée réduite contre une loi normale, une couleur par crypto."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for symbole, crypto in panel.groupby("symbol"):
        valeurs = variable(crypto).dropna()
        centres_reduits = np.sort((valeurs - valeurs.mean()) / valeurs.std())
        probabilites = (np.arange(len(centres_reduits)) + 0.5) / len(centres_reduits)
        quantiles_normaux = [NormalDist().inv_cdf(p) for p in probabilites]
        ax.plot(quantiles_normaux, centres_reduits, ".", markersize=3, color=COULEURS[symbole], label=symbole)

    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1, label="Loi normale")
    ax.set_xlabel("Quantiles d'une loi normale")
    ax.set_ylabel(f"Quantiles des {nom} (en écarts-types)")
    ax.set_title(f"QQ-plot des {nom}")
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


def medianes_par_an(variables: pd.DataFrame) -> pd.DataFrame:
    """Médiane de chaque variable de positionnement, par crypto et par année."""
    annee = mois_de_la_bougie(variables).str[:4].rename("année")
    medianes = variables.groupby([variables["symbol"], annee]).median()
    return medianes.rename(columns=NOMS_POSITIONNEMENT)


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


def graphique_series(panel: pd.DataFrame, variables: pd.DataFrame) -> None:
    """Moyennes quotidiennes du prix, de l'open interest, du funding et de deux ratios long/short."""
    lignes = {
        "Prix (base 100, échelle log)": panel["close"],
        "Open interest en USDT (base 100, échelle log)": panel["oi_usd"],
        "Funding (% par 8 h, axe coupé à −0,06 %)": variables["funding_8h"],
        "Long/short, tous les comptes": variables["ls_global"],
        "Long/short, gros traders en positions": variables["ls_top_positions"],
    }
    fig, axes = plt.subplots(len(lignes), 1, figsize=(13, 15), sharex=True)

    for ax, (titre, serie) in zip(axes, lignes.items()):
        for symbole in panel["symbol"].unique():
            par_jour = serie[panel["symbol"] == symbole].resample("D").mean()
            if "base 100" in titre:
                par_jour = 100 * par_jour / par_jour.iloc[0]
            ax.plot(par_jour.index, par_jour, linewidth=1.2, color=COULEURS[symbole], label=symbole)
        if "base 100" in titre:
            ax.set_yscale("log")
        for date in EVENEMENTS:
            ax.axvline(pd.Timestamp(date, tz="UTC"), color="grey", linestyle=":", linewidth=1)
        ax.set_title(titre, loc="left")
        ax.grid(axis="y", alpha=0.3)

    for date, evenement in EVENEMENTS.items():
        axes[0].annotate(evenement, (pd.Timestamp(date, tz="UTC"), 1), xycoords=("data", "axes fraction"),
                         xytext=(3, -12), textcoords="offset points", fontsize=9, color="grey")
    axes[2].axhline(0.01, color="black", linestyle="--", linewidth=1)
    # Deux pics de SOL (janvier 2023 et octobre 2025) descendent sous −0,2 % et écraseraient la série
    axes[2].set_ylim(-0.06, 0.11)
    axes[3].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[4].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    plt.show()


def graphique_correlations(variables: pd.DataFrame) -> None:
    """Corrélations de rang (Spearman) entre variables de positionnement, une carte par crypto."""
    nb_cryptos = variables["symbol"].nunique()
    fig, axes = plt.subplots(1, nb_cryptos, figsize=(6 * nb_cryptos, 6.5), layout="constrained")
    libelles = list(NOMS_POSITIONNEMENT.values())

    for ax, (symbole, crypto) in zip(axes, variables.groupby("symbol")):
        correlations = crypto.drop(columns="symbol").corr(method="spearman")
        image = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(correlations)):
            for j in range(len(correlations)):
                ax.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(libelles)), libelles, rotation=45, ha="right")
        ax.set_yticks(range(len(libelles)), libelles if ax is axes[0] else [])
        ax.set_title(symbole)

    fig.colorbar(image, ax=axes, shrink=0.7, label="Corrélation de Spearman")
    plt.show()


def correlations_rendement_passe(panel: pd.DataFrame, variables: pd.DataFrame) -> pd.DataFrame:
    """Corrélation de Spearman de chaque variable avec le rendement des 7 derniers jours, par crypto."""
    colonnes = {}
    for symbole, crypto in panel.groupby("symbol"):
        rendement_7j = np.log(crypto["close"]).diff(24 * 7)
        variables_crypto = variables.loc[variables["symbol"] == symbole]
        colonnes[symbole] = {libelle: rendement_7j.corr(variables_crypto[nom], method="spearman")
                             for nom, libelle in NOMS_POSITIONNEMENT.items()}
    return pd.DataFrame(colonnes)


def persistance(variables: pd.DataFrame) -> pd.DataFrame:
    """Autocorrélation de chaque variable à 1 h, 24 h et 7 jours, en moyenne sur les cryptos."""
    lignes = []
    for _, crypto in variables.groupby("symbol"):
        for nom, libelle in NOMS_POSITIONNEMENT.items():
            lignes.append({
                "Variable": libelle,
                "1 h": crypto[nom].autocorr(1),
                "24 h": crypto[nom].autocorr(24),
                "7 jours": crypto[nom].autocorr(24 * 7),
            })
    return pd.DataFrame(lignes).groupby("Variable", sort=False).mean()


# Étape 3 : la cible

# Chaque variante change un seul seuil de la cible de base (voir src/cible.py)
VARIANTES = {
    "Base": {},
    "k = 2,5": {"k": 2.5},
    "k = 4": {"k": 4},
    "Plancher 3 %": {"plancher": 0.03},
    "Plancher 8 %": {"plancher": 0.08},
    "δ = 5 %": {"delta": 0.05},
    "δ = 15 %": {"delta": 0.15},
    "δ = 1,2 σ": {"delta_sigma": 1.2},
    "δ = 1,5 σ": {"delta_sigma": 1.5},
    "Horizon 12 h": {"horizon": 12},
    "Horizon 48 h": {"horizon": 48},
}
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def tableau_episodes(panel_cible: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Une ligne par épisode, décrit par son heure à la plus forte hausse."""
    # Deux heures positives à moins d'un horizon d'écart ont des fenêtres qui se chevauchent :
    # elles décrivent la même hausse, donc le même épisode.
    positives = panel_cible[panel_cible["shortsqueeze"].fillna(False)].reset_index()
    nouvel_episode = (positives["symbol"] != positives["symbol"].shift()) | \
                     (positives["t"].diff() >= pd.Timedelta(hours=horizon))
    groupes = positives.groupby(nouvel_episode.cumsum())

    plus_forte_hausse = positives.loc[groupes["hausse_pic"].idxmax()].set_index(groupes.size().index)
    episodes_cible = pd.DataFrame({
        "symbol": plus_forte_hausse["symbol"],
        "debut": groupes["t"].first(),
        "fin": groupes["t"].last(),
        "heures_positives": groupes.size(),
        "date_pic": plus_forte_hausse["date_pic"],
        "delai_pic_h": plus_forte_hausse["delai_pic_h"],
        "hausse_pic_pct": 100 * plus_forte_hausse["hausse_pic"],
        "variation_oi_pct": 100 * plus_forte_hausse["variation_oi"],
        "ratio_achats_ventes": plus_forte_hausse["ratio_achats_ventes"],
    })
    return episodes_cible.reset_index(drop=True)


def frequence_cible(panel_cible: pd.DataFrame, episodes_cible: pd.DataFrame) -> pd.DataFrame:
    """Heures positives (nombre et %) et nombre d'épisodes, par crypto et par année, puis au total."""
    annee = mois_de_la_bougie(panel_cible).str[:4].rename("année")
    frequence = panel_cible.groupby([panel_cible["symbol"], annee])["shortsqueeze"].agg(
        heures="count", heures_positives="sum")
    annee_episode = (episodes_cible["debut"] - pd.Timedelta("1h")).dt.strftime("%Y").rename("année")
    frequence["episodes"] = episodes_cible.groupby([episodes_cible["symbol"], annee_episode]).size()
    frequence["episodes"] = frequence["episodes"].fillna(0)

    total = frequence.groupby("symbol").sum()
    total.index = pd.MultiIndex.from_product([total.index, ["Total"]], names=["symbol", "année"])
    frequence = pd.concat([frequence, total]).sort_index().astype("int64")
    # Part des heures positives : c'est la précision d'un modèle qui prédirait au hasard (base naïve de la PR-AUC)
    frequence["part_positive_pct"] = 100 * frequence["heures_positives"] / frequence["heures"]
    return frequence


def entonnoir(panel_cible: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'heures qui passent chaque étape C1 → C1+C2 → C1+C2+C3, par crypto."""
    connues = panel_cible.dropna(subset="shortsqueeze")
    c1 = connues["c1"]
    etapes = pd.DataFrame({
        "C1 :\nforte hausse": c1,
        "C1 et\nopen interest en hausse": c1 & (connues["variation_oi"] > 0),
        "C1 + C2 :\nbaisse de l'open interest": c1 & connues["c2"],
        "C1 + C2 + C3 :\nshort squeeze": connues["shortsqueeze"],
    })
    return etapes.groupby(connues["symbol"]).sum()


def graphique_entonnoir(nombres: pd.DataFrame) -> None:
    """Part des heures C1 (en %) qui passent chaque étape, une barre par crypto."""
    parts = 100 * nombres.div(nombres.iloc[:, 0], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    largeur = 0.8 / len(parts)
    for i, (symbole, ligne) in enumerate(parts.iterrows()):
        positions = np.arange(len(ligne)) + i * largeur
        barres = ax.bar(positions, ligne, width=largeur, color=COULEURS[symbole], label=symbole)
        ax.bar_label(barres, labels=[f"{n:,}".replace(",", " ") for n in nombres.loc[symbole]], fontsize=8)

    ax.set_xticks(np.arange(len(parts.columns)) + 0.4 - largeur / 2, parts.columns)
    ax.set_ylabel("Part des heures C1 (%)")
    ax.set_title("Parmi les fortes hausses, combien sont des rachats de shorts ? (nombre d'heures au-dessus des barres)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.show()


def simultaneite(panel_cible: pd.DataFrame) -> pd.DataFrame:
    """Parmi les heures positives d'une crypto (ligne), part (en %) où une autre crypto (colonne) l'est aussi."""
    positives = (panel_cible.reset_index()
                 .pivot(index="t", columns="symbol", values="shortsqueeze")
                 .fillna(False).astype("int64"))
    ensemble = positives.T @ positives  # nombre d'heures positives en même temps, pour chaque paire
    parts = 100 * ensemble.div(np.diag(ensemble), axis=0)
    au_moins_une_autre = positives.sum(axis=1) >= 2
    parts["Au moins une autre"] = [100 * au_moins_une_autre[positives[s] == 1].mean() for s in positives]
    return parts


def graphique_calendrier(episodes_cible: pd.DataFrame) -> None:
    """Heure UTC et jour de la semaine du pic de chaque épisode, toutes cryptos confondues."""
    # Le pic a lieu pendant la bougie qui finit à date_pic : on prend l'heure de début de cette bougie
    debut_bougie_pic = episodes_cible["date_pic"] - pd.Timedelta("1h")
    heures = debut_bougie_pic.dt.hour.value_counts().reindex(range(24), fill_value=0)
    jours = debut_bougie_pic.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)

    fig, (ax_heure, ax_jour) = plt.subplots(1, 2, figsize=(14, 4), width_ratios=[2, 1])
    ax_heure.bar(heures.index, heures, color="grey")
    for heure_funding in [0, 8, 16]:
        ax_heure.axvline(heure_funding - 0.5, color="black", linestyle="--", linewidth=1)
    ax_heure.set_xticks(range(0, 24, 2))
    ax_heure.set_xlabel("Heure UTC du pic (tirets : paiements de funding)")
    ax_heure.set_ylabel("Nombre d'épisodes")

    ax_jour.bar(JOURS, jours, color="grey")
    ax_jour.tick_params(axis="x", rotation=45)
    ax_jour.set_xlabel("Jour du pic")
    fig.suptitle(f"Quand ont lieu les pics des {len(episodes_cible)} épisodes ?")
    fig.tight_layout()
    plt.show()


def profil_episodes(episodes_cible: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'épisodes et valeurs médianes de leurs heures positives, de leur pic et de la baisse d'open interest."""
    colonnes = ["heures_positives", "delai_pic_h", "hausse_pic_pct", "variation_oi_pct", "ratio_achats_ventes"]
    profil = episodes_cible.groupby("symbol")[colonnes].median()
    profil.insert(0, "episodes", episodes_cible.groupby("symbol").size())
    return profil


def variation_oi_reduite(crypto: pd.DataFrame) -> pd.Series:
    """Variation de l'open interest sur l'horizon, divisée par son écart-type des 30 derniers jours."""
    return crypto["oi"].pct_change(HORIZON, fill_method=None) / crypto["sigma_oi_30j"]


def seuils_baisse_oi(panel_cible: pd.DataFrame) -> pd.DataFrame:
    """Fréquence des baisses d'open interest exprimées en écarts-types, face à une loi normale, puis selon δ."""
    lignes = {}
    for symbole, crypto in panel_cible.groupby("symbol"):
        reduite = variation_oi_reduite(crypto).dropna()
        baisse_au_pic = -crypto["variation_oi"]  # de t au pic, ce que teste C2
        lignes[symbole] = {
            "Kurtosis (excès)": reduite.kurt(),
            "Sous −1,5 σ (%)": (reduite <= -1.5).mean() * 100,
            "Sous −3 σ (%)": (reduite <= -3).mean() * 100,
            "Baisse au pic ≥ 5 % (%)": (baisse_au_pic >= 0.05).mean() * 100,
            "Baisse au pic ≥ 1,5 σ (%)": (baisse_au_pic >= 1.5 * crypto["sigma_oi_30j"]).mean() * 100,
        }
    lignes["Loi normale"] = {
        "Kurtosis (excès)": 0,
        "Sous −1,5 σ (%)": NormalDist().cdf(-1.5) * 100,
        "Sous −3 σ (%)": NormalDist().cdf(-3) * 100,
    }
    return pd.DataFrame(lignes).T


def sensibilite(panel_complet: pd.DataFrame) -> pd.DataFrame:
    """Part des heures positives (en %) et nombre d'épisodes pour chaque variante de la cible, par crypto."""
    lignes = {}
    for nom, seuils in VARIANTES.items():
        panel_cible = periode_etude(ajouter_cible(panel_complet, **seuils))
        episodes_cible = tableau_episodes(panel_cible, seuils.get("horizon", HORIZON))
        for symbole, crypto in panel_cible.groupby("symbol"):
            lignes[(nom, symbole)] = {
                "Heures positives (%)": 100 * crypto["shortsqueeze"].mean(),
                "Épisodes": (episodes_cible["symbol"] == symbole).sum(),
            }
    return pd.DataFrame(lignes).T.unstack().reindex(list(VARIANTES))


def graphique_episode(panel_cible: pd.DataFrame, symbole: str, debut: str) -> None:
    """Prix, open interest, funding et ratio long/short de 3 jours avant à 1 jour après le pic d'un épisode."""
    crypto = panel_cible.loc[panel_cible["symbol"] == symbole]
    debut = pd.Timestamp(debut, tz="UTC")
    pic = crypto.loc[debut, "date_pic"]
    fenetre = crypto.loc[debut - pd.Timedelta("3D"): pic + pd.Timedelta("1D")]

    series = {
        "Prix (USDT)": fenetre["close"],
        "Open interest (en crypto)": fenetre["oi"],
        "Funding (% par 8 h)": fenetre["funding_horaire"] * 8 * 100,
        "L/S tous comptes": fenetre["ls_global"],
    }
    fig, axes = plt.subplots(len(series), 1, figsize=(13, 10), sharex=True)
    for ax, (titre, serie) in zip(axes, series.items()):
        ax.plot(serie.index, serie, color=COULEURS[symbole])
        ax.axvline(debut, color="crimson", linestyle="--", linewidth=1)
        ax.axvline(pic, color="black", linestyle="--", linewidth=1)
        ax.set_title(titre, loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[2].axhline(0.01, color="grey", linestyle=":", linewidth=1)
    axes[3].axhline(1, color="grey", linestyle=":", linewidth=1)
    fig.suptitle(f"{symbole} : épisode qui commence le {debut:%Y-%m-%d %H:%M} (rouge), pic le {pic:%Y-%m-%d %H:%M} (noir)")
    fig.tight_layout()
    plt.show()

# étape 4 : positionnement et squeezes, première réponse à H1/H2

NOMS_ETAPE_4 = {
    "funding_8h": "Funding (% par 8 h)",
    "basis_pct": "Basis (%)",
    "ls_global": "L/S tous comptes",
    "ls_top_positions": "L/S gros traders, positions",
    "taker_24h": "Ratio taker sur 24 h",
    "variation_oi_24h": "Variation OI 24 h (%)",
    "variation_oi_7j": "Variation OI 7 jours (%)",
}
NOMS_PRIX = {
    "rendement_7j": "Rendement 7 jours (%)",
    "volatilite_30j": "Volatilité quotidienne 30 jours (%)",
    "ecart_plus_haut_30j": "Écart au plus haut de 30 jours (%)",
}


def variables_etape_4(panel_cible_complet: pd.DataFrame) -> pd.DataFrame:
    """Panel cible complété des variables de positionnement et de prix connues à t, sur la période d'étude."""
    panel = panel_cible_complet.copy()
    par_crypto = panel.groupby("symbol")
    panel["funding_8h"] = panel["funding_horaire"] * 8 * 100
    panel["basis_pct"] = panel["basis"] * 100
    panel["variation_oi_24h"] = 100 * par_crypto["oi"].pct_change(24, fill_method=None)
    panel["variation_oi_7j"] = 100 * par_crypto["oi"].pct_change(24 * 7, fill_method=None)
    # Le ratio taker d'une heure n'a pas de mémoire (étape 2) : on le calcule sur les 24 dernières heures
    achats_24h = par_crypto["taker_buy_volume"].transform(lambda s: s.rolling(24).sum())
    volume_24h = par_crypto["volume"].transform(lambda s: s.rolling(24).sum())
    panel["taker_24h"] = achats_24h / (volume_24h - achats_24h)

    panel["rendement_7j"] = 100 * par_crypto["close"].pct_change(24 * 7, fill_method=None)
    panel["volatilite_30j"] = 100 * panel["sigma_30j"] * np.sqrt(24)
    plus_haut_30j = par_crypto["high"].transform(lambda s: s.rolling("30D").max())
    panel["ecart_plus_haut_30j"] = 100 * (panel["close"] / plus_haut_30j - 1)
    return periode_etude(panel)


def evenements_de_marche(episodes_cible: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les épisodes dont les pics sont à moins de 24 h d'écart, toutes cryptos confondues."""
    episodes = episodes_cible.sort_values("date_pic")
    nouvel_evenement = episodes["date_pic"].diff() >= pd.Timedelta(hours=HORIZON)
    evenements = episodes.groupby(nouvel_evenement.cumsum()).agg(
        premier_pic=("date_pic", "first"),
        cryptos=("symbol", "nunique"),
        liste=("symbol", lambda s: ", ".join(sorted(set(s.str.replace("USDT", ""))))),
    )
    return evenements.reset_index(drop=True)


def numero_decile(valeurs: pd.Series) -> pd.Series:
    """Décile (1 à 10) de chaque valeur. Les ex-aequo restent ensemble : un décile peut être vide ou plus gros."""
    decile = pd.Series(10, index=valeurs.index)
    for d in range(9, 0, -1):
        decile[valeurs <= valeurs.quantile(d / 10)] = d
    return decile


def taux_squeeze_par_decile(panel: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Nombre d'heures, heures positives et taux de squeeze (%) par crypto et par décile de la variable."""
    donnees = panel[["symbol", variable, "shortsqueeze"]].dropna()
    donnees["shortsqueeze"] = donnees["shortsqueeze"].astype("int64")
    donnees["decile"] = donnees.groupby("symbol")[variable].transform(numero_decile)

    resultats = donnees.groupby(["symbol", "decile"])["shortsqueeze"].agg(heures="count", heures_positives="sum")
    resultats["taux_pct"] = 100 * resultats["heures_positives"] / resultats["heures"]
    return resultats.reset_index()


def rendement_par_decile(panel: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Quantiles 5 %, 50 % et 95 % du rendement à 24 h, par crypto et par décile de la variable."""
    donnees = panel[["symbol", "close", variable]].copy()
    prix_dans_24h = donnees.groupby("symbol")["close"].shift(-24)
    donnees["rendement_24h"] = 100 * (prix_dans_24h / donnees["close"] - 1)
    donnees = donnees.dropna(subset=[variable, "rendement_24h"])
    donnees["decile"] = donnees.groupby("symbol")[variable].transform(numero_decile)

    rendements = donnees.groupby(["symbol", "decile"])["rendement_24h"]
    resultats = pd.DataFrame({
        "5 %": rendements.quantile(0.05),
        "Médiane": rendements.median(),
        "95 %": rendements.quantile(0.95),
    })
    return resultats.reset_index()


def graphique_taux_par_decile(taux: pd.DataFrame, nom_variable: str) -> None:
    """Carte crypto × décile du taux de squeeze, avec une ligne pour l'ensemble des cryptos."""
    carte = taux.pivot(index="symbol", columns="decile", values="taux_pct").reindex(columns=range(1, 11))
    carte = carte.reindex([s for s in SYMBOLS if s in carte.index])
    # Les déciles sont calculés crypto par crypto : chaque crypto pèse autant dans la ligne « Ensemble »
    ensemble = taux.groupby("decile")[["heures_positives", "heures"]].sum()
    carte.loc["Ensemble"] = 100 * ensemble["heures_positives"] / ensemble["heures"]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(carte) + 1.5))
    image = ax.imshow(carte, cmap="Reds", vmin=0, aspect="auto")
    for i in range(carte.shape[0]):
        for j in range(carte.shape[1]):
            if not np.isnan(carte.iloc[i, j]):
                ax.text(j, i, f"{carte.iloc[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(10), range(1, 11))
    ax.set_yticks(range(len(carte)), carte.index)
    ax.set_xlabel(f"Décile de {nom_variable} (1 = valeurs les plus basses)")
    fig.colorbar(image, ax=ax, label="Heures suivies d'un squeeze (%)")
    ax.set_title(f"Taux de squeeze selon {nom_variable}")
    fig.tight_layout()
    plt.show()


def graphique_rendement_par_decile(rendements: pd.DataFrame, nom_variable: str) -> None:
    """Quantiles 5 %, 50 % et 95 % du rendement à 24 h selon le décile, une case par crypto."""
    symboles = [s for s in SYMBOLS if s in set(rendements["symbol"])]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for ax, symbole in zip(axes.flat, symboles):
        crypto = rendements[rendements["symbol"] == symbole]
        for quantile, couleur in [("95 %", "#1baf7a"), ("Médiane", "black"), ("5 %", "#d62728")]:
            ax.plot(crypto["decile"], crypto[quantile], marker="o", markersize=3, color=couleur, label=quantile)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_title(symbole)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(f"Décile de {nom_variable}")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rendement à 24 h (%)")
    axes[0, 0].legend()
    fig.suptitle(f"Rendement des 24 h suivantes selon {nom_variable}")
    fig.tight_layout()
    plt.show()


def charger_fusion() -> pd.DataFrame:
    """Panel Binance complété des colonnes Bybit (suffixe _bybit), sur la période d'étude."""
    return periode_etude(lire_interim(PANEL_FUSION_FILE))


def comparer_bybit(fusion: pd.DataFrame) -> pd.DataFrame:
    """Accord entre Binance et Bybit, par crypto : prix, open interest sur 24 h et funding."""
    lignes = {}
    for symbole, crypto in fusion.groupby("symbol"):
        variation_oi = crypto["oi"].pct_change(24, fill_method=None)
        variation_oi_bybit = crypto["oi_bybit"].pct_change(24, fill_method=None)
        lignes[symbole] = {
            "Corrélation des rendements horaires": crypto["close"].pct_change(fill_method=None).corr(
                crypto["close_bybit"].pct_change(fill_method=None)),
            "Corrélation des variations d'OI sur 24 h": variation_oi.corr(variation_oi_bybit),
            "Corrélation du funding": crypto["funding_horaire"].corr(crypto["funding_horaire_bybit"]),
            "Écart de prix moyen (%)": 100 * (crypto["close"] / crypto["close_bybit"] - 1).abs().mean(),
            "Heures sans OI Bybit (%)": 100 * crypto["oi_bybit"].isna().mean(),
        }
    return pd.DataFrame(lignes).T.reindex([s for s in SYMBOLS if s in lignes])


def episodes_sur_bybit(fusion: pd.DataFrame, episodes_cible: pd.DataFrame) -> pd.DataFrame:
    """Par crypto : baisse médiane de l'open interest entre t et le pic sur les deux plateformes, et part des
    épisodes où l'open interest Bybit baisse aussi."""
    oi_bybit = fusion.reset_index()[["t", "symbol", "oi_bybit"]]
    episodes = episodes_cible[["symbol", "date_pic", "delai_pic_h", "variation_oi_pct"]].copy()
    episodes["t"] = episodes["date_pic"] - pd.to_timedelta(episodes["delai_pic_h"], unit="h")
    episodes = episodes.merge(oi_bybit, on=["t", "symbol"])
    episodes = episodes.merge(oi_bybit.rename(columns={"t": "date_pic", "oi_bybit": "oi_bybit_pic"}),
                              on=["date_pic", "symbol"])
    episodes["variation_oi_bybit_pct"] = 100 * (episodes["oi_bybit_pic"] / episodes["oi_bybit"] - 1)

    resume = episodes.groupby("symbol").agg(
        episodes=("symbol", "size"),
        variation_oi_binance_mediane_pct=("variation_oi_pct", "median"),
        variation_oi_bybit_mediane_pct=("variation_oi_bybit_pct", "median"),
    )
    resume["baisse_aussi_sur_bybit_pct"] = 100 * (episodes["variation_oi_bybit_pct"] < 0).groupby(
        episodes["symbol"]).mean()
    return resume.reindex([s for s in SYMBOLS if s in resume.index])


HEURES_AUTOUR_DU_PIC = range(-72, 25)
COULEURS_GROUPES = ["#d62728", "#ff9f1c", "grey"]


def groupes_etude_evenement(panel_cible: pd.DataFrame, episodes_cible: pd.DataFrame) -> dict:
    """Pics des squeezes, pics des fortes hausses sans baisse d'open interest, et heures ordinaires tirées au hasard."""
    hausses_sans_rachat = tableau_episodes(panel_cible.assign(shortsqueeze=panel_cible["c1"] & ~panel_cible["c2"]))
    ordinaires = panel_cible[panel_cible["c1"].eq(False)].sample(1000, random_state=0)
    return {
        f"Squeezes ({len(episodes_cible)})": episodes_cible[["symbol", "date_pic"]],
        f"Fortes hausses sans rachat, C1 sans C2 ({len(hausses_sans_rachat)})": hausses_sans_rachat[
            ["symbol", "date_pic"]],
        "Heures ordinaires (1 000, tirées au hasard)": pd.DataFrame(
            {"symbol": ordinaires["symbol"], "date_pic": ordinaires.index}),
    }


def fenetre_autour_du_pic(crypto: pd.DataFrame, pic: pd.Timestamp) -> pd.DataFrame:
    """Variables de 72 h avant à 24 h après le pic ; les niveaux qui diffèrent d'une crypto à l'autre sont
    exprimés en variation depuis 72 h avant le pic."""
    fenetre = crypto.reindex(pic + pd.to_timedelta(list(HEURES_AUTOUR_DU_PIC), unit="h"))
    depart = fenetre.iloc[0]
    variables = pd.DataFrame({
        "Prix (% depuis −72 h)": 100 * (fenetre["close"] / depart["close"] - 1),
        "Open interest (% depuis −72 h)": 100 * (fenetre["oi"] / depart["oi"] - 1),
        "Funding (% par 8 h)": 100 * 8 * fenetre["funding_horaire"],
        "L/S tous comptes (% depuis −72 h)": 100 * (fenetre["ls_global"] / depart["ls_global"] - 1),
        "L/S gros traders, positions (% depuis −72 h)": 100 * (
            fenetre["ls_top_positions"] / depart["ls_top_positions"] - 1),
        "Ratio taker de l'heure": taker_ratio_bougies(fenetre),
    })
    return variables.set_index(pd.Index(HEURES_AUTOUR_DU_PIC, name="Heures depuis le pic"))


def etude_evenement(panel_complet: pd.DataFrame, groupes: dict) -> pd.DataFrame:
    """Moyenne de chaque variable heure par heure autour des pics, pour chaque groupe (colonnes : groupe, variable)."""
    cryptos = dict(tuple(panel_complet.groupby("symbol")))
    profils = {}
    for nom, pics in groupes.items():
        fenetres = [fenetre_autour_du_pic(cryptos[s], p) for s, p in zip(pics["symbol"], pics["date_pic"])]
        profils[nom] = pd.concat(fenetres).groupby(level=0).mean()
    return pd.concat(profils, axis=1)


def graphique_etude_evenement(profils: pd.DataFrame) -> None:
    """Une case par variable, une courbe par groupe, de 72 h avant à 24 h après le pic."""
    groupes = profils.columns.get_level_values(0).unique()
    variables = profils.columns.get_level_values(1).unique()
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for ax, variable in zip(axes.flat, variables):
        for groupe, couleur in zip(groupes, COULEURS_GROUPES):
            ax.plot(profils.index, profils[(groupe, variable)], color=couleur, linewidth=1.5, label=groupe)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(variable)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Heures avant (−) et après (+) le pic")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Moyenne des variables autour du pic (tirets : pic ; pour les heures ordinaires, une heure au hasard)")
    fig.tight_layout()
    plt.show()


def part_episodes_par_decile(panel_variables: pd.DataFrame, episodes_cible: pd.DataFrame, noms: dict) -> pd.DataFrame:
    """Pour chaque variable et chaque décile : part des épisodes dont la première heure positive y tombe,
    et part des heures du décile (ce qu'on attendrait si la variable n'avait aucun lien avec les squeezes)."""
    debuts = episodes_cible[["symbol", "debut"]].rename(columns={"debut": "t"})
    lignes = []
    for nom, libelle in noms.items():
        donnees = panel_variables[["symbol", nom]].dropna()
        donnees["decile"] = donnees.groupby("symbol")[nom].transform(numero_decile)
        deciles_debuts = debuts.merge(donnees.reset_index(), on=["t", "symbol"])["decile"]
        parts = pd.DataFrame({
            "episodes_pct": 100 * deciles_debuts.value_counts(normalize=True),
            "heures_pct": 100 * donnees["decile"].value_counts(normalize=True),
        }).reindex(range(1, 11), fill_value=0)
        lignes.append(parts.assign(variable=libelle).rename_axis("decile").reset_index())
    return pd.concat(lignes, ignore_index=True)


def graphique_part_episodes(parts: pd.DataFrame) -> None:
    """Une case par variable : part des épisodes par décile (barres) et part des heures (points)."""
    variables = parts["variable"].unique()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    for ax, variable in zip(axes.flat, variables):
        donnees = parts[parts["variable"] == variable]
        ax.bar(donnees["decile"], donnees["episodes_pct"], color="#d62728", alpha=0.7, label="Épisodes")
        ax.plot(donnees["decile"], donnees["heures_pct"], "k_", markersize=12, markeredgewidth=2,
                label="Heures (aucun lien)")
        ax.set_title(variable, fontsize=10)
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Décile (1 = valeurs les plus basses)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Part (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Dans quel décile tombe la première heure de chaque épisode ? (10 cryptos ensemble)")
    fig.tight_layout()
    plt.show()


def croisement_funding_oi(panel_variables: pd.DataFrame) -> pd.DataFrame:
    """Nombre d'heures et taux de squeeze (%) selon le signe du funding et le sens de l'open interest sur 24 h."""
    donnees = panel_variables.dropna(subset=["funding_8h", "variation_oi_24h", "shortsqueeze"])
    funding = np.where(donnees["funding_8h"] < 0, "Funding négatif", "Funding positif ou nul")
    open_interest = np.where(donnees["variation_oi_24h"] > 0, "OI en hausse sur 24 h", "OI en baisse sur 24 h")
    squeeze = donnees["shortsqueeze"].astype("int64")
    tableau = squeeze.groupby([funding, open_interest]).agg(heures="count", taux_pct="mean")
    tableau["taux_pct"] *= 100
    return tableau.unstack()


def correlations_prix_positionnement(panel_variables: pd.DataFrame) -> pd.DataFrame:
    """Corrélation de Spearman entre chaque variable de positionnement et chaque variable de prix,
    en moyenne sur les cryptos."""
    colonnes = list(NOMS_ETAPE_4) + list(NOMS_PRIX)
    tableaux = [crypto[colonnes].corr(method="spearman").loc[list(NOMS_ETAPE_4), list(NOMS_PRIX)]
                for _, crypto in panel_variables.groupby("symbol")]
    moyenne = sum(tableaux) / len(tableaux)
    return moyenne.rename(index=NOMS_ETAPE_4, columns=NOMS_PRIX)