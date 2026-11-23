"""Statistiques et graphiques descriptifs appelés par rapport.ipynb."""

from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cible import HORIZON, ajouter_cible
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
EVENEMENTS = {"2024-01-10": "ETF bitcoin au comptant", "2024-11-05": "Élection américaine",
              "2025-10-10": "Krach du 10 octobre"}


def charger_panel(avec_chauffe: bool = False) -> pd.DataFrame:
    """Panel horaire, lu en local s'il existe, sinon sur le S3. Sans la chauffe par défaut."""
    if PANEL_FILE.exists():
        panel = pd.read_parquet(PANEL_FILE)
    else:
        panel = lire_public("interim/panel.parquet")
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
