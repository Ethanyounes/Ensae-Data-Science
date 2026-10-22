"""Construit le panel horaire à partir de data/raw/binance/ et l'écrit dans data/interim/panel.parquet.

Lancement depuis la racine du dépôt : uv run python -m scripts.construire_panel
À lancer une fois le téléchargement Binance terminé pour toutes les cryptos de SYMBOLS.
"""

from src.config import SYMBOLS
from src.panel import PANEL_FILE, build_panel, rapport_qualite

if __name__ == "__main__":
    panel = build_panel(SYMBOLS)
    print(f"Panel : {len(panel)} lignes, {panel['symbol'].nunique()} cryptos -> {PANEL_FILE}")
    print("\nValeurs manquantes (%) :")
    print(rapport_qualite(panel).T.to_string())
