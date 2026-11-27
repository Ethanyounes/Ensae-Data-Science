"""Télécharge l'indice Fear & Greed.

Lancement depuis la racine du dépôt : uv run python -m scripts.telecharger_fng
Si le fichier existe déjà, il est relu et non retéléchargé.
"""

import pandas as pd

from src.collecte_fng import FNG_FILE, download_fear_greed

if __name__ == "__main__":
    if FNG_FILE.exists():
        fng = pd.read_parquet(FNG_FILE)
    else:
        fng = download_fear_greed()
        FNG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fng.to_parquet(FNG_FILE, index=False)
    print(f"{'Fear & Greed':24} {len(fng):6} lignes, "
          f"{fng['date'].iloc[0]} -> {fng['date'].iloc[-1]}")
