"""Envoi des données sur le S3 du SSPCloud et lecture publique.

L'envoi demande les identifiants du fichier .env : uv run --env-file .env python ...
"""

from pathlib import Path

import pandas as pd
import s3fs

from src.config import DATA_DIR, S3_BUCKET, S3_ENDPOINT, S3_PREFIX, S3_PUBLIC_URL


def connexion_s3() -> s3fs.S3FileSystem:
    """Connexion au S3 (les clés sont lues dans les variables d'environnement)."""
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})


def chemin_s3(chemin_relatif: str) -> str:
    """Par exemple "raw/x.parquet" -> "ethany1/diffusion/ensae-squeeze/raw/x.parquet"."""
    return f"{S3_BUCKET}/{S3_PREFIX}/{chemin_relatif}"


def envoyer(chemin_local: Path, chemin_relatif: str) -> None:
    """Envoie un fichier ou un dossier entier sur le S3."""
    connexion_s3().put(str(chemin_local), chemin_s3(chemin_relatif), recursive=Path(chemin_local).is_dir())


def lire_public(chemin_relatif: str) -> pd.DataFrame:
    """Lit un parquet du S3 sans identifiants, par exemple lire_public("interim/panel.parquet")."""
    return pd.read_parquet(f"{S3_PUBLIC_URL}/{chemin_relatif}")


def lire_donnees(fichier: Path) -> pd.DataFrame:
    """Parquet de data/, lu en local s'il existe, sinon au même chemin sur le S3."""
    if fichier.exists():
        return pd.read_parquet(fichier)
    return lire_public(fichier.relative_to(DATA_DIR).as_posix())
