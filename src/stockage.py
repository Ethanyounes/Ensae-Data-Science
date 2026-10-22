"""Échanges avec le stockage S3 du SSPCloud.

Écriture (envoi des données) : demande les identifiants du fichier .env.
    uv run --env-file .env python ...
Lecture : par l'URL publique du dossier diffusion/, sans identifiants.
"""

from pathlib import Path

import pandas as pd
import s3fs

from src.config import S3_BUCKET, S3_ENDPOINT, S3_PREFIX, S3_PUBLIC_URL


def connexion_s3() -> s3fs.S3FileSystem:
    """Connexion au S3 du SSPCloud.

    Les clés (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN) sont lues
    automatiquement dans les variables d'environnement : elles n'apparaissent jamais dans le code.
    """
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})


def chemin_s3(chemin_relatif: str) -> str:
    """Chemin complet dans le bucket : "raw/x.parquet" -> "ethany1/diffusion/ensae-squeeze/raw/x.parquet"."""
    return f"{S3_BUCKET}/{S3_PREFIX}/{chemin_relatif}"


def envoyer(chemin_local: Path, chemin_relatif: str) -> None:
    """Envoie un fichier ou un dossier entier (avec ses sous-dossiers) sur le S3."""
    fs = connexion_s3()
    fs.put(str(chemin_local), chemin_s3(chemin_relatif), recursive=Path(chemin_local).is_dir())


def lire_public(chemin_relatif: str) -> pd.DataFrame:
    """Lit un parquet du dossier public diffusion/, sans identifiants.

    C'est la fonction qu'utilise le notebook : n'importe qui peut la lancer.
    Exemple : lire_public("interim/panel.parquet")
    """
    return pd.read_parquet(f"{S3_PUBLIC_URL}/{chemin_relatif}")