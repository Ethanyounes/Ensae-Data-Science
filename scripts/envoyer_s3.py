"""Dépose les données sur le S3 du SSPCloud (dossier public en lecture), avec la même arborescence que data/.

uv run --env-file .env python -m scripts.envoyer_s3

Les fichiers bruts déjà présents sur le S3 ne sont pas renvoyés ; les panels de data/interim/ le sont toujours,
car ils changent à chaque reconstruction.
"""

from src.config import DATA_DIR, INTERIM_DIR, RAW_DIR
from src.stockage import chemin_s3, connexion_s3

if __name__ == "__main__":
    s3 = connexion_s3()
    deja_envoyes = {chemin.split("/", 3)[3] for chemin in s3.find(chemin_s3("raw"))}

    a_envoyer = [f for f in sorted(RAW_DIR.rglob("*.parquet"))
                 if f.relative_to(DATA_DIR).as_posix() not in deja_envoyes]
    a_envoyer += sorted(INTERIM_DIR.glob("*.parquet"))
    print(f"{len(deja_envoyes)} fichiers bruts déjà sur le S3, {len(a_envoyer)} fichiers à envoyer")

    # Un seul appel avec des listes : s3fs envoie les fichiers en parallèle
    s3.put([str(f) for f in a_envoyer], [chemin_s3(f.relative_to(DATA_DIR).as_posix()) for f in a_envoyer])
    print("Envoi terminé :", ", ".join(f.name for f in a_envoyer if f.parent == INTERIM_DIR))
