"""Collecte des données brutes : archives Binance (data.binance.vision)."""
import requests 
from src.config import BINANCE_BASE_URL

# Description de chaque type d'archive Binance :
#   marche    : "futures/um" (perpétuels USDT) ou "spot"
#   frequence : "daily" (un fichier par jour) ou "monthly" (un fichier par mois)
#   dossier   : nom du dossier chez Binance
#   intervalle: True si le type a un sous-dossier d'intervalle (klines)
BINANCE_KINDS = {
    "metrics": {"marche": "futures/um", "frequence": "daily",
                "dossier": "metrics", "intervalle": False},
    "fundingRate": {"marche": "futures/um", "frequence": "monthly",
                    "dossier": "fundingRate", "intervalle": False},
    "klines": {"marche": "futures/um", "frequence": "monthly",
               "dossier": "klines", "intervalle": True},
    "premiumIndexKlines": {"marche": "futures/um", "frequence": "monthly",
                           "dossier": "premiumIndexKlines", "intervalle": True},
    "spot_klines": {"marche": "spot", "frequence": "monthly",
                    "dossier": "klines", "intervalle": True},
}


def binance_url(symbol: str, kind: str, period: str, interval: str = "1h") -> str:
    """URL d'une archive Binance.

    period : "AAAA-MM-JJ" pour un type quotidien, "AAAA-MM" pour un type mensuel.
    Exemple : binance_url("BTCUSDT", "klines", "2023-01")
    -> .../futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip
    """
    if kind not in BINANCE_KINDS:
        raise ValueError(f"Type inconnu : {kind!r}. Types possibles : {list(BINANCE_KINDS)}")
    info = BINANCE_KINDS[kind]

    # La longueur de period doit correspondre à la fréquence du type
    longueur_attendue = 10 if info["frequence"] == "daily" else 7
    if len(period) != longueur_attendue:
        raise ValueError(f"{kind} attend une période de {longueur_attendue} caractères, reçu {period!r}")

    dossier = f"{BINANCE_BASE_URL}/{info['marche']}/{info['frequence']}/{info['dossier']}/{symbol}"
    if info["intervalle"]:
        # Pour les klines, l'intervalle remplace le type dans le nom du fichier
        return f"{dossier}/{interval}/{symbol}-{interval}-{period}.zip"
    return f"{dossier}/{symbol}-{kind}-{period}.zip"

def fetch_binance_file(symbol: str, kind: str, period: str, interval: str = "1h") -> bytes:
    """Télécharge le fichier d'archive Binance correspondant.

    Retourne le contenu du fichier (bytes) ou lève une exception si le téléchargement échoue.
    """
  
    url = binance_url(symbol, kind, period, interval)
    response = requests.get(url)
    response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
    return response.content
