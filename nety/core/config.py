# nety/core/config.py
import time

class Config(object):
    # General Settings
    APP_NAME = "Nety AI"
    VERSION = "0.1.4"
    DEBUG = True

    # Database Settings
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_USER = "nety_user"
    DB_PASSWORD = "secure_password"
    DB_NAME = "nety_db"

    # API Settings
    API_HOST = "localhost"


class BoucleSettings:    # Paramètres de la boucle principale
    LOOP_DELAY = 0.05
    # Intervalle de sommeil en secondes lorsque aucune donnée n'est reçue. 0.01=10ms, 0.1=100ms, 1=1s, 10=10s, 60=1min
    MAX_RETRIES = 5       # Nombre maximum de tentatives pour valider les données
    LOG_IDLE = False        # Activer/désactiver la journalisation lorsque le système est inactif



class ModuleTags:
    """Tags par type de module pour garder une nomenclature stable."""

    # Groupes principaux
    CORTEX_CEREBRAL = "CCM"
    CORTEX_LIMBIQUE = "LCM"
    TRONC_CEREBRAL = "BCM"
    CERVELET = "CBM"

    # Détails des sous-modules (codes courts pour futur routage/filtrage)
    SUBMODULES = {
        CORTEX_CEREBRAL: {
            "GEDRC": "CCM-0-GEDRC",
            "TPM": "CCM-1-TPM",
            "IPM": "CCM-2-IPM",
            "APM": "CCM-3-APM",
            "VPM": "CCM-4-VPM",
        },
        CORTEX_LIMBIQUE: {
            "GLDC": "LCM-0-GLDC",
            "ESM": "LCM-1-ESM",
            "UPMM": "LCM-2-UPMM",
            "SMM": "LCM-3-SMM",
        },
        TRONC_CEREBRAL: {
            "GDPM": "BCM-0-GDPM",
            "VFMM": "BCM-1-VFMM",
            "IMCM": "BCM-2-IMCM",
            "RDMM": "BCM-3-RDMM",
        },
        CERVELET: {
            "GMLM": "CBM-0-GMLM",
            "MCM": "CBM-1-MCM",
            "LM": "CBM-2-LM",
        },
    }

    @classmethod
    def all_group_tags(cls):
        return [
            cls.CORTEX_CEREBRAL,
            cls.CORTEX_LIMBIQUE,
            cls.TRONC_CEREBRAL,
            cls.CERVELET,
        ]

    @classmethod
    def all_module_tags(cls):
        # Renvoie la liste plate de tous les tags de sous-modules.
        return [tag for group in cls.SUBMODULES.values() for tag in group.values()]
