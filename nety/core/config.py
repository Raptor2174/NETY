# Il y as 4 type de modules diferants: (ils reproduisent les differentes tache d'un cerveau biologique, mais c'est surtout pour organiser le systeme en differentes parties specialisées)
# - 1. Les modules du cortexte cerebral (CCM) (traitement des donnees principales)
#   - 1.0. Le centre generale de reception des donnees externes (General External Data Reception Center - GEDRC)
#   - 1.1. Le module de traitement de texte (Text Processing Module - TPM)
#   - 1.2. Le module de traitement d'images (Image Processing Module - IPM)
#   - 1.3. Le module de traitement audio (Audio Processing Module - APM)
#   - 1.4. Le module de traitement video (Video Processing Module - VPM)
# au stade technologique actuel il est imposible d'avoir plus de modules pour persevoir et comprendre des donnees sensorielles comme le gout, l'odorat, le toucher, la temperature, la douleur, etc.
# - 2. Les modules du cortex limbique (LCM) (Simulation des emotions et de la personnalite)
#   - 2.0. Le centre general des donnees limbiques (General Limbic Data Center - GLDC)
#   - 2.1. Le module de simulations des emotions (Emotion Simulation Module - ESM)
#   - 2.2. Le module de gestion de la personnalite Unique (Unique Personality Management Module - UPMM)
#   - 2.3. Le module d'auto motivation (Self-Motivation Module - SMM)
# - 3. Les modules du tronc cerebral (BCM) (Gestion des fonctions vitales et de la communication entre les autres modules)
#   - 3.0. La direction generale des directives et des priorites (General Directives and Priorities Management - GDPM)
#   - 3.1. Le module de gestion des fonctions vitales (Vital Functions Management Module - VFMM)
#   - 3.2. Le module de communication inter-modules (Inter-Module Communication Module - IMCM)
#   - 3.3. Le module de reflexion et de prise de decision (Reflection and Decision-Making Module - RDMM)
# - 4. Les modules du cervelet (CBM) (Coordination motrice et apprentissage moteur)
#   - 4.0. La direction generale motrice et d'apprentissage (General Motor and Learning Management - GMLM)
#   - 4.1. Le module de coordination motrice (Motor Coordination Module - MCM)
#   - 4.2. Le module d'apprentissage (Learning Module - LM)
# Chaque module est conçu pour imiter une fonction spécifique du cerveau humain, permettant ainsi au système d'intelligence artificielle de traiter et de réagir aux informations de manière plus ordonnée et efficace, sere aussi de ligne directife pour le clean architecture.

class Config(object):
    # General Settings
    APP_NAME = "Nety AI"
    VERSION = "1.0.0"
    DEBUG = True

    # Database Settings
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_USER = "nety_user"
    DB_PASSWORD = "secure_password"
    DB_NAME = "nety_db"

    # API Settings
    API_HOST = "localhost"


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
