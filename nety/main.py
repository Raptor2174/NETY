
from nety.core.config import Config
from nety.core.system_init import initialize_system

def load_configuration():
    """Charge la configuration de l'application depuis le fichier config.py."""
    config = Config()
    return config

def lunch_application():
    """Lance l'application NETY en initialisant le système."""
    config = load_configuration()
    print(f"Lancement de l'application {config.APP_NAME} version {config.VERSION}")
    initialize_system()