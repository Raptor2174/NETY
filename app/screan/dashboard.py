



def setup_dashboard():
    """Configure et initialise le tableau de bord de l'application."""
    from app.config import config

    app_name = config.APP_NAME
    version = config.VERSION
    debug_mode = config.DEBUG

    print(f"Initialisation du tableau de bord: {app_name} (Version: {version})")
    if debug_mode==True:
        print("Mode débogage activé.")
    else:
        print("Mode débogage désactivé.")

    # Autres configurations et initialisations spécifiques au tableau de bord
    # ...