import time

from nety.core.brain import Brain
from nety.core.config import Config
from nety.core.system_init import initialize_system


class BoucleSettings:
    """
    Paramètres de la boucle principale
    """
    LOOP_DELAY = 0.05        # 50 ms entre chaque itération
    IDLE_LOG_INTERVAL = 40   # log toutes les 40 boucles sans données


class NETYSystem:
    """
    Système principal NETY
    """

    def __init__(self):
        self.config = Config()
        self.running = False
        self.idle_counter = 0
        self.brain = None  # Initialisation différée

    # ======================
    # DÉMARRAGE
    # ======================
    def start(self):
        print(
            f"Initialisation du système {self.config.APP_NAME} "
            f"version {self.config.VERSION}"
        )
        initialize_system()
        self.running = True

        self.brain = Brain()

    # ======================
    # BOUCLE PRINCIPALE
    # ======================
    def run(self):
        print("Boucle principale NETY démarrée")

        while self.running:
            input_data = self.check_for_input()

            if input_data is None:
                self.handle_idle()
                time.sleep(BoucleSettings.LOOP_DELAY)
                continue

            self.idle_counter = 0  # reset dès qu'on reçoit quelque chose

            if not self.validate_input(input_data):
                time.sleep(BoucleSettings.LOOP_DELAY)
                continue

            processed_data = self.process_data(input_data)

            if not self.validate_output(processed_data):
                time.sleep(BoucleSettings.LOOP_DELAY)
                continue

            self.send_output(processed_data)

            time.sleep(BoucleSettings.LOOP_DELAY)

    # ======================
    # GESTION IDLE
    # ======================
    def handle_idle(self):
        self.idle_counter += 1

        if self.idle_counter % BoucleSettings.IDLE_LOG_INTERVAL == 0:
            print("En attente de données…")

    # ======================
    # ENTRÉES
    # ======================
    def check_for_input(self):
        """
        Vérifie s'il y a des données d'entrée
        
        V1 : Lit depuis un fichier texte (pour test)
        V2 : Lira depuis l'interface dashboard
        """
        import os
        
        # Vérifier si un fichier d'entrée existe
        input_file = "tmp_to_nety.txt"
        
        try:
            if os.path.exists(input_file):
                with open(input_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                # Si le fichier contient du texte
                if content:
                    # Vider le fichier après lecture
                    with open(input_file, "w", encoding="utf-8") as f:
                        f.write("")
                    
                    print(f"📥 Données reçues : {content}")
                    return content
        
        except Exception as e:
            print(f"⚠️ Erreur lecture entrée : {e}")
        
        return None

    def validate_input(self, data) -> bool:
        print(f"Validation entrée : {data}")
        return True

    # ======================
    # TRAITEMENT
    # ======================
    def process_data(self, data):
        if self.brain:
            return self.brain.think(data)
        print(f"Traitement : {data}")
        return f"processed_{data}"

    # ======================
    # SORTIES
    # ======================
    def validate_output(self, data) -> bool:
        return True

    def send_output(self, output_data):
        print(f"Sortie NETY : {output_data}")

    # ======================
    # ARRÊT
    # ======================
    def stop(self):
        print(
            f"Arrêt du système {self.config.APP_NAME} "
            f"version {self.config.VERSION}"
        )
        self.running = False


# ======================
# POINT D'ENTRÉE
# ======================
if __name__ == "__main__":
    nety = NETYSystem()
    nety.start()

    try:
        nety.run()
    except KeyboardInterrupt:
        nety.stop()

