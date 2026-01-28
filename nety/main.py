from nety.core.config import Config
from nety.core.system_init import initialize_system

import time



class NETYSystem:
    """
    Classe principale orchestrant le cycle de vie de NETY :
    - démarrage
    - boucle principale
    - arrêt
    """

    def __init__(self):
        self.config = Config()
        self.running = False

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

    # ======================
    # BOUCLE PRINCIPALE
    # ======================
    def run(self):
        print(
            f"Lancement de la boucle principale de {self.config.APP_NAME}"
        )

        while self.running:
            input_data = self.receive_input()

            if input_data is None:
                time.sleep(0.1)
                continue

            if not self.validate_input(input_data):
                continue

            processed_data = self.process_data(input_data)

            if not self.validate_output(processed_data):
                continue

            self.send_output(processed_data)

    # ======================
    # ENTRÉES
    # ======================
    def receive_input(self):
        """
        Point d’entrée des données (mock pour l’instant).
        À terme : texte, audio, image, etc.
        """
        return "input_test"

    def validate_input(self, data) -> bool:
        print(f"Validation de l'entrée : {data}")
        return True

    # ======================
    # TRAITEMENT
    # ======================
    def process_data(self, data):
        print(f"Traitement des données : {data}")
        return f"processed_{data}"

    # ======================
    # SORTIES
    # ======================
    def validate_output(self, data) -> bool:
        print(f"Validation de la sortie : {data}")
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
# POINT D’ENTRÉE
# ======================
if __name__ == "__main__":
    nety = NETYSystem()
    nety.start()

    try:
        nety.run()
    except KeyboardInterrupt:
        nety.stop()
