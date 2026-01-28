from typing import Optional
import os


def get_modules_status():
    """Retourne le statut des modules NETY"""
    return [
        {"type": "CCM", "name": "TPM", "status": "running"},
        {"type": "LCM", "name": "ESM", "status": "inactive"},
        {"type": "BCM", "name": "IMCM", "status": "running"},
        {"type": "CBM", "name": "LM", "status": "idle"},
    ]


class NetyAdapter:
    """Adaptateur pour communiquer avec le système NETY AI"""
    
    def __init__(self):
        self.admin_room = None
        self.nety_running = False
        self.nety_system = None

    def set_admin_room(self, admin_room):
        """Définit la référence à AdminRoom"""
        self.admin_room = admin_room

    def send_to_nety(self, data: str):
        """Envoie des données vers NETY"""
        try:
            with open("tmp_to_nety.txt", "w") as f:
                f.write(data)
        except Exception as e:
            print(f"Erreur lors de l'envoi vers NETY: {e}")

    def check_for_admin_message(self) -> Optional[str]:
        """Vérifie s'il y a des messages depuis le dashboard"""
        try:
            if os.path.exists("tmp_from_dashboard.txt"):
                with open("tmp_from_dashboard.txt", "r") as f:
                    content = f.read().strip()
                # Nettoyer le fichier après lecture
                with open("tmp_from_dashboard.txt", "w") as f:
                    f.write("")
                return content
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Erreur lors de la lecture du message: {e}")
        return None

    def process_prompt(self, prompt: str) -> str:
        """Traite un prompt et retourne une réponse"""
        if not self.nety_running:
            return "⚠️ L'IA NETY n'est pas démarrée"
        
        # Pour l'instant, simulation de traitement
        self.send_to_nety(prompt)
        return f"✓ Prompt reçu et traité: {prompt[:50]}..."

    def start_nety(self):
        """Démarre le système NETY"""
        self.nety_running = True
        return True

    def stop_nety(self):
        """Arrête le système NETY"""
        self.nety_running = False
        return True

    def is_running(self) -> bool:
        """Retourne True si NETY est en cours d'exécution"""
        return self.nety_running



 



