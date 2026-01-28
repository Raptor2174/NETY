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
    """Adaptateur pour communiquer avec le système NETY AI
    
    Note: Cette implémentation est une simulation. Les méthodes start_nety()
    et stop_nety() ne démarrent pas réellement le système NETY, elles gèrent
    seulement un état booléen pour l'interface utilisateur.
    """
    
    def __init__(self):
        self.nety_running = False

    def send_to_nety(self, data: str) -> bool:
        """Envoie des données vers NETY
        
        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        if not data:
            return False
            
        try:
            with open("tmp_to_nety.txt", "w", encoding='utf-8') as f:
                f.write(data)
            return True
        except (IOError, OSError) as e:
            print(f"Erreur lors de l'envoi vers NETY: {e}")
            return False

    def check_for_admin_message(self) -> Optional[str]:
        """Vérifie s'il y a des messages depuis le dashboard
        
        Returns:
            Optional[str]: Le message lu, ou None si aucun message
        """
        try:
            if os.path.exists("tmp_from_dashboard.txt"):
                with open("tmp_from_dashboard.txt", "r", encoding='utf-8') as f:
                    content = f.read().strip()
                # Nettoyer le fichier après lecture seulement si non vide
                if content:
                    with open("tmp_from_dashboard.txt", "w", encoding='utf-8') as f:
                        f.write("")
                    return content
        except (FileNotFoundError, IOError, OSError) as e:
            print(f"Erreur lors de la lecture du message: {e}")
        return None

    def process_prompt(self, prompt: str) -> str:
        """Traite un prompt et retourne une réponse
        
        Args:
            prompt: Le prompt à traiter
            
        Returns:
            str: La réponse du système
        """
        if not self.nety_running:
            return "⚠️ L'IA NETY n'est pas démarrée"
        
        # Simulation de traitement
        success = self.send_to_nety(prompt)
        if success:
            truncated = f"{prompt[:50]}..." if len(prompt) > 50 else prompt
            return f"✓ Prompt reçu et traité ({len(prompt)} caractères): {truncated}"
        else:
            return "❌ Erreur lors de l'envoi du prompt"

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



 



