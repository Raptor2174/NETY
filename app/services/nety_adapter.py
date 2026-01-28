from typing import Optional, List
import os
from datetime import datetime


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
        self.logs: List[str] = []
        self._add_log("💡 Système NETY Dashboard initialisé")

    def _add_log(self, message: str):
        """Ajoute un message au journal des logs
        
        Args:
            message: Le message à ajouter
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        # Limiter à 1000 entrées pour éviter une croissance infinie
        if len(self.logs) > 1000:
            self.logs.pop(0)

    def get_logs(self) -> List[str]:
        """Retourne tous les logs collectés
        
        Returns:
            List[str]: Liste des messages de log
        """
        return self.logs.copy()

    def clear_logs(self):
        """Efface tous les logs"""
        self.logs.clear()
        self._add_log("🗑️ Logs effacés")

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
            self._add_log(f"📤 Données envoyées vers NETY: {data[:50]}...")
            return True
        except (IOError, OSError) as e:
            error_msg = f"Erreur lors de l'envoi vers NETY: {e}"
            print(error_msg)
            self._add_log(f"❌ {error_msg}")
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
            msg = "⚠️ L'IA NETY n'est pas démarrée"
            self._add_log(f"⚠️ Tentative de traitement de prompt alors que l'IA est arrêtée")
            return msg
        
        self._add_log(f"🤖 Traitement du prompt ({len(prompt)} caractères)")
        
        # Simulation de traitement
        success = self.send_to_nety(prompt)
        if success:
            truncated = f"{prompt[:50]}..." if len(prompt) > 50 else prompt
            response = f"✓ Prompt reçu et traité ({len(prompt)} caractères): {truncated}"
            self._add_log(f"✅ Prompt traité avec succès")
            return response
        else:
            self._add_log(f"❌ Échec du traitement du prompt")
            return "❌ Erreur lors de l'envoi du prompt"

    def start_nety(self):
        """Démarre le système NETY"""
        self.nety_running = True
        self._add_log("🚀 IA NETY démarrée")
        self._add_log("✓ Initialisation des modules NETY...")
        self._add_log("✓ Système prêt à recevoir des commandes")
        return True

    def stop_nety(self):
        """Arrête le système NETY"""
        self.nety_running = False
        self._add_log("⏹️ IA NETY arrêtée")
        return True

    def is_running(self) -> bool:
        """Retourne True si NETY est en cours d'exécution"""
        return self.nety_running



 



