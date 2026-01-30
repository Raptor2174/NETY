"""
NETY Adapter - Interface entre le Dashboard Tkinter et l'IA NETY
Utilise le NetyBridge pour la communication inter-threads
"""

from typing import Optional, List, Dict
from nety.core.nety_bridge import bridge


class NetyAdapter:
    """
    Adaptateur pour le Dashboard
    Communique avec l'IA NETY via le NetyBridge (thread-safe)
    """
    
    def __init__(self):
        """Initialise l'adaptateur en se connectant au Bridge"""
        self.admin_messages = []
        # Pas besoin de variables locales, tout passe par le Bridge
        bridge._add_log("📱 NetyAdapter (Dashboard) connecté au Bridge")
    
    # ==========================================
    # 📨 ENVOI VERS L'IA
    # ==========================================
    def send_to_nety(self, data: str, msg_type: str = "prompt") -> bool:
        """
        Envoie des données vers NETY via le Bridge
        
        Args:
            data: Contenu à envoyer
            msg_type: Type de message ('prompt', 'command', 'chat')
        
        Returns:
            bool: True si envoyé avec succès
        """
        if not data:
            return False
        
        return bridge.send_to_nety(data, msg_type=msg_type)
    
    def process_prompt(self, prompt: str) -> str:
        """
        Traite un prompt et attend une réponse
        
        Args:
            prompt: Le prompt à traiter
            
        Returns:
            str: La réponse du système ou message d'erreur
        """
        if not self.is_running():
            return "⚠️ L'IA NETY n'est pas démarrée"
        
        # Envoyer le prompt
        success = self.send_to_nety(prompt, msg_type="prompt")
        
        if not success:
            return "❌ Erreur lors de l'envoi du prompt"
        
        # Attendre la réponse (avec timeout)
        import time
        timeout = 5.0  # 5 secondes max
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            response = bridge.get_from_nety(timeout=0.1)
            if response and response.get("type") == "response":
                return response.get("content", "Réponse vide")
            time.sleep(0.05)
        
        return "⏱️ Timeout: Pas de réponse de l'IA"
    
    # ==========================================
    # 📩 RÉCEPTION DEPUIS L'IA
    # ==========================================
    def check_for_nety_response(self) -> Optional[Dict]:
        """
        Vérifie s'il y a des réponses de l'IA
        
        Returns:
            Dict ou None
        """
        return bridge.get_from_nety(timeout=0.01)
    
    # ==========================================
    # 📝 LOGS
    # ==========================================
    def get_logs(self) -> List[str]:
        """Récupère tous les logs depuis le Bridge"""
        return bridge.get_logs()
    
    def clear_logs(self):
        """Efface les logs"""
        bridge.clear_logs()
    
    # ==========================================
    # 🔧 ÉTAT DU SYSTÈME
    # ==========================================
    def is_running(self) -> bool:
        """Retourne True si l'IA est en marche"""
        return bridge.is_system_running()
    
    def is_brain_ready(self) -> bool:
        """Retourne True si le Brain est initialisé"""
        return bridge.is_brain_ready()
    
    def start_nety(self):
        """
        Note: Le démarrage réel se fait dans run.py
        Cette méthode sert juste pour l'UI
        """
        bridge._add_log("ℹ️ Demande de démarrage depuis Dashboard (l'IA démarre via run.py)")
        return True
    
    def stop_nety(self):
        """Arrête le système NETY"""
        bridge.send_to_nety("stop", msg_type="command")
        return True
    
    # ==========================================
    # 🔧 MODULES
    # ==========================================
    def get_modules_status(self) -> List[Dict[str, str]]:
        """
        Récupère l'état des modules depuis le Bridge
        
        Returns:
            Liste de dicts avec 'type', 'name', 'status'
        """
        modules_dict = bridge.get_modules_status()
        
        # Convertir en format attendu par le Dashboard
        modules_list = []
        for code, status in modules_dict.items():
            # Parser le code (ex: "CCM-1-TPM")
            parts = code.split("-")
            if len(parts) >= 3:
                module_type = parts[0]
                module_name = parts[2]
                modules_list.append({
                    "type": module_type,
                    "name": module_name,
                    "status": status
                })
        
        # Si aucun module, retourner un exemple
        if not modules_list:
            return [
                {"type": "CCM", "name": "Brain", "status": "active" if self.is_brain_ready() else "inactive"},
                {"type": "LCM", "name": "Memory", "status": "inactive"},
                {"type": "BCM", "name": "ML_Engine", "status": "active" if self.is_running() else "inactive"},
            ]
        
        return modules_list
    
    # ==========================================
    # 📊 STATISTIQUES
    # ==========================================
    def get_stats(self) -> Dict:
        """Récupère les statistiques du système"""
        return bridge.get_stats()
    
    def check_for_admin_message(self) -> Optional[str]:
        """
        Vérifie s'il y a des messages provenant de l'administrateur
        via le tableau de bord.
        """
        if self.admin_messages:
            return self.admin_messages.pop(0)
        return None

    def get_all_admin_messages(self):
        """
        Récupère tous les messages admin en attente.
        À adapter selon ta logique métier.
        """
        return []  # À remplacer par la vraie récupération des messages
    
    def get_admin_message(self) -> Optional[str]:
        """
        Récupère un message admin s'il y en a un.
        """
        if self.admin_messages:
            return self.admin_messages.pop(0)
        return None


# Fonction utilitaire pour compatibilité
def get_modules_status():
    """Fonction standalone pour récupérer les modules"""
    adapter = NetyAdapter()
    return adapter.get_modules_status()