"""
NETY Bridge - Pont de communication entre Dashboard et IA
Singleton pour assurer une seule instance partagée entre les threads
"""

import queue
import threading
from typing import List, Dict, Optional, Any
from datetime import datetime


class NetyBridge:
    """
    Pont de communication unique entre le Dashboard et l'IA NETY
    Pattern Singleton pour garantir une seule instance
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Configuration des logs
    MAX_LOGS = 5000  # Capacité maximale des logs (augmentée de 1000 à 5000)
    LOGS_ROTATION_THRESHOLD = 0.9  # Rotation à 90% de capacité
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Queues de communication
        self.to_nety_queue = queue.Queue()        # Dashboard → IA
        self.from_nety_queue = queue.Queue()      # IA → Dashboard
        
        # Logs partagés (thread-safe) avec gestion améliorée
        self.logs: List[str] = []
        self.logs_lock = threading.Lock()
        self.logs_rotation_count = 0  # Suivi des rotations
        
        # État du système
        self.system_running = False
        self.brain_initialized = False
        
        # État des modules (synchronisé avec Brain)
        self.modules_status: Dict[str, str] = {}
        self.modules_lock = threading.Lock()
        
        # Statistiques
        self.messages_sent = 0
        self.messages_received = 0
        
        self._initialized = True
        self._add_log("🌉 NETY Bridge initialisé")
    
    # ==========================================
    # 📨 COMMUNICATION : DASHBOARD → IA
    # ==========================================
    def send_to_nety(self, message: str, msg_type: str = "prompt") -> bool:
        """
        Envoie un message depuis le Dashboard vers l'IA
        
        Args:
            message: Le contenu du message
            msg_type: Type de message ('prompt', 'command', 'chat')
        
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            payload = {
                "type": msg_type,
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            self.to_nety_queue.put(payload)
            self.messages_sent += 1
            self._add_log(f"📤 [Dashboard→IA] {msg_type.upper()}: {message[:50]}...")
            return True
        except Exception as e:
            self._add_log(f"❌ Erreur envoi Dashboard→IA: {e}")
            return False
    
    def get_from_dashboard(self, timeout: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Récupère un message depuis le Dashboard (appelé par l'IA)
        
        Args:
            timeout: Temps d'attente maximum
        
        Returns:
            Dict ou None si aucun message
        """
        try:
            message = self.to_nety_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            return None
    
    # ==========================================
    # 📩 COMMUNICATION : IA → DASHBOARD
    # ==========================================
    def send_from_nety(self, message: str, msg_type: str = "response") -> bool:
        """
        Envoie une réponse depuis l'IA vers le Dashboard
        
        Args:
            message: Le contenu de la réponse
            msg_type: Type ('response', 'status', 'error')
        
        Returns:
            bool: True si envoyé avec succès
        """
        try:
            payload = {
                "type": msg_type,
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            self.from_nety_queue.put(payload)
            self.messages_received += 1
            self._add_log(f"📥 [IA→Dashboard] {msg_type.upper()}: {message[:50]}...")
            return True
        except Exception as e:
            self._add_log(f"❌ Erreur envoi IA→Dashboard: {e}")
            return False
    
    def get_from_nety(self, timeout: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Récupère une réponse de l'IA (appelé par le Dashboard)
        
        Args:
            timeout: Temps d'attente maximum
        
        Returns:
            Dict ou None si aucune réponse
        """
        try:
            message = self.from_nety_queue.get(timeout=timeout)
            return message
        except queue.Empty:
            return None
    
    # ==========================================
    # 📝 GESTION DES LOGS (Thread-Safe)
    # ==========================================
    def _add_log(self, message: str):
        """
        Ajoute un log de manière thread-safe avec gestion intelligente de rotation
        
        Stratégie: Quand on atteint 90% de capacité, garder les 75% les plus récents
        (supprimer les 25% les plus anciens) pour éviter les tronquages soudains
        """
        with self.logs_lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            self.logs.append(log_entry)
            
            # Gestion intelligente de la capacité
            current_capacity_ratio = len(self.logs) / self.MAX_LOGS
            if current_capacity_ratio >= self.LOGS_ROTATION_THRESHOLD:
                # Garder les 75% les plus récents (supprimer les 25% les plus anciens)
                remove_count = len(self.logs) // 4
                self.logs = self.logs[remove_count:]
                self.logs_rotation_count += 1
                rotation_msg = f"[{timestamp}] 🔄 Rotation logs #{self.logs_rotation_count} (gardé {len(self.logs)}/{self.MAX_LOGS})"
                self.logs.append(rotation_msg)
            
            # Aussi afficher dans la console
            print(log_entry)
    
    def get_logs(self) -> List[str]:
        """
        Récupère tous les logs (thread-safe)
        IMPORTANT: Retourne la totalité des logs actuellement en mémoire
        """
        with self.logs_lock:
            return self.logs.copy()
    
    def clear_logs(self):
        """Efface les logs"""
        with self.logs_lock:
            self.logs.clear()
            self.logs_rotation_count = 0
            self._add_log("🗑️ Logs effacés")
    
    def get_logs_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques détaillées des logs (diagnostic)
        Utile pour monitorer la santé du système de logs
        """
        with self.logs_lock:
            return {
                "total_logs": len(self.logs),
                "max_capacity": self.MAX_LOGS,
                "capacity_used_percent": (len(self.logs) / self.MAX_LOGS) * 100,
                "rotation_count": self.logs_rotation_count,
                "first_log": self.logs[0] if self.logs else None,
                "last_log": self.logs[-1] if self.logs else None
            }
    
    # ==========================================
    # 🔧 GESTION DES MODULES (Thread-Safe)
    # ==========================================
    def update_modules_status(self, modules: Dict[str, str]):
        """
        Met à jour l'état des modules (appelé par Brain)
        
        Args:
            modules: Dictionnaire {code_module: statut}
        """
        with self.modules_lock:
            self.modules_status = modules.copy()
            self._add_log(f"🔄 État modules mis à jour ({len(modules)} modules)")
    
    def get_modules_status(self) -> Dict[str, str]:
        """Récupère l'état des modules (thread-safe)"""
        with self.modules_lock:
            return self.modules_status.copy()
    
    # ==========================================
    # 📊 ÉTAT DU SYSTÈME
    # ==========================================
    def set_system_running(self, running: bool):
        """Définit l'état de marche du système"""
        self.system_running = running
        status = "EN MARCHE" if running else "ARRÊTÉ"
        self._add_log(f"⚙️ Système NETY: {status}")
    
    def is_system_running(self) -> bool:
        """Retourne True si le système est en marche"""
        return self.system_running
    
    def set_brain_initialized(self, initialized: bool):
        """Définit si le Brain est initialisé"""
        self.brain_initialized = initialized
        if initialized:
            self._add_log("🧠 Brain NETY initialisé")
    
    def is_brain_ready(self) -> bool:
        """Retourne True si le Brain est prêt"""
        return self.brain_initialized
    
    # ==========================================
    # 📈 STATISTIQUES
    # ==========================================
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système avec détails des logs"""
        logs_stats = self.get_logs_stats()
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "logs_count": logs_stats["total_logs"],
            "logs_capacity_percent": logs_stats["capacity_used_percent"],
            "logs_rotations": logs_stats["rotation_count"],
            "system_running": self.system_running,
            "brain_ready": self.brain_initialized,
            "modules_count": len(self.modules_status)
        }


# Instance globale singleton
bridge = NetyBridge()
