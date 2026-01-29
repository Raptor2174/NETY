import time
from nety.core.brain import Brain
from nety.core.config import Config
from nety.core.system_init import initialize_system
from nety.core.nety_bridge import bridge  # ← NOUVEAU


class BoucleSettings:
    """Paramètres de la boucle principale"""
    LOOP_DELAY = 0.05        # 50 ms entre chaque itération
    IDLE_LOG_INTERVAL = 40   # log toutes les 40 boucles sans données


class NETYSystem:
    """Système principal NETY"""

    def __init__(self):
        self.config = Config()
        self.running = False
        self.idle_counter = 0
        self.brain = None

    # ======================
    # DÉMARRAGE
    # ======================
    def start(self):
        print(
            f"Initialisation du système {self.config.APP_NAME} "
            f"version {self.config.VERSION}"
        )
        initialize_system()
        
        # Initialiser le Brain
        self.brain = Brain()
        
        # Notifier le Bridge
        bridge.set_brain_initialized(True)
        bridge.set_system_running(True)
        
        self.running = True

    # ======================
    # BOUCLE PRINCIPALE
    # ======================
    def run(self):
        bridge._add_log("🔄 Boucle principale NETY démarrée")

        while self.running:
            # PRIORITÉ 1: Vérifier les messages du Dashboard via Bridge
            dashboard_message = bridge.get_from_dashboard(timeout=0.01)
            
            if dashboard_message:
                # Traiter le message du Dashboard
                self.handle_dashboard_message(dashboard_message)
                continue
            
            # PRIORITÉ 2: Vérifier le fichier (pour compatibilité)
            input_data = self.check_for_input()

            if input_data is None:
                self.handle_idle()
                time.sleep(BoucleSettings.LOOP_DELAY)
                continue

            self.idle_counter = 0

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
    # GESTION MESSAGES DASHBOARD
    # ======================
    def handle_dashboard_message(self, message: dict):
        """
        Traite un message provenant du Dashboard
        
        Args:
            message: Dict avec 'type', 'content', 'timestamp'
        """
        msg_type = message.get("type", "unknown")
        content = message.get("content", "")
        
        # 🆕 AJOUTER CES LIGNES POUR DÉBUGGER ET NETTOYER
        # ================================================
        # DEBUG : Afficher AVANT nettoyage
        print(f"🐛 AVANT nettoyage: '{content}'")
        
        # Nettoyer tous les préfixes possibles
        prefixes_to_remove = ["CHAT: ", "PROMPT: ", "CHAT:", "PROMPT:"]
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
                break  # Arrêter après le premier match
        
        # DEBUG : Afficher APRÈS nettoyage
        print(f"🐛 APRÈS nettoyage: '{content}'")
        # ================================================
        
        bridge._add_log(f"📨 Message Dashboard reçu: {msg_type}")
        
        if msg_type == "prompt":
            # Traiter comme un prompt normal
            response = self.process_data(content)  # ✅ content est nettoyé
            # Renvoyer la réponse au Dashboard
            bridge.send_from_nety(response, msg_type="response")
            
        elif msg_type == "command":
            # Exécuter une commande système
            self.execute_command(content)
            
        elif msg_type == "chat":
            # Traiter comme une conversation
            response = self.process_data(content)  # ✅ content est nettoyé
            bridge.send_from_nety(response, msg_type="chat_response")

    def execute_command(self, command: str):
        """Exécute une commande système"""
        bridge._add_log(f"⚙️ Commande reçue: {command}")
        
        if command == "stop":
            self.stop()
        elif command == "status":
            stats = bridge.get_stats()
            bridge.send_from_nety(str(stats), msg_type="status")
        else:
            bridge.send_from_nety(f"Commande inconnue: {command}", msg_type="error")

    # ======================
    # GESTION IDLE
    # ======================
    def handle_idle(self):
        self.idle_counter += 1

        if self.idle_counter % BoucleSettings.IDLE_LOG_INTERVAL == 0:
            # Ne plus logger "En attente" pour ne pas polluer
            pass

    # ======================
    # ENTRÉES
    # ======================
    def check_for_input(self):
        """Vérifie s'il y a des données d'entrée depuis fichier"""
        import os
        
        input_file = "tmp_to_nety.txt"
        
        try:
            if os.path.exists(input_file):
                with open(input_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                if content:
                    with open(input_file, "w", encoding="utf-8") as f:
                        f.write("")
                    
                    bridge._add_log(f"📥 Données fichier reçues: {content[:50]}...")
                    return content
        
        except Exception as e:
            bridge._add_log(f"⚠️ Erreur lecture fichier: {e}")
        
        return None

    def validate_input(self, data) -> bool:
        return True

    # ======================
    # TRAITEMENT
    # ======================
    def process_data(self, data):
        """Traite les données avec le Brain"""
        if self.brain:
            try:
                result = self.brain.think(data)
                
                # Synchroniser l'état des modules
                modules_status = self.brain.get_modules_status()
                bridge.update_modules_status(modules_status)
                
                return result
            except Exception as e:
                error_msg = f"Erreur lors du traitement: {type(e).__name__}: {str(e)}"
                bridge._add_log(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                return f"❌ Erreur: {str(e)}"
        
        return f"processed_{data}"

    # ======================
    # SORTIES
    # ======================
    def validate_output(self, data) -> bool:
        return True

    def send_output(self, output_data):
        """Envoie la sortie (logs + potentiellement Dashboard)"""
        bridge._add_log(f"✅ Sortie NETY: {output_data[:100]}...")

    # ======================
    # ARRÊT
    # ======================
    def stop(self):
        bridge._add_log(
            f"🛑 Arrêt du système {self.config.APP_NAME} "
            f"version {self.config.VERSION}"
        )
        bridge.set_system_running(False)
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