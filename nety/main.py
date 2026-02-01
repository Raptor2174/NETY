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

    def __init__(self, model_type=None):
        self.config = Config()
        self.running = False
        self.idle_counter = 0
        self.brain = None
        self.model_type = model_type

    # ======================
    # DÉMARRAGE
    # ======================
    def start(self):
        print(
            f"Initialisation du système {self.config.APP_NAME} "
            f"version {self.config.VERSION}"
        )
        initialize_system()
        
        # Initialiser le Brain avec le modèle choisi
        self.brain = Brain(model_type=self.model_type)
        
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
        
        # Nettoyer tous les préfixes possibles
        prefixes_to_remove = ["CHAT: ", "PROMPT: ", "CHAT:", "PROMPT:"]
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
                break  # Arrêter après le premier match
        
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
def main():
    """Fonction principale - Point d'entrée pour console_scripts"""
    # Demander à l'utilisateur quel modèle
    chosen_model = select_model_interactive()
    
    # Créer le système avec le modèle choisi
    system = NETYSystem(model_type=chosen_model)
    system.start()
    
    try:
        system.run()
    except KeyboardInterrupt:
        system.stop()


if __name__ == "__main__":
    main()


def select_model_interactive():
    """Menu interactif de sélection de modèle"""

    print("\n" + "="*70)
    print("🤖 SÉLECTION DU MODÈLE D'IA POUR NETY")
    print("="*70 + "\n")

    models = [
        {
            "name": "Mistral-7B (Local GPU - Puissant mais gourmand)",
            "type": "LOCAL",
            "key": "mistral",
            "cost": "Gratuit (utilise ton matériel)",
            "internet": "Non requis",
            "speed": "Moyen (dépend GPU)",
            "quality": "Excellent",
            "ram": "8 GB + 4 GB VRAM (GPU)",
            "note": "⚠️ Nécessite un GPU compatible CUDA"
        },
        {
            "name": "BLOOMZ-560M (Local CPU - Léger et rapide)",
            "type": "LOCAL",
            "key": "bloomz",
            "cost": "Gratuit (utilise ton matériel)",
            "internet": "Non requis",
            "speed": "Rapide (CPU uniquement)",
            "quality": "Correct",
            "ram": "2 GB",
            "note": ""
        },
        {
            "name": "Groq Cloud - Llama 3.3 (Cloud ultra rapide)",
            "type": "CLOUD",
            "key": "groq",
            "cost": "Gratuit (14.4k req/jour)",
            "internet": "Requis",
            "speed": "⚡⚡⚡ Ultra rapide (500 tok/sec)",
            "quality": "Excellent",
            "ram": "0 GB (cloud)",
            "note": ""
        },
        # ✨ NOUVEAU
        {
            "name": "RNN Local - TextualCortex (Expérimental)",
            "type": "LOCAL",
            "key": "rnn",
            "cost": "Gratuit (utilise ton matériel)",
            "internet": "Non requis",
            "speed": "Rapide (CPU/GPU)",
            "quality": "🧪 En apprentissage",
            "ram": "500 MB",
            "note": "🔬 Mode test - Pour observer les progrès du RNN"
        },
    ]

    # Afficher les options
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Type: {model['type']}")
        print(f"   💰 {model['cost']}")
        print(f"   📶 {model['internet']}")
        print(f"   ⚡ {model['speed']}")
        print(f"   🧠 {model['quality']}")
        print(f"   💾 RAM: {model['ram']}")
        if model['note']:
            print(f"   {model['note']}")
        print()

    print("💡 Recommandations:")
    print("   • Pas de GPU → BLOOMZ (option 2)")
    print("   • GPU disponible → Mistral (option 1)")
    print("   • PC faible + internet → Groq (option 3)")
    print("   • Tester le RNN local → RNN (option 4) 🧪")
    print()

    # Sélection
    while True:
        try:
            choice = input("👉 Choisis ton modèle (1, 2, 3 ou 4): ").strip()
            choice_int = int(choice)
            if 1 <= choice_int <= len(models):
                selected = models[choice_int - 1]
                print(f"\n✅ Modèle sélectionné: {selected['name']}\n")
                return selected['key']
            else:
                print(f"❌ Choix invalide. Entre un nombre entre 1 et {len(models)}.")
        except ValueError:
            print("❌ Entrée invalide. Entre un nombre.")
        except KeyboardInterrupt:
            print("\n\n👋 Annulation...")
            exit(0)