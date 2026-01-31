"""
Point d'entrée principal NETY
Lance le système IA et le Dashboard en parallèle avec communication via Bridge
"""

from nety.main import NETYSystem
from nety.core.model_selector import ModelSelector
from app.main import launch_app
import threading
import time
import sys


def start_nety_system(model_type=None):
    """Démarre le système IA NETY dans un thread dédié"""
    print("🚀 Démarrage du système IA NETY...")
    
    try:
        nety = NETYSystem(model_type=model_type)
        nety.start()
        
        try:
            nety.run()
        except KeyboardInterrupt:
            print("\n⚠️ Interruption détectée")
            nety.stop()
        except Exception as e:
            print(f"❌ Erreur dans le système NETY: {e}")
            nety.stop()
            
    except AttributeError as e:
        if "'Int8Params' object has no attribute 'SCB'" in str(e):
            print(f"\n❌ Erreur de quantization 8-bit détectée")
            print("💡 Cette erreur ne devrait plus se produire avec la dernière version.")
            print("🔧 Si vous voyez ce message, le code n'a pas été mis à jour correctement.")
            print("   La quantization 8-bit sur CPU a été désactivée dans response_generator.py")
            sys.exit(1)
        else:
            print(f"❌ Erreur critique: {e}")
            raise
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def start_dashboard():
    """Démarre le Dashboard Tkinter"""
    # Attendre que l'IA soit initialisée
    time.sleep(0.5)
    print("🖥️  Lancement du Dashboard...")
    launch_app()


if __name__ == "__main__":
    print("=" * 60)
    print("      NETY - Full Brain Simulation Architecture")
    print("=" * 60)
    print()
    
    # Étape 1: Demander le modèle à l'utilisateur
    selector = ModelSelector()
    model_type = selector.get_user_choice()
    print(f"✅ Modèle sélectionné: {model_type}")
    print()
    
    # Thread 1: Système IA NETY (Brain + ML Engine)
    nety_thread = threading.Thread(
        target=start_nety_system,
        args=(model_type,),
        name="NETY-AI-Thread",
        daemon=True
    )
    nety_thread.start()
    
    # Thread 2: Dashboard Tkinter (Interface utilisateur)
    # Lance dans le thread principal pour Tkinter
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du système...")
    finally:
        print("👋 NETY arrêté proprement")