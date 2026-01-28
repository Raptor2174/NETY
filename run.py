"""
Point d'entrée principal NETY
Lance le système IA et le Dashboard en parallèle avec communication via Bridge
"""

from nety.main import NETYSystem
from app.main import launch_app
import threading
import time


def start_nety_system():
    """Démarre le système IA NETY dans un thread dédié"""
    print("🚀 Démarrage du système IA NETY...")
    nety = NETYSystem()
    nety.start()
    
    try:
        nety.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interruption détectée")
        nety.stop()
    except Exception as e:
        print(f"❌ Erreur dans le système NETY: {e}")
        nety.stop()


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
    
    # Thread 1: Système IA NETY (Brain + ML Engine)
    nety_thread = threading.Thread(
        target=start_nety_system,
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