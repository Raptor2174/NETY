"""
Point d'entrée principal NETY
Lance le système IA et le Dashboard en parallèle avec communication via Bridge
"""

from nety.main import NETYSystem
from app.main import launch_app
import threading
import time
import sys


def start_nety_system():
    """Démarre le système IA NETY dans un thread dédié"""
    print("🚀 Démarrage du système IA NETY...")
    
    max_retries = 2
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            nety = NETYSystem()
            nety.start()
            
            try:
                nety.run()
            except KeyboardInterrupt:
                print("\n⚠️ Interruption détectée")
                nety.stop()
                break
            except Exception as e:
                print(f"❌ Erreur dans le système NETY: {e}")
                nety.stop()
                break
                
        except AttributeError as e:
            if "'Int8Params' object has no attribute 'SCB'" in str(e):
                retry_count += 1
                print(f"\n⚠️ Erreur de quantization détectée (tentative {retry_count}/{max_retries})")
                print("💡 Suggestion: Le modèle quantizé 8-bit rencontre un problème.")
                print("   Veuillez mettre à jour response_generator.py pour:")
                print("   - Désactiver load_in_8bit")
                print("   - Ou mettre à jour bitsandbytes: pip install --upgrade bitsandbytes")
                
                if retry_count >= max_retries:
                    print("\n❌ Échec du chargement du modèle après plusieurs tentatives")
                    print("🔧 Actions recommandées:")
                    print("   1. pip install --upgrade bitsandbytes transformers torch")
                    print("   2. Modifier response_generator.py: load_in_8bit=False")
                    print("   3. Ou utiliser un autre modèle")
                    sys.exit(1)
                
                time.sleep(2)
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