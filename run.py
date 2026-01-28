from nety.main import NETYSystem
from app.main import launch_app
import threading

def start_nety():
    nety = NETYSystem()
    nety.start()
    try:
        nety.run()
    except KeyboardInterrupt:
        nety.stop()

if __name__ == "__main__":
    # Thread IA
    nety_thread = threading.Thread(target=start_nety, daemon=True)
    nety_thread.start()

    # App Dashboard (thread principal)
    launch_app()
