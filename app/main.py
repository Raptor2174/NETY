import sys
from pathlib import Path

# Ajouter le répertoire racine au sys.path pour permettre l'import de nety
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ui.window import launch_app

if __name__ == "__main__":
    launch_app()

