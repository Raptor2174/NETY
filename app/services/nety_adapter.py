from typing import Optional
from ui.admin_room import AdminRoomView
from nety.services.nety_recepteur import NetyRecepteur

def get_modules_status():
    return [
        {"type": "CCM", "name": "TPM", "status": "running"},
        {"type": "LCM", "name": "ESM", "status": "inactive"},
        {"type": "BCM", "name": "IMCM", "status": "running"},
        {"type": "CBM", "name": "LM", "status": "idle"},
     ]

def send_to_nety(self, data: str):
    with open("tmp_to_nety.txt", "w") as f:
        f.write(data)

def check_for_admin_message(self) -> Optional[str]:
    try:
        with open("tmp_from_dashboard.txt", "r") as f:
            content = f.read().strip()
            open("tmp_from_dashboard.txt", "w").close()
            return content
    except FileNotFoundError:
        return None

class NetyAdapter:
    def __init__(self):
        self.admin_room = None

    def set_admin_room(self, admin_room):
        """Définit la référence à AdminRoom"""
        self.admin_room = admin_room



 



