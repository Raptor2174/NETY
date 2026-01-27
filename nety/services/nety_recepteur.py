from curses import raw
from app.services.nety_adapter import NetyAdapter

def receive_from_dashboard(self, raw: str) -> str:
    return raw.upper()  # juste un exemple simple

class NetyRecepteur:
    def __init__(self):
        self.adapter = NetyAdapter()

    

    
    