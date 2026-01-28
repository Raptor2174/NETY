from app.services.nety_adapter import NetyAdapter

class NetyRecepteur:
    def __init__(self):
        self.adapter = NetyAdapter()

    def check_for_input(self):
        """
        Vérifie s'il y a des données entrantes du tableau de bord
        ou d'autres sources.
        """
        data = self.receive_from_dashboard("input_data")
        return data

    def receive_from_dashboard(self, raw: str) -> str:
        return raw.upper()  # juste un exemple simple

    

    
    