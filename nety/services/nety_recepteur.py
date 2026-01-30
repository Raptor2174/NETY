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

    def check_for_admin_message(self):
        """
        Vérifie s'il y a des messages provenant de l'administrateur
        via le tableau de bord.
        """
        message = self.adapter.get_admin_message()
        return message
    
    def check_all_admin_messages(self):
        """
        Récupère tous les messages admin en attente.
        """
        messages = self.adapter.get_all_admin_messages()
        return messages

    def receive_from_dashboard(self, raw: str) -> str:
        return raw.upper()  # juste un exemple simple





