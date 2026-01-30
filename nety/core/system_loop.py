
from ...app.services.nety_adapter import NetyAdapter
from nety.services.nety_recepteur import NetyRecepteur

class SystemLoop:
    def __init__(self):
        self.running = True
        self.adapter = NetyAdapter()

    def run(self):
        while self.running:
            # 1) Vérifier si un message admin est arrivé
            admin_data = self.adapter.check_for_admin_message()
            if admin_data:
                # Transmettre à NETY IA pour traitement
                recepteur = NetyRecepteur()
                nety_request = recepteur.receive_from_dashboard(admin_data)

                # 2) Appeler la logique IA principale
                nety_response = self.process_nety(nety_request)

                # 3) Retourner la réponse au dashboard
                self.adapter.send_to_nety(nety_response)

    def process_nety(self, data):
        # C’est le pipeline logique de NETY
        return f"processed: {data}"
