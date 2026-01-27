from app.services.nety_adapter import send_to_nety, check_for_admin_message
from nety.services.nety_recepteur import receive_from_dashboard

class SystemLoop:
    def __init__(self):
        self.running = True

    def run(self):
        while self.running:
            # 1) Vérifier si un message admin est arrivé
            admin_data = check_for_admin_message(self)
            if admin_data:
                # Transmettre à NETY IA pour traitement
                nety_request = receive_from_dashboard(admin_data, raw="true")

                # 2) Appeler la logique IA principale
                nety_response = self.process_nety(nety_request)

                # 3) Retourner la réponse au dashboard
                send_to_nety(self, nety_response)

    def process_nety(self, data):
        # C’est le pipeline logique de NETY
        return f"processed: {data}"
