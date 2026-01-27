from ui.admin_room import AdminRoomView


def get_modules_status():
    return [
        {"type": "CCM", "name": "TPM", "status": "running"},
        {"type": "LCM", "name": "ESM", "status": "inactive"},
        {"type": "BCM", "name": "IMCM", "status": "running"},
        {"type": "CBM", "name": "LM", "status": "idle"},
     ]


class NetyAdapter:
    def __init__(self):
        self.admin_room = None

    def set_admin_room(self, admin_room):
        """Définit la référence à AdminRoom"""
        self.admin_room = admin_room

    def process_prompt(self, prompt):
        """Traite un prompt envoyé par l'interface"""
        try:
            # Simule le traitement du prompt
            response = self._execute_prompt(prompt)
            return response
        except Exception as e:
            return f"Erreur lors du traitement: {str(e)}"

    def _execute_prompt(self, prompt):
        """Exécute le prompt"""
        # Placeholder pour la logique IA
        return f"IA réponse à: {prompt}"




