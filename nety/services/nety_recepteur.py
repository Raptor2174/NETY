from app.services.nety_adapter import NetyAdapter

class NetyRecepteur:
    def __init__(self):
        self.adapter = NetyAdapter()

    def recevoir_prompt(self, prompt):
        """Reçoit un prompt et le transmet à l'adapter"""
        return self.adapter.process_prompt(prompt)
    
    