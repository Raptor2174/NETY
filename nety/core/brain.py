"""
Module Brain - Cerveau central de NETY
"""


class Brain:
    """
    Le cerveau principal de NETY
    Orchestre tous les modules et gère l'interaction avec le système
    """
    
    def __init__(self):
        """Initialise le Brain et ses modules"""
        self.modules = {}
        self.state = "initialized"
        self.memory = []
        self.context = {}
        print("🧠 Brain initialisé")
    
    def think(self, input_data):
        """
        Traite une donnée d'entrée
        
        Args:
            input_data: Données à traiter (str, dict, etc.)
            
        Returns:
            Résultat du traitement
        """
        # Logique de traitement basique
        if isinstance(input_data, str):
            response = f"Brain a traité: {input_data}"
        else:
            response = f"Brain a traité: {str(input_data)}"
        
        # Sauvegarder en mémoire
        self.add_to_memory(input_data, response)
        
        return response
    
    def get_modules_status(self):
        """
        Retourne l'état de tous les modules
        
        Returns:
            dict: État des modules
        """
        return {
            "brain": self.state,
            "modules_count": len(self.modules),
            "active": True
        }
    
    def register_module(self, name, module):
        """Enregistre un nouveau module"""
        self.modules[name] = module
    
    def shutdown(self):
        """Arrête proprement le Brain"""
        self.state = "stopped"
        print("🧠 Brain arrêté")
    
    # ===== Méthodes supplémentaires pour compatibilité =====
    
    def process(self, data):
        """Alias pour think()"""
        return self.think(data)
    
    def add_to_memory(self, input_data, output_data):
        """Ajoute une interaction à la mémoire"""
        self.memory.append({
            "input": input_data,
            "output": output_data
        })
        # Limiter la mémoire à 100 entrées
        if len(self.memory) > 100:
            self.memory.pop(0)
    
    def get_memory(self):
        """Récupère la mémoire"""
        return self.memory
    
    def clear_memory(self):
        """Vide la mémoire"""
        self.memory = []
    
    def set_context(self, key, value):
        """Définit un élément de contexte"""
        self.context[key] = value
    
    def get_context(self, key=None):
        """Récupère le contexte ou un élément"""
        if key is None:
            return self.context
        return self.context.get(key)
    
    def get_module(self, name):
        """Récupère un module spécifique"""
        return self.modules.get(name)
    
    def activate_module(self, name):
        """Active un module"""
        if name in self.modules:
            module = self.modules[name]
            if hasattr(module, 'activate'):
                module.activate()
            return True
        return False
    
    def deactivate_module(self, name):
        """Désactive un module"""
        if name in self.modules:
            module = self.modules[name]
            if hasattr(module, 'deactivate'):
                module.deactivate()
            return True
        return False
    
    def get_state(self):
        """Retourne l'état actuel du Brain"""
        return self.state
    
    def set_state(self, new_state):
        """Définit un nouvel état"""
        self.state = new_state