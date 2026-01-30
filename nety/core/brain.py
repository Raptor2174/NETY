"""
Module Brain - Cerveau central de NETY
"""
# nety/core/brain.py
from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.cortex_limbic.memory_manager import MemoryManager
from nety.knowledge_base.knowledge_manager import KnowledgeManager
from nety.core.intent_analyzer import IntentAnalyzer
from nety.core.response_generator import ResponseGenerator


class Brain:
    """Le cerveau principal de NETYOrchestre tous les modules et gère l'interaction avec le système"""
    
    def __init__(self):
        # Initialisation des modules
        self.limbic_filter = LimbicFilter()
        self.memory = MemoryManager()
        self.knowledge = KnowledgeManager()
        self.intent_analyzer = IntentAnalyzer()
        self.response_generator = ResponseGenerator()
        
        # Historique des interactions pour get_context()
        self.context_history = []
        
        # État des modules
        self.modules_status = {
            "cortex_limbic": "actif",
            "memory": "actif",
            "knowledge_base": "actif",
            "intent_analyzer": "actif"
        }
        
        # Dictionnaire des modules pour compatibilité
        self.modules = {}
        self.context = {}
        self.state = "active"
    
    def think(self, message: str) -> str:
        """Méthode principale pour traiter un message"""
        
        # ✅ NETTOYER LE MESSAGE AVANT STOCKAGE
        # Retirer les préfixes même ici (au cas où)
        cleaned_message = message
        for prefix in ["CHAT: ", "PROMPT: ", "CHAT:", "PROMPT:"]:
            if cleaned_message.startswith(prefix):
                cleaned_message = cleaned_message[len(prefix):].strip()
                break
        
        # Stocker l'entrée NETTOYÉE
        interaction = {"input": cleaned_message}  # ✅ Version propre
        
        # Traiter le message via le pipeline complet
        response = self.process_message(cleaned_message)  # ✅ Ici aussi
        
        # Stocker la sortie
        interaction["output"] = response
        self.context_history.append(interaction)
        
        # Limiter l'historique à 100 interactions
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
        
        return response
    
    # Dans brain.py, méthode retrieve_context()

    def retrieve_context(self, message: str, intent: dict) -> dict:
        """Récupère le contexte basé sur le message et l'intention"""
        
        # ... code existant pour knowledge ...
        
        # ✅ EXTRAIRE LES INFORMATIONS CLÉS DE L'HISTORIQUE
        user_name = None
        for interaction in reversed(self.context_history[-10:]):
            user_msg = interaction.get('input', '').lower()
            # Détecter "je m'appelle X" ou "je suis X"
            if "je m'appel" in user_msg or "je suis" in user_msg:
                # Extraire le nom (simpliste)
                words = user_msg.split()
                try:
                    if "m'appel" in user_msg:
                        idx = words.index("m'appel") if "m'appel" in words else words.index("m'appelle")
                        user_name = words[idx + 1].strip('.,!?')
                    elif "je suis" in user_msg:
                        idx = words.index("suis")
                        user_name = words[idx + 1].strip('.,!?')
                except:
                    pass
        
        context = {
            "message": message,
            "intent": intent,
            "history": self.context_history[-5:],
            "knowledge": self.knowledge.get_knowledge(intent.get('type', 'general')) if hasattr(self.knowledge, 'get_knowledge') else {},
            "user_name": user_name  # ✅ Info clé extraite
        }
        return context
    
    def process_message(self, message: str) -> str:
        """Pipeline complet de traitement"""
        
        # [1] Analyse d'intention
        intent = self.intent_analyzer.analyze(message)
        
        # [2] Récupération contextuelle
        context = self.retrieve_context(message, intent)
        
        # [3] Filtrage par cortex limbique
        personality_filter = self.limbic_filter.apply_filter(context)
        
        # [4] Génération de réponse
        response = self.response_generator.generate(
            message, 
            context, 
            personality_filter
        )
        
        return response
    
    def get_modules_status(self):
        """Retourne l'état de tous les modules"""
        return self.modules_status.copy()
    
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
        summary = f"Input: {str(input_data)[:50]} | Output: {str(output_data)[:50]}"
        self.memory.add_memory(summary)
    
    def get_memory(self):
        """Récupère la mémoire"""
        return self.memory
    
    def clear(self):
        """Vide le Brain (mémoire et contexte)"""
        self.clear_memory()
        self.context_history = []

    def clear_all(self):
        """Vide complètement le Brain"""
        self.clear_memory()
        self.context_history = []
        # Réinitialiser KnowledgeManager en ré-instanciant l'objet
        self.knowledge = KnowledgeManager()

    def reset(self):
        """Réinitialise le Brain"""
        self.clear_memory()
        self.context_history = []
        self.state = "active"

    def clear_memory(self):
        """Vide la mémoire"""
        # Réinitialiser MemoryManager en ré-instanciant l'objet
        self.memory = MemoryManager()
    
    def set_context(self, key, value):
        """Définit un élément de contexte"""
        self.context[key] = value
    
    def get_context_value(self, key=None):
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
    
    def update_memory(self, message: str, response: str):
        """Met à jour la mémoire"""
        summary = f"User: {message[:50]}... | Response: {response[:50]}..."
        self.memory.add_memory(summary)
    
    def get_context(self) -> list:
        """
        Retourne l'historique des interactions (contexte)
        Utilisé pour les tests et le debugging
        
        Returns:
            List of interactions with {input, output} keys
            Liste des interactions {input, output}
        """
        return self.context_history.copy()


# Alias pour compatibilité
NETYBrain = Brain
