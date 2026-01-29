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
        """
        Méthode principale pour traiter un message
        C'est l'entrée principale du Brain depuis le système NETY
        
        Args:
            message: Le message ou les données à traiter
        
        Returns:
            La réponse générée par le Brain
        """
        try:
            # Stocker l'entrée
            interaction = {"input": message}
            
            # Traiter le message via le pipeline complet
            response = self.process_message(message)
            
            # Stocker la sortie
            interaction["output"] = response
            self.context_history.append(interaction)
            
            # Limiter l'historique à 100 interactions
            if len(self.context_history) > 100:
                self.context_history = self.context_history[-100:]
            
            return response
        except Exception as e:
            error_msg = f"Erreur dans Brain.think(): {type(e).__name__}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return f"Erreur de traitement: {str(e)}"
    
    def retrieve_context(self, message: str, intent: dict) -> dict:
        """
        Récupère le contexte basé sur le message et l'intention
        """
        
        # 🆕 RECHERCHER DES CONNAISSANCES PERTINENTES
        knowledge_content = ""
        try:
            # Utiliser le SearchEngine pour trouver du contexte
            from nety.knowledge_base import SearchEngine
            search = SearchEngine()
            results = search.search(message, limit=3, use_semantic=False)
            
            if results:
                knowledge_content = "\n\n".join([
                    f"[{r.get('title', 'Sans titre')}]\n{r.get('content', '')}" 
                    for r in results
                ])
        except Exception as e:
            print(f"⚠️ Erreur recherche KB: {e}")
            knowledge_content = ""
        
        context = {
            "message": message,
            "intent": intent,
            "history": self.context_history[-5:] if self.context_history else [],
            "knowledge": knowledge_content  # ✅ AJOUT DE LA CLÉ !
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
