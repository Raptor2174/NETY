# nety/core/brain.py

from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.cortex_limbic.memory_manager import MemoryManager
from nety.knowledge_base.knowledge_manager import KnowledgeManager
from nety.core.intent_analyzer import IntentAnalyzer
from nety.core.response_generator import ResponseGenerator


class Brain:
    """
    Le cerveau principal de NETY
    Orchestre tous les modules et gère l'interaction avec le système
    """
    
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
    
    def think(self, message: str) -> str:
        """
        Méthode principale pour traiter un message
        C'est l'entrée principale du Brain depuis le système NETY
        
        Args:
            message: Le message ou les données à traiter
        
        Returns:
            La réponse générée par le Brain
        """
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
        
        # [5] Validation
        validated_response = self.validate_response(response)
        
        # [6] Apprentissage
        self.update_memory(message, validated_response)
        
        return validated_response
    
    def retrieve_context(self, message: str, intent: dict) -> dict:
        """Récupère le contexte des deux mémoires"""
        
        # Contexte identité (cortex limbique)
        identity_context = {
            "personality": self.limbic_filter.personality.get_personality(),
            "rules": self.memory.get_rules()
        }
        
        # Contexte connaissances (base étendue)
        knowledge_context = self.knowledge.search(message, intent)
        
        return {
            "identity": identity_context,
            "knowledge": knowledge_context
        }
    
    def validate_response(self, response: str) -> str:
        """Valide la réponse"""
        # Logique de validation
        return response
    
    def update_memory(self, message: str, response: str):
        """Met à jour la mémoire"""
        summary = f"User: {message[:50]}... | Response: {response[:50]}..."
        self.memory.add_memory(summary)
    
    def get_modules_status(self) -> dict:
        """
        Retourne l'état de tous les modules du Brain
        Utilisé par le système pour synchroniser avec le Dashboard
        
        Returns:
            Dictionnaire {module_name: status}
        """
        return self.modules_status.copy()
    
    def get_context(self) -> list:
        """
        Retourne l'historique des interactions (contexte)
        Utilisé pour les tests et le debugging
        
        Returns:
            Liste des interactions {input, output}
        """
        return self.context_history.copy()


# Alias pour compatibilité
NETYBrain = Brain