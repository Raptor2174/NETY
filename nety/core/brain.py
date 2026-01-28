# nety/core/nety_brain.py

from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.cortex_limbic.memory_manager import MemoryManager
from nety.knowledge_base.knowledge_manager import KnowledgeManager
from nety.core.intent_analyzer import IntentAnalyzer
from nety.core.response_generator import ResponseGenerator

class NETYBrain:
    """
    Le cerveau principal de NETY
    Orchestre tous les modules
    """
    
    def __init__(self):
        # Initialisation des modules
        self.limbic_filter = LimbicFilter()
        self.memory = MemoryManager()
        self.knowledge = KnowledgeManager()
        self.intent_analyzer = IntentAnalyzer()
        self.response_generator = ResponseGenerator()
    
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