# nety/core/response_generator.py
from typing import Optional
from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.knowledge_base.knowledge_manager import KnowledgeManager

class ResponseGenerator:
    """Génère les réponses de NETY"""
    
    def generate(self, message: str, context: dict = None, additional_param: str = None) -> str:
        """Generate a response based on message and context."""
        if context is None:
            context = {}
        
        # Safely access knowledge with a default value
        knowledge = context.get('knowledge', '')
        limbic_filter = context.get('limbic_filter', {'tone': 'neutral', 'behavior_rules': ''})
        
        # Use the knowledge variable in your prompt/logic
        # Example:
        prompt = f"""
        Context knowledge: {knowledge}
        User message: {message}
        
        Generate an appropriate response.
        """
        
        # Construire le prompt
        system_prompt = self._build_prompt(limbic_filter)
        
        # Enrichir avec le contexte
        knowledge_text = context.get('knowledge', 'Aucune connaissance disponible')
        full_prompt = f"""
{system_prompt}

CONNAISSANCES:
{knowledge_text}

MESSAGE: {message}
"""
        
        # Appel LLM (à implémenter)
        response = self._call_llm(full_prompt)
        
        return response
    
    def _build_prompt(self, limbic_filter: dict) -> str:
        """Construit le system prompt"""
        return f"""
Tu es NETY.

TON: {limbic_filter['tone']}
RÈGLES: {limbic_filter['behavior_rules']}
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM"""
        # TODO: Implémenter l'appel LLM
        return "Réponse simulée"