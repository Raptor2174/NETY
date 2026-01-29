# nety/core/response_generator.py
from typing import Optional
from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.knowledge_base.knowledge_manager import KnowledgeManager

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

class ResponseGenerator:
    def __init__(self):
        # Charger un modèle léger en français
        self.model_name = "bigscience/bloomz-560m"  # 560M paramètres, léger
        self.generator = pipeline("text-generation", model=self.model_name)
    
    
    
    def generate(self, message: str, context: Optional[dict] = None, personality_filter: Optional[dict] = None) -> str:
        """Generate a response based on message and context."""
        if context is None:
            context = {}
        if personality_filter is None:
            personality_filter = {'tone': 'neutral', 'behavior_rules': ''}
        
        # Safely access knowledge with a default value
        knowledge = context.get('knowledge', '')
        
        # Use the knowledge variable in your prompt/logic
        # Example:
        prompt = f"""
        Context knowledge: {knowledge}
        User message: {message}
        
        Generate an appropriate response.
        """
        
        # Construire le prompt
        system_prompt = self._build_prompt(personality_filter)
        
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
        """Génère une vraie réponse avec un LLM"""
        try:
            result = self.generator(
                prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7
            )
            return result[0]['generated_text']
        except Exception as e:
            return f"Erreur de génération: {str(e)}"
        
