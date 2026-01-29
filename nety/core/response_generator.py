from typing import Optional
from transformers import pipeline

class ResponseGenerator:
    """Génère les réponses de NETY"""
    
    def __init__(self):
        # Charger le modèle une seule fois
        self.generator = pipeline(
            "text-generation", 
            model="bigscience/bloomz-560m"
        )
    
    def generate(self, message: str, context: Optional[dict] = None, 
                 limbic_filter: Optional[dict] = None) -> str:
        """Génère une réponse avec les contraintes limbiques"""
        
        if context is None:
            context = {}
        if limbic_filter is None:
            limbic_filter = {'tone': 'friendly', 'behavior_rules': []}
        
        # Construire le prompt
        system_prompt = self._build_prompt(limbic_filter)
        
        # Enrichir avec le contexte
        knowledge = context.get('knowledge', '')
        full_prompt = f"""{system_prompt}

CONNAISSANCES:
{knowledge}

MESSAGE: {message}
RÉPONSE:"""
        
        # Appel LLM
        response = self._call_llm(full_prompt)
        
        return response
    
    def _build_prompt(self, limbic_filter: dict) -> str:
        """Construit le system prompt"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        return f"""Tu es NETY, une IA conversationnelle.

TON: {tone}
RÈGLES: {rules}"""
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM"""
        try:
            result = self.generator(
                prompt,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=getattr(getattr(self.generator, "tokenizer", None), "eos_token_id", None)
            )
            
            # Retirer le prompt de la réponse complète
            full_text = result[0]['generated_text']
            response = full_text[len(prompt):].strip()
            
            return response if response else "..."
            
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            return "Erreur de génération."