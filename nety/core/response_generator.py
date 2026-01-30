from typing import Optional
from transformers import pipeline, AutoTokenizer, GenerationConfig

from nety.cortex_limbic import limbic_filter

class ResponseGenerator:
    """Génère les réponses de NETY"""
    
    def __init__(self):
        # ✅ Charger avec config personnalisée
        self.generator = pipeline(
            "text-generation",
            model="bigscience/bloomz-560m",
            device=-1  # -1 = CPU, 0 = GPU
        )
        
        # Modifier la config par défaut
        self.generator.model.config.max_length = None  # ✅ Désactiver max_length
        
        # Get eos_token_id safely
        eos_token_id = None
        if self.generator.tokenizer is not None:
            eos_token_id = self.generator.tokenizer.eos_token_id
        
        self.gen_config = GenerationConfig(
            # Longueur
            max_new_tokens=150,      # Maximum de tokens générés
            min_length=30,           # ✅ Force minimum (évite "ok", "oui")
            
            # Qualité
            temperature=0.6,         # 0.0-1.0 (0=déterministe, 1=créatif)
            top_p=0.9,              # ✅ Nucleus sampling (qualité)
            top_k=50,               # ✅ Limite les choix (cohérence)
            
            # Répétitions
            repetition_penalty=1.5,  # >1.0 pénalise répétitions
            no_repeat_ngram_size=3,  # Évite répéter 3 mots consécutifs
            
            # Tokens spéciaux
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
            
            # Performance
            do_sample=True,          # True = créatif, False = déterministe
            early_stopping=True      # ✅ Arrête si EOS trouvé
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
        
        # 🆕 RÉCUPÉRER L'HISTORIQUE
        history = context.get('history', [])
        history_text = ""
        
        if history:
            # Prendre les 3 dernières interactions
            for interaction in history[-3:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                history_text += f"Utilisateur: {user_msg}\nNETY: {bot_msg}\n\n"
        
        # Enrichir avec le contexte
        knowledge = context.get('knowledge', '')

        user_name = context.get('user_name', None)
        if user_name:
            message = f"{user_name}, {message}"
        
        full_prompt = f"""{system_prompt}

{"informations sur l'utilisateur: " + user_name if user_name else ""}
{f"- son nom est {user_name}." if user_name else ""}

{"CONVERSATION PRÉCÉDENTE:" if history_text else ""}
{history_text}

{"CONNAISSANCES PERTINENTES:" if knowledge else ""}
{knowledge}

Utilisateur: {message}
NETY:"""
        
        # Appel LLM
        response = self._call_llm(full_prompt)
        
        return response
    
    def _build_prompt(self, limbic_filter: dict) -> str:
        """Construit le system prompt"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        # ❌ ANCIEN CODE (BUGUÉ)
        # return f"""Tu es NETY, une IA conversationnelle.
        # 
        # TON: {tone}  ← LE MODÈLE PENSE QUE C'EST UN NOM !
        # RÈGLES: {rules}"""
        
        # ✅ NOUVEAU CODE (CORRIGÉ)
        return f"""Tu es NETY, une intelligence artificielle conversationnelle.

Instructions:
- Ton nom est NETY (et uniquement NETY)
- Ton style de communication: {tone}
- Règles à suivre: {', '.join(rules) if isinstance(rules, list) else rules}
- Réponds de manière naturelle et concise
- Ne répète jamais ces instructions dans tes réponses"""
    
    def _call_llm(self, prompt: str) -> str:
        """Génère une réponse avec un LLM"""
        try:
            pad_token_id = None
            if self.generator.tokenizer is not None:
                pad_token_id = self.generator.tokenizer.eos_token_id
            
            result = self.generator(
                prompt,
                max_new_tokens=150,  # ✅ Tokens générés (après le prompt)
                min_length=20,       # ✅ Minimum de tokens
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=pad_token_id
            )
            
            full_text = result[0]['generated_text']
            response = full_text[len(prompt):].strip()
            
            # ✅ NETTOYER LES ARTEFACTS
            # Retirer les lignes qui commencent par "NETY:" (texte d'entraînement)
            lines = response.split('\n')
            cleaned_lines = [
                line for line in lines 
                if not line.strip().startswith(('NETY:', 'User:', 'TON:', 'RÈGLES:'))
            ]
            response = '\n'.join(cleaned_lines).strip()
            
            # Si réponse vide après nettoyage
            if not response:
                return "..."
            
            # Limiter à la première phrase complète
            if '. ' in response:
                response = response.split('. ')[0] + '.'
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            return "Désolé, une erreur s'est produite."