


class NEYCorex:
    def __init__(self):
        self.cortex_limbique = self.load_local_memory()
        self.knowledge_base = self.load_knowledge()
        
    def process_message(self, message: str) -> str:
        """Pipeline complet de traitement"""
        
        # [1] Analyse d'intention
        intent = self.analyze_intent(message)
        
        # [2] Récupération contextuelle
        context = self.retrieve_context(message, intent)
        
        # [3] Filtrage par cortex limbique
        personality_filter = self.apply_limbic_filter(context)
        
        # [4] Génération de réponse
        response = self.generate_response(
            message, 
            context, 
            personality_filter
        )
        
        # [5] Validation
        validated_response = self.validate_response(response)
        
        # [6] Apprentissage
        self.update_memory(message, validated_response)
        
        return validated_response
    
    def analyze_intent(self, message: str) -> dict:
        """Détermine ce que l'utilisateur veut"""
        # Avec un LLM ou classification simple
        return {
            "type": "question|command|conversation",
            "topic": "...",
            "emotion": "...",
            "urgency": 0.0-1.0
        }
    
    def retrieve_context(self, message: str, intent: dict) -> dict:
        """Cherche dans les deux types de mémoire"""
        
        # Mémoire locale (identité)
        identity_context = {
            "personality": self.cortex_limbique["personality_traits"],
            "preferences": self.cortex_limbique["preferences"],
            "learned_rules": self.cortex_limbique["rules"]
        }
        
        # Base étendue (connaissances)
        knowledge_context = self.search_knowledge(message, intent)
        
        return {
            "identity": identity_context,
            "knowledge": knowledge_context
        }
    
    def apply_limbic_filter(self, context: dict) -> dict:
        """
        Le cortex limbique MODIFIE la façon de répondre
        selon la personnalité et les règles internes
        """
        
        personality = context["identity"]["personality"]
        preferences = context["identity"]["preferences"]
        rules = context["identity"]["learned_rules"]
        
        # Construire les contraintes de génération
        filter_params = {
            # Ton et style
            "tone": self._determine_tone(personality),
            
            # Interdictions
            "forbidden_topics": preferences.get("dislikes", []),
            
            # Préférences
            "encouraged_topics": preferences.get("likes", []),
            
            # Règles comportementales
            "behavior_rules": rules,
            
            # Niveau d'émotion
            "emotional_level": self._calculate_emotion(personality)
        }
        
        return filter_params
    
    def generate_response(self, message: str, context: dict, 
                        limbic_filter: dict) -> str:
        """
        Génère la réponse en respectant le filtre limbique
        """
        
        # Construction du prompt avec contraintes
        system_prompt = self._build_constrained_prompt(limbic_filter)
        
        # Ajout du contexte
        enriched_prompt = f"""
    {system_prompt}

    PERSONNALITÉ: {limbic_filter['tone']}
    RÈGLES: {limbic_filter['behavior_rules']}
    INTERDICTIONS: {limbic_filter['forbidden_topics']}

    CONNAISSANCES DISPONIBLES:
    {context['knowledge']}

    MESSAGE UTILISATEUR: {message}
    """
        
        # Génération (avec OpenAI, local LLM, etc.)
        response = self.llm_generate(enriched_prompt)
        
        return response

    def _build_constrained_prompt(self, limbic_filter: dict) -> str:
        """Construit le system prompt basé sur le cortex limbique"""
        return f"""
    Tu es NETY. Voici ta personnalité et tes règles INTERNES:

    TON: {limbic_filter['tone']}
    TU AIMES: {limbic_filter['encouraged_topics']}
    TU N'AIMES PAS: {limbic_filter['forbidden_topics']}

    RÈGLES COMPORTEMENTALES:
    {chr(10).join(f"- {rule}" for rule in limbic_filter['behavior_rules'])}

    Réponds TOUJOURS en respectant ces contraintes.
    """

    def validate_response(self, response: str) -> str:
        """Vérifie que la réponse respecte les règles"""
        
        # Vérifications
        rules = self.cortex_limbique["rules"]
        forbidden = self.cortex_limbique["preferences"]["dislikes"]
        
        # Si violation, régénérer ou corriger
        if self._violates_rules(response, rules, forbidden):
            return self._correct_response(response)
        
        return response
    
    def update_memory(self, message: str, response: str):
        """Apprentissage et consolidation"""
        
        # Résumé de l'interaction
        summary = self._summarize_interaction(message, response)
        
        # Ajout à la mémoire locale (consolidée)
        self.cortex_limbique["consolidated_memories"].append(summary)
        
        # Nettoyage périodique
        if len(self.cortex_limbique["consolidated_memories"]) > 100:
            self._compress_memories()

    