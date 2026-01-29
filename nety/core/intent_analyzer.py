class IntentAnalyzer:
    """Analyzes user intent from messages"""
    
    def analyze(self, message: str) -> dict:
        """
        Analyzes the intent of a message
        
        Args:
            message: The user's message
            
        Returns:
            dict: Intent information including type, confidence, and parameters
        """
        # Basic intent detection logic
        intent_type = self._detect_intent_type(message)
        confidence = self._calculate_confidence(message, intent_type)
        parameters = self._extract_parameters(message)
        
        return {
            "type": intent_type,
            "confidence": confidence,
            "parameters": parameters
        }
    
    def _detect_intent_type(self, message: str) -> str:
        """Detects the type of intent"""
        # TODO: Implement intent classification logic
        return "general"
    
    def _calculate_confidence(self, message: str, intent_type: str) -> float:
        """Calculates confidence score"""
        # TODO: Implement confidence scoring
        return 0.5
    
    def _extract_parameters(self, message: str) -> dict:
        """Extracts parameters from the message"""
        # TODO: Implement parameter extraction
        return {}
# nety/core/intent_analyzer.py


class IntentAnalyzer:
    """
    Analyse l'intention derrière un message utilisateur
    Détermine le type de requête et extrait les entités clés
    """
    
    def __init__(self):
        # Catégories d'intentions de base
        self.intent_categories = {
            "question": ["comment", "pourquoi", "quoi", "qui", "où", "quand"],
            "command": ["fais", "crée", "génère", "transforme", "calcule"],
            "greeting": ["bonjour", "salut", "hey", "coucou", "hello"],
            "farewell": ["au revoir", "bye", "à plus", "ciao"],
            "thanks": ["merci", "thanks"],
        }
    
    def analyze(self, message: str) -> dict:
        """
        Analyse le message et retourne l'intention détectée
        
        Args:
            message: Le message à analyser
        
        Returns:
            Dictionnaire contenant l'intention et les métadonnées
        """
        if not message or not isinstance(message, str):
            return {"type": "unknown", "confidence": 0.0, "entities": []}
        
        message_lower = message.lower().strip()
        
        # Détection spécifique pour les questions (priorité haute)
        if "?" in message:
            return {
                "type": "question",
                "confidence": 0.9,
                "entities": self._extract_entities(message),
                "original_message": message
            }
        
        # Détection par mots-clés (priorité moyenne)
        detected_intent = "statement"  # Par défaut
        confidence = 0.5
        
        for intent_type, keywords in self.intent_categories.items():
            for keyword in keywords:
                if keyword in message_lower:
                    detected_intent = intent_type
                    confidence = 0.8
                    # Sortir dès la première correspondance trouvée
                    return {
                        "type": detected_intent,
                        "confidence": confidence,
                        "entities": self._extract_entities(message),
                        "original_message": message
                    }
        
        return {
            "type": detected_intent,
            "confidence": confidence,
            "entities": self._extract_entities(message),
            "original_message": message
        }
    
    def _extract_entities(self, message: str) -> list:
        """
        Extrait les entités du message (version simplifiée)
        
        Returns:
            Liste d'entités détectées
        """
        # Pour l'instant, retourne une liste vide
        # TODO: Implémenter extraction d'entités avec spaCy
        return []
