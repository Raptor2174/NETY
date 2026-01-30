


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
