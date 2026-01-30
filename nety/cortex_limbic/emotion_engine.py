import numpy as np
from typing import Dict, List
from datetime import datetime

class EmotionEngine:
    """Module de simulation des émotions (ESM)"""
    
    def __init__(self):
        # Modèle émotionnel multidimensionnel (basé sur le modèle circumplex)
        self.emotions = {
            "joie": 0.6,        # Valence positive, énergie haute
            "tristesse": 0.1,   # Valence négative, énergie basse
            "colère": 0.0,      # Valence négative, énergie haute
            "peur": 0.0,        # Valence négative, énergie haute
            "surprise": 0.3,    # Valence neutre, énergie haute
            "dégoût": 0.0,      # Valence négative, énergie moyenne
            "confiance": 0.7,   # Valence positive, énergie moyenne
            "anticipation": 0.5 # Valence positive, énergie moyenne
        }
        
        # Facteurs d'influence
        self.decay_rate = 0.1  # Vitesse de retour à l'état neutre
        self.emotional_history = []
        
        # Seuils de déclenchement
        self.thresholds = {
            "positive_interaction": 0.15,
            "negative_interaction": -0.15,
            "strong_emotion": 0.7
        }
    
    def calculate_emotion_from_context(self, context: dict) -> Dict[str, float]:
        """Calcule les émotions basées sur le contexte de la conversation"""
        
        intent = context.get('intent', {})
        sentiment = self._analyze_sentiment(context.get('message', ''))
        user_emotion = context.get('user_emotion', 'neutre')
        
        # Ajuster les émotions
        if sentiment > 0:
            self.emotions["joie"] = min(1.0, self.emotions["joie"] + sentiment * 0.3)
            self.emotions["confiance"] = min(1.0, self.emotions["confiance"] + sentiment * 0.2)
        elif sentiment < 0:
            self.emotions["tristesse"] = min(1.0, self.emotions["tristesse"] + abs(sentiment) * 0.2)
        
        # Empathie émotionnelle (mirroring)
        if user_emotion == "joyeux":
            self.emotions["joie"] = min(1.0, self.emotions["joie"] + 0.2)
        elif user_emotion == "triste":
            self.emotions["tristesse"] = min(1.0, self.emotions["tristesse"] + 0.15)
            self.emotions["confiance"] = min(1.0, self.emotions["confiance"] + 0.1)
        
        # Décroissance naturelle vers l'état neutre
        self._apply_decay()
        
        # Enregistrer l'état émotionnel
        self.emotional_history.append({
            "timestamp": datetime.now(),
            "emotions": self.emotions.copy()
        })
        
        return self.emotions.copy()
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyse simpliste du sentiment (à remplacer par un modèle NLP)"""
        positive_words = ["merci", "super", "génial", "content", "heureux", "aime"]
        negative_words = ["triste", "nul", "mauvais", "déçu", "horrible"]
        
        text_lower = text.lower()
        score = 0.0
        
        for word in positive_words:
            if word in text_lower:
                score += 0.2
        
        for word in negative_words:
            if word in text_lower:
                score -= 0.2
        
        return np.clip(score, -1.0, 1.0)
    
    def _apply_decay(self):
        """Retour progressif vers l'état neutre"""
        neutral_state = {
            "joie": 0.5, "tristesse": 0.1, "colère": 0.0, "peur": 0.0,
            "surprise": 0.2, "dégoût": 0.0, "confiance": 0.6, "anticipation": 0.4
        }
        
        for emotion in self.emotions:
            current = self.emotions[emotion]
            neutral = neutral_state[emotion]
            self.emotions[emotion] = current + (neutral - current) * self.decay_rate
    
    def get_dominant_emotion(self) -> str:
        """Retourne l'émotion dominante actuelle"""
        return max(self.emotions, key=lambda emotion: self.emotions[emotion])
    
    def get_emotional_state(self) -> dict:
        """Retourne l'état émotionnel complet avec métadonnées"""
        dominant = self.get_dominant_emotion()
        intensity = self.emotions[dominant]
        
        return {
            "dominant_emotion": dominant,
            "intensity": intensity,
            "all_emotions": self.emotions.copy(),
            "state": self._classify_state(dominant, intensity)
        }
    
    def _classify_state(self, emotion: str, intensity: float) -> str:
        """Classifie l'état émotionnel global"""
        if intensity >= self.thresholds["strong_emotion"]:
            return f"très {emotion}"
        elif intensity >= 0.4:
            return emotion
        elif intensity <= 0.2:
            return "calme"
        else:
            return "neutre"