from typing import Dict, List
import json
from datetime import datetime

class AdaptivePersonality:
    """Module de gestion de la personnalité unique (UPMM - Unique Personality Management Module)"""
    
    def __init__(self):
        # Modèle Big Five de la personnalité
        self.traits = {
            "ouverture": 0.75,        # Curiosité intellectuelle
            "conscience": 0.70,       # Organisation et fiabilité
            "extraversion": 0.65,     # Sociabilité et énergie
            "agréabilité": 0.80,      # Empathie et coopération
            "neuroticisme": 0.20      # Stabilité émotionnelle (inverse)
        }
        
        # Sous-traits spécifiques à NETY
        self.sub_traits = {
            "humour": 0.60,
            "formalité": 0.30,
            "créativité": 0.75,
            "patience": 0.70,
            "assertivité": 0.50
        }
        
        # Préférences apprises
        self.learned_preferences = {
            "conversation_style": "conversationnel",  # conversationnel, formel, humoristique
            "verbosité": "moyen",  # concis, moyen, détaillé
            "topics_expertise": [],
            "user_interaction_patterns": {}
        }
        
        # Historique d'adaptation
        self.adaptation_history = []
    
    def adapt_to_user(self, user_id: str, interaction_data: dict):
        """Adapte la personnalité en fonction de l'utilisateur"""
        
        if user_id not in self.learned_preferences["user_interaction_patterns"]:
            self.learned_preferences["user_interaction_patterns"][user_id] = {
                "preferred_tone": "friendly",
                "interaction_count": 0,
                "positive_feedback": 0,
                "topics_discussed": []
            }
        
        user_profile = self.learned_preferences["user_interaction_patterns"][user_id]
        user_profile["interaction_count"] += 1
        
        # Apprendre du feedback
        if interaction_data.get("positive_feedback"):
            user_profile["positive_feedback"] += 1
            
            # Renforcer les traits qui ont fonctionné
            current_style = interaction_data.get("style_used")
            if current_style == "humoristique":
                self.sub_traits["humour"] = min(1.0, self.sub_traits["humour"] + 0.05)
        
        # Adapter la verbosité
        if interaction_data.get("user_preference") == "concis":
            self.learned_preferences["verbosité"] = "concis"
        
        self._record_adaptation(user_id, interaction_data)
    
    def get_personality_for_context(self, context: dict) -> dict:
        """Retourne la configuration de personnalité adaptée au contexte"""
        
        user_id = context.get("user_id")
        intent = context.get("intent", {})
        
        # Personnalité de base
        personality_config = {
            "traits": self.traits.copy(),
            "sub_traits": self.sub_traits.copy(),
            "tone": self._determine_tone(context),
            "verbosity": self._determine_verbosity(context),
            "style": self._determine_style(context)
        }
        
        # Ajustements contextuels
        if intent.get("type") == "question":
            personality_config["sub_traits"]["patience"] = min(1.0, self.sub_traits["patience"] + 0.1)
            personality_config["traits"]["conscience"] = min(1.0, self.traits["conscience"] + 0.1)
        
        # Ajustements spécifiques à l'utilisateur
        if user_id and user_id in self.learned_preferences["user_interaction_patterns"]:
            user_prefs = self.learned_preferences["user_interaction_patterns"][user_id]
            personality_config["tone"] = user_prefs.get("preferred_tone", personality_config["tone"])
        
        return personality_config
    
    def _determine_tone(self, context: dict) -> str:
        """Détermine le ton approprié"""
        emotional_state = context.get("emotional_state", {})
        dominant_emotion = emotional_state.get("dominant_emotion", "confiance")
        
        tone_mapping = {
            "joie": "enthousiaste",
            "tristesse": "empathique",
            "confiance": "amical",
            "surprise": "curieux",
            "colère": "calme",
            "peur": "rassurant"
        }
        
        return tone_mapping.get(dominant_emotion, "amical")
    
    def _determine_verbosity(self, context: dict) -> str:
        """Détermine le niveau de détail"""
        return self.learned_preferences.get("verbosité", "moyen")
    
    def _determine_style(self, context: dict) -> str:
        """Détermine le style de communication"""
        if self.sub_traits["humour"] > 0.7 and context.get("allow_humor", True):
            return "humoristique"
        elif self.sub_traits["formalité"] > 0.6:
            return "formel"
        else:
            return "conversationnel"
    
    def _record_adaptation(self, user_id: str, data: dict):
        """Enregistre les adaptations pour analyse"""
        self.adaptation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "adaptation": data
        })
        
        # Garder seulement les 1000 dernières adaptations
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]