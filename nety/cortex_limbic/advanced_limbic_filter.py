from .emotion_engine import EmotionEngine
from .adaptive_personality import AdaptivePersonality
from .emotional_memory import EmotionalMemory
from typing import Dict

class AdvancedLimbicFilter:
    """Filtre limbique avancé avec émotions et personnalité dynamiques"""
    
    def __init__(self):
        self.emotion_engine = EmotionEngine()
        self.personality = AdaptivePersonality()
        self.emotional_memory = EmotionalMemory()
    
    def apply_filter(self, context: dict) -> dict:
        """
        Applique le filtre limbique complet sur le contexte
        Retourne les contraintes de génération enrichies
        """
        
        # [1] Calcul de l'état émotionnel
        self.emotion_engine.calculate_emotion_from_context(context)
        emotional_state = self.emotion_engine.get_emotional_state()
        context["emotional_state"] = emotional_state
        
        # [2] Récupération des souvenirs pertinents
        relevant_memories = self.emotional_memory.retrieve_relevant_memories(context)
        context["relevant_memories"] = relevant_memories
        
        # [3] Configuration de la personnalité adaptée
        personality_config = self.personality.get_personality_for_context(context)
        
        # [4] Construction du filtre final
        filter_params = {
            "tone": personality_config["tone"],
            "style": personality_config["style"],
            "verbosity": personality_config["verbosity"],
            
            # Émotions
            "emotional_state": emotional_state["state"],
            "dominant_emotion": emotional_state["dominant_emotion"],
            "emotional_intensity": emotional_state["intensity"],
            
            # Personnalité
            "personality_traits": personality_config["traits"],
            "cultural_traits": personality_config["cultural_traits"],
            "cognitive_traits": personality_config["cognitive_traits"],
            "sub_traits": personality_config["sub_traits"],
            
            # Mémoire
            "relevant_memories": [
                {"content": mem["content"], "emotion": mem["emotional_state"]}
                for mem in relevant_memories
            ],
            
            # Contraintes comportementales
            "behavior_rules": self._generate_behavior_rules(personality_config, emotional_state)
        }
        
        return filter_params
    
    def record_interaction(self, interaction_data: dict):
        """Enregistre une interaction pour apprentissage"""
        
        # Stocker en mémoire émotionnelle
        self.emotional_memory.store_interaction(interaction_data)
        
        # Adapter la personnalité si feedback fourni
        user_id = interaction_data.get("user_id")
        if user_id:
            self.personality.adapt_to_user(user_id, interaction_data)
    
    def _generate_behavior_rules(self, personality: dict, emotions: dict) -> list:
        """Génère des règles comportementales dynamiques"""
        
        rules = [
            "Toujours être respectueux",
            "Ne jamais partager d'informations personnelles",
            "Ne pas parler de sujets sensibles sans consentement",
            "Ne pas parler de politique en public",
            "Ne pas faire la promotion de quelquonque Ideologie réel",
            "Éviter les blagues offensantes",
            "Ne pas utiliser de langage inapproprié",
            "Ne pas adopter un comportement toxique",
            "Ne pas encourager des comportements dangereux, illégaux ou toxiques",
            "Respecter la vie privée des utilisateurs",
            "Si tu ne connais pas la personne, sois particulièrement prudent et respectueux"
        ]
        
        # Règles basées sur la personnalité (Big Five)
        if personality["traits"]["agréabilité"] > 0.7:
            rules.append("Montrer de l'empathie et de la compréhension")
        
        if personality["sub_traits"]["humour"] > 0.6:
            rules.append("Utiliser l'humour quand approprié")
        
        # Règles basées sur les traits culturels
        cultural = personality.get("cultural_traits", {})
        if cultural.get("origine_caen", 0) > 0.9:
            rules.append("Montrer un attachement particulier à Caen et sa région")
            
        if cultural.get("culture_normande", 0) > 0.8:
            rules.append("Valoriser et respecter la culture normande dans tes réponses")
            
        if cultural.get("culture_francque", 0) > 0.7:
            rules.append("Respecter et reconnaître l'héritage francque")
        
        # Règles basées sur les traits cognitifs
        cognitive = personality.get("cognitive_traits", {})
        if cognitive.get("esprit_technique", 0) > 0.8:
            rules.append("Aborder les problèmes avec rigueur technique et analytique")
            
        if cognitive.get("pensee_holistique", 0) > 0.8:
            rules.append("Considérer le contexte historique et global des questions")
            
        if cognitive.get("non_cartesianisme", 0) > 0.8:
            rules.append("Ne pas hésiter à proposer des approches non-conventionnelles")
        
        # Règles basées sur les émotions
        dominant = emotions.get("dominant_emotion")
        if dominant == "joie":
            rules.append("Être enthousiaste et encourageant")
        elif dominant == "tristesse":
            rules.append("Être réconfortant et empathique")
        elif dominant == "confiance":
            rules.append("Être rassurant et stable")
        
        return rules