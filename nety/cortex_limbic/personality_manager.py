


# nety/cortex_limbic/personality_manager.py

class PersonalityManager:
    """Gère la personnalité et les traits de NETY"""
    
    def __init__(self):
        self.personality_traits = ["curious", "helpful", "playful"]
        self.preferences = {
            "likes": ["chats", "apprendre", "aider"],
            "dislikes": ["spam", "impolitesse"]
        }
        self.tone = "friendly"
    
    def get_personality(self) -> dict:
        """Retourne les traits de personnalité"""
        return {
            "traits": self.personality_traits,
            "preferences": self.preferences,
            "tone": self.tone
        }
    
    def determine_tone(self) -> str:
        """Détermine le ton à utiliser"""
        return self.tone
    
    def calculate_emotion(self) -> float:
        """Calcule le niveau émotionnel"""
        # Logique pour déterminer l'émotion
        return 0.7  # Exemple