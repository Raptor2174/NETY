

# nety/cortex_limbic/limbic_filter.py

from .personality_manager import PersonalityManager
from .memory_manager import MemoryManager

class LimbicFilter:
    """Applique les filtres du cortex limbique"""
    
    def __init__(self):
        self.personality = PersonalityManager()
        self.memory = MemoryManager()
    
    def apply_filter(self, context: dict) -> dict:
        """
        Applique le filtre limbique sur le contexte
        Retourne les contraintes de génération
        """
        
        personality_data = self.personality.get_personality()
        rules = self.memory.get_rules()
        
        filter_params = {
            "tone": self.personality.determine_tone(),
            "forbidden_topics": personality_data["preferences"]["dislikes"],
            "encouraged_topics": personality_data["preferences"]["likes"],
            "behavior_rules": rules,
            "emotional_level": self.personality.calculate_emotion()
        }
        
        return filter_params