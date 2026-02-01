"""
Cortex Limbique et Système Émotionnel de NETY

Module contenant:
  - Emotion Engine: Moteur d'émotions (ESM)
  - Limbic Filter: Filtrage émotionnel et personnalité
  - Memory Manager: Gestion de la mémoire
  - Personality Manager: Gestion de la personnalité
  - Textual Cortex: Cortex Textuel RNN Autonome (NOUVEAU!)
  - Advanced Limbic Filter: Filtres avancés
  - Emotional Memory: Mémoire émotionnelle
  - Adaptive Personality: Personnalité adaptative
"""

from nety.cortex_limbic.emotion_engine import EmotionEngine
from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.cortex_limbic.memory_manager import MemoryManager
from nety.cortex_limbic.personality_manager import PersonalityManager
from nety.cortex_limbic.textual_cortex import TextualCortex
from nety.cortex_limbic.advanced_limbic_filter import AdvancedLimbicFilter
from nety.cortex_limbic.emotional_memory import EmotionalMemory
from nety.cortex_limbic.adaptive_personality import AdaptivePersonality

__all__ = [
    "EmotionEngine",
    "LimbicFilter",
    "MemoryManager",
    "PersonalityManager",
    "TextualCortex",
    "AdvancedLimbicFilter",
    "EmotionalMemory",
    "AdaptivePersonality",
]

__version__ = "2.0"
__description__ = "Cortex Limbique et Système Neuronal de NETY avec RNN Autonome"
