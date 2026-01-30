

# nety/core/llm_config.py
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class LLMConfig:
    """Configuration pour les modèles LLM"""
    
    # Modèle actuel
    CURRENT_MODEL = "mistral"  # ou "bloomz" pour revenir en arrière
    
    # Configurations disponibles
    MODELS = {
        "bloomz": {
            "name": "bigscience/bloomz-560m",
            "type": "bloom",
            "requires_gpu": False,
            "min_ram_gb": 2,
            "context_length": 2048
        },
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "type": "mistral",
            "requires_gpu": False,  # Fonctionne en CPU avec quantization
            "min_ram_gb": 8,
            "context_length": 32768
        }
    }
    
    # Paramètres de génération pour Mistral
    MISTRAL_GENERATION_CONFIG = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "length_penalty": 1.5,
        "do_sample": True,
    }
    
    # Paramètres de quantization
    USE_QUANTIZATION = True  # Activer pour économiser RAM
    QUANTIZATION_BITS = 4    # 4-bit ou 8-bit
    
    @classmethod
    def get_current_config(cls):
        """Retourne la config du modèle actuel"""
        return cls.MODELS[cls.CURRENT_MODEL]
    
    @classmethod
    def has_gpu(cls):
        """Détecte si GPU disponible"""
        return torch.cuda.is_available()
    
    @classmethod
    def get_device(cls):
        """Retourne le device optimal"""
        if cls.has_gpu():
            return "cuda"
        return "cpu"