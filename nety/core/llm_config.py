import torch
import os

class LLMConfig:
    """Configuration des modèles de langage"""
    
    # ✅ NOUVELLE OPTION : Backend intelligent
    SMART_BACKEND = True  # Active le système hybride
    PREFER_LOCAL = True   # Préfère le local quand disponible
    
    # Modèle actuel (local)
    CURRENT_MODEL = "mistral"
    
    # ✅ NOUVELLE SECTION : OpenAI
    OPENAI_CONFIG = {
        "enabled": True,
        "api_key": os.getenv("OPENAI_API_KEY", ""),  # À définir dans .env
        "model": "gpt-3.5-turbo",  # ou "gpt-4" si budget OK
        "max_tokens": 150,
        "temperature": 0.7,
    }
    
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
            "requires_gpu": False,  # Fonctionne aussi sur CPU
            "min_ram_gb": 8,
            "context_length": 32768,
            # ✅ NOUVEAU : Configuration GPU
            "gpu_vram_gb": 4,  # Nécessite ~4GB VRAM en 4-bit
        }
    }
    
    # ✅ MODIFIÉ : Paramètres optimisés pour GPU
    MISTRAL_GENERATION_CONFIG = {
        "max_new_tokens": 100,  # Réduit de 200 → 100
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "early_stopping": True,
    }
    
    # Quantization (optimisé pour GPU)
    USE_QUANTIZATION = True
    QUANTIZATION_BITS = 4  # ✅ 4-bit pour GPU (au lieu de 8)
    
    def get_device(self):
        """Détecte le device optimal"""
        if torch.cuda.is_available():
            return f"cuda:0 (GPU: {torch.cuda.get_device_name(0)})"
        return "cpu"
    
    def has_gpu(self):
        """Vérifie si GPU disponible"""
        return torch.cuda.is_available()