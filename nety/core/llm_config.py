import torch
import os

class LLMConfig:
    """Configuration des modèles de langage - 100% Gratuit"""
    
    # Backend intelligent
    SMART_BACKEND = True
    PREFER_LOCAL = True
    
    # ═══════════════════════════════════════════════════
    # GROQ CLOUD (OPTION 3)
    # ═══════════════════════════════════════════════════
    
    GROQ_CONFIG = {
        "enabled": True,
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "base_url": "https://api.groq.com/openai/v1",
        
        # Modèles disponibles (gratuits)
        "models": {
            "llama-3.2-1b-preview": {
                "context_window": 128000,
                "max_tokens": 150,
                "speed": "ultra",  # ~500 tok/sec
                "quality": 3,
                "description": "Léger et rapide"
            },
            "llama-3.2-3b-preview": {
                "context_window": 128000,
                "max_tokens": 150,
                "speed": "ultra",
                "quality": 4,
                "description": "⭐ Optimal - Meilleur rapport qualité/vitesse"
            },
            "llama-3.1-8b-instant": {
                "context_window": 128000,
                "max_tokens": 150,
                "speed": "fast",
                "quality": 5,
                "description": "Plus intelligent"
            },
            "mixtral-8x7b-32768": {
                "context_window": 32768,
                "max_tokens": 150,
                "speed": "medium",
                "quality": 5,
                "description": "Très puissant (47B params)"
            },
        },
        
        "default_model": "llama-3.2-3b-preview",  # ⭐ RECOMMANDÉ
        "temperature": 0.7,
        
        # Limites gratuites
        "rate_limits": {
            "requests_per_minute": 30,
            "requests_per_day": 14400,
            "tokens_per_minute": 6000,
        }
    }
    
    # ═══════════════════════════════════════════════════
    # MISTRAL LOCAL (OPTION 1)
    # ═══════════════════════════════════════════════════
    
    MODELS = {
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "type": "mistral",
            "requires_gpu": True,
            "min_ram_gb": 8,
            "gpu_vram_gb": 4,
            "context_length": 32768,
        },
        "bloomz": {
            "name": "bigscience/bloomz-560m",
            "type": "bloom",
            "requires_gpu": False,
            "min_ram_gb": 2,
            "context_length": 2048,
        }
    }
    
    CURRENT_MODEL = "mistral"
    
    # Paramètres Mistral
    MISTRAL_GENERATION_CONFIG = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "early_stopping": True,
    }
    
    # Quantization
    USE_QUANTIZATION = True
    QUANTIZATION_BITS = 4
    
    def get_device(self):
        """Détecte le device optimal"""
        if torch.cuda.is_available():
            return f"cuda:0 (GPU: {torch.cuda.get_device_name(0)})"
        return "cpu"
    
    def has_gpu(self):
        """Vérifie si GPU disponible"""
        return torch.cuda.is_available()
    
    def is_groq_available(self) -> bool:
        """Vérifie si Groq API est configuré"""
        return bool(self.GROQ_CONFIG.get("api_key"))