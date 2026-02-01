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
        
        # Modèles disponibles (gratuits) - Mis à jour 2026
        "models": {
            "llama-3.3-70b-versatile": {
                "context_window": 128000,
                "max_tokens": 150,
                "speed": "ultra",  # ~500 tok/sec
                "quality": 5,
                "description": "⭐ RECOMMANDÉ - Meilleur modèle Groq (Jan 2026)"
            },
            "llama-3.1-8b-instant": {
                "context_window": 128000,
                "max_tokens": 150,
                "speed": "ultra-fast",
                "quality": 4,
                "description": "Très rapide et léger"
            },
            "mixtral-8x7b-32768": {
                "context_window": 32768,
                "max_tokens": 150,
                "speed": "fast",
                "quality": 5,
                "description": "Excellent pour le français (47B params)"
            },
            "gemma2-9b-it": {
                "context_window": 8192,
                "max_tokens": 150,
                "speed": "fast",
                "quality": 4,
                "description": "Efficace et rapide (Google)"
            },
        },
        
        "default_model": "llama-3.3-70b-versatile",  # ✅ CORRIGÉ
        "temperature": 0.7,
        
        # Limites gratuites
        "rate_limits": {
            "requests_per_minute": 30,
            "requests_per_day": 14400,
            "tokens_per_minute": 6000,
        }
    }
    
    # ═══════════════════════════════════════════════════
    # BLOOMZ LOCAL (OPTION 1)
    # ═══════════════════════════════════════════════════
    
    MODELS = {
        "bloomz": {
            "name": "bigscience/bloomz-560m",
            "type": "bloom",
            "requires_gpu": False,
            "min_ram_gb": 2,
            "context_length": 2048,
        }
    }

    # ═══════════════════════════════════════════════════
    # RNN LOCAL (OPTION 4) - NOUVEAU !
    # ═══════════════════════════════════════════════════

    RNN_CONFIG = {
        "enabled": True,
        "name": "RNN Local - TextualCortex",
        "type": "LOCAL",
        "description": "🧪 EXPÉRIMENTAL - RNN bi-directionnel entraîné localement",
        "speed": "Rapide",
        "quality": "En apprentissage",
        "ram": "500 MB",
        "device": "CPU/GPU",
        "vocab_size": 2000,
        "max_length": 50,
        "temperature": 0.8,
    }

    # Modèles disponibles
    AVAILABLE_MODELS = {
        "bloomz": MODELS["bloomz"],
        "groq": GROQ_CONFIG,
        "rnn": RNN_CONFIG,  # ✨ RNN local
    }
    
    CURRENT_MODEL = "bloomz"
    
    # Paramètres BLOOMZ
    BLOOMZ_GENERATION_CONFIG = {
        "max_new_tokens": 120,
        "temperature": 0.6,
        "do_sample": True,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
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