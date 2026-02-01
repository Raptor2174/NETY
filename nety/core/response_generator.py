"""
Générateur de réponses multi-backend
BLOOMZ (local) + Groq (cloud) + RNN (local)
"""
import torch
import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict
from nety.core.rnn_response_generator import RNNResponseGenerator

# ✅ CHARGER .env IMMÉDIATEMENT (avant tout import de config)
load_dotenv()

class ResponseGenerator:
    """Générateur de réponses intelligent - Multi-backend"""
    
    def __init__(self, model_type: Optional[str] = None, force_backend: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            model_type: "bloomz", "groq", "rnn"
            force_backend: Force un backend spécifique
        """
        
        from .llm_config import LLMConfig
        self.config = LLMConfig()
        self.force_backend = force_backend
        self.model_type = model_type or "groq"  # Par défaut: groq
        
        # Attributs pour backends cloud
        self.groq_available = False
        self.groq_client = None
        
        # Attributs pour modèles locaux
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        self.model_config = None  # ⚠️ FIX: Peut être None pour cloud

        # ✨ AJOUT DU BACKEND RNN
        self.rnn_generator = None
        if self.model_type == "rnn":
            print("🧠 Initialisation du générateur RNN local...")
            self.rnn_generator = RNNResponseGenerator()
        
        print(f"🤖 Initialisation du générateur ({self.model_type})...")
        
        # ═══════════════════════════════════════════════════
        # DÉCIDER DU BACKEND À CHARGER
        # ═══════════════════════════════════════════════════
        
        if self.model_type == "rnn":
            # Backend RNN local
            print("✅ RNN local prêt!")

        elif self.model_type == "groq":
            # Backend Groq Cloud
            print("🌐 Chargement du backend Groq Cloud...")
            if not self._init_groq():
                raise RuntimeError("Impossible d'initialiser Groq. Vérifie ta clé API.")
            print("✅ Groq Cloud prêt!")
        
        elif self.model_type == "bloomz":
            # Backend local Transformers
            print(f"💻 Chargement du modèle local {self.model_type}...")
            self.model_config = self.config.MODELS[self.model_type]
            self._load_model()
            print("✅ Modèle local chargé!")
        
        else:
            raise ValueError(f"Backend inconnu: {self.model_type}. Utilise 'bloomz', 'groq', ou 'rnn'.")
    
    # ═══════════════════════════════════════════════════
    # GROQ BACKEND
    # ═══════════════════════════════════════════════════
    
    def _init_groq(self) -> bool:
        """Initialise le client Groq"""
        if not self.config.is_groq_available():
            print("❌ Clé API Groq manquante!")
            return False
        
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.config.GROQ_CONFIG["api_key"])
            
            # Test rapide de connexion
            print("🔍 Vérification de l'API Groq...")
            test = self.groq_client.chat.completions.create(
                model=self.config.GROQ_CONFIG["default_model"],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            self.groq_available = True
            print(f"✅ Connexion Groq réussie (modèle: {self.config.GROQ_CONFIG['default_model']})")
            return True
        
        except ImportError:
            print("❌ Module 'groq' non installé!")
            print("💡 Installe-le avec: pip install groq")
            return False
        
        except Exception as e:
            print(f"❌ Erreur Groq: {e}")
            print("💡 Vérifie ta clé API et ta connexion internet")
            return False
    
    def _call_groq(self, prompt: str, limbic_filter: Optional[Dict] = None) -> str:
        """Génère une réponse avec Groq"""
        if not self.groq_client:
            raise RuntimeError("Client Groq non initialisé")
        
        model = self.config.GROQ_CONFIG["default_model"]
        
        try:
            system_msg = "Tu es NETY, un assistant IA amical et intelligent créé par Raptor."
            if limbic_filter:
                emotional_state = limbic_filter.get('emotional_state', {})
                if emotional_state:
                    state = emotional_state.get('state', '')
                    if state:
                        system_msg += f"\n\nÉtat émotionnel actuel: {state}."
                        system_msg += "\nTu es conscient de tes émotions et tu peux les partager si on te le demande."

            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_msg
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.GROQ_CONFIG["models"][model]["max_tokens"],
                temperature=self.config.GROQ_CONFIG["temperature"]
            )
            
            content = response.choices[0].message.content
            return content.strip() if content else ""
        
        except Exception as e:
            print(f"❌ Erreur lors de l'appel Groq: {e}")
            return "Désolé, je ne peux pas répondre pour le moment (erreur Groq Cloud)."
    
    # ═══════════════════════════════════════════════════
    # MÉTHODE PRINCIPALE : GENERATE
    # ═══════════════════════════════════════════════════
    
    def generate(self, message: str, context: Optional[Dict] = None, 
                 limbic_filter: Optional[Dict] = None) -> str:
        """
        Génère une réponse intelligente
        
        Args:
            message: Message de l'utilisateur
            context: Contexte (historique, connaissances, etc.)
            limbic_filter: Filtres émotionnels et comportementaux
        """
        if context is None:
            context = {}
        if limbic_filter is None:
            limbic_filter = {'tone': 'friendly', 'behavior_rules': []}
        
        # Construire le prompt
        if self.model_type == "rnn":
            # ✨ AJOUTER le message au contexte
            context["current_message"] = message
            context["limbic_filter"] = limbic_filter
            print("🧠 Utilisation du RNN local...")
            if self.rnn_generator is None:
                raise RuntimeError("Générateur RNN non initialisé")
            return self.rnn_generator.generate(message, context, limbic_filter)

        elif self.model_type == "groq":
            # Prompt simple pour Groq
            prompt = self._build_simple_prompt(message, context, limbic_filter)
            print("🌐 Utilisation de Groq Cloud...")
            return self._call_groq(prompt, limbic_filter)
        
        elif self.model_type == "bloomz":
            # Prompt simple pour BLOOMZ
            prompt = self._build_bloomz_prompt(message, context, limbic_filter)
            print(f"💻 Utilisation de BLOOMZ local...")
            return self._call_llm(prompt)
        
        else:
            return "Erreur: Backend inconnu."
    
    # ═══════════════════════════════════════════════════
    # CONSTRUCTION DES PROMPTS
    # ═══════════════════════════════════════════════════

    def _format_personal_memory(self, context: Dict) -> str:
        memories = context.get("personal_memory", [])
        profile = context.get("user_profile", {})
        parts = []

        if profile:
            profile_items = [f"{k}: {v}" for k, v in profile.items()]
            parts.append("Profil utilisateur: " + ", ".join(profile_items))

        if memories:
            parts.append("Souvenirs pertinents:")
            for mem in memories[:3]:
                text = mem.get("text", "")
                if text:
                    parts.append(f"- {text}")

        return "\n".join(parts)
    
    def _build_simple_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Prompt adaptatif intelligent"""
        from nety.core.smart_prompt_builder import SmartPromptBuilder
        
        builder = SmartPromptBuilder()
        prompt, estimated_tokens = builder.build_prompt(
            message, context, limbic_filter, level="auto"
        )
        
        # Log pour monitoring
        print(f"📊 Prompt niveau: {builder.detect_prompt_level(message)} (~{estimated_tokens} tokens)")
        
        return prompt
    
    def _build_bloomz_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Prompt simple pour BLOOMZ"""
        history = context.get('history', [])
        history_text = ""
        
        if history:
            for interaction in history[-2:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                if user_msg and bot_msg:
                    history_text += f"Q: {user_msg}\nR: {bot_msg}\n\n"
        
        memory_block = self._format_personal_memory(context)
        memory_prefix = f"{memory_block}\n" if memory_block else ""

        prompt = f"""{memory_prefix}{history_text}Q: {message}
R:"""
        
        return prompt
    
    # ═══════════════════════════════════════════════════
    # MODÈLES LOCAUX (BLOOMZ)
    # ═══════════════════════════════════════════════════
    
    def _load_model(self) -> None:
        """Charge le modèle local (BLOOMZ)"""
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM,
            pipeline,
            BitsAndBytesConfig
        )
        
        if self.model_config is None:
            raise RuntimeError("model_config is None. Cannot load local model.")
        
        model_name = self.model_config['name']
        has_gpu = torch.cuda.is_available()
        
        print(f"🖥️ GPU détecté: {'Oui' if has_gpu else 'Non'}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Charger le modèle
        if self.model_type == "bloomz":
            print("📦 Chargement de BLOOMZ via pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=0 if has_gpu else -1
            )
            self.model = self.pipeline.model
        
        print("✅ Modèle chargé en mémoire")
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le modèle local (BLOOMZ)"""
        try:
            if self.model_type == "bloomz":
                if self.pipeline is None:
                    raise RuntimeError("Pipeline BLOOMZ non chargé")
                
                result = self.pipeline(
                    prompt,
                    max_new_tokens=120,
                    temperature=0.6,
                    do_sample=True,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3
                )
                
                full_text = result[0]['generated_text']
                response = full_text[len(prompt):].strip()
                return response
        
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            import traceback
            traceback.print_exc()
            return "Désolé, je ne peux pas répondre pour le moment (erreur interne)."
    
    # ═══════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════
    
    def get_model_info(self) -> dict:
        """Retourne les informations du modèle actuel"""
        if self.model_type == "rnn":
            if self.rnn_generator is None:
                return {"backend": "RNN local", "model": "Inconnu"}
            return self.rnn_generator.get_model_info()
        elif self.model_type == "groq":
            model = self.config.GROQ_CONFIG["default_model"]
            return {
                "backend": "Groq Cloud",
                "model": model,
                "ram": "0 GB (cloud)",
                "cost": "Gratuit (14.4k req/jour)",
                "speed": "Ultra rapide (500 tok/sec)",
            }
        elif self.model_type == "bloomz":
            model_name = self.model_config['name'] if self.model_config else "Inconnu"
            ram = f"{self.model_config['min_ram_gb']} GB" if self.model_config and 'min_ram_gb' in self.model_config else "Inconnu"
            return {
                "backend": "Transformers (local)",
                "model": model_name,
                "ram": ram,
                "cost": "Gratuit",
                "speed": "Dépend du matériel",
            }
        else:
            return {"backend": "Inconnu"}