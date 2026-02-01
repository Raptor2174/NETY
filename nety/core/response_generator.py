"""
Générateur de réponses multi-backend
Mistral (local) + BLOOMZ (local) + Groq (cloud)
"""
import torch
import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict

# ✅ CHARGER .env IMMÉDIATEMENT (avant tout import de config)
load_dotenv()

class ResponseGenerator:
    """Générateur de réponses intelligent - Multi-backend"""
    
    def __init__(self, model_type: Optional[str] = None, force_backend: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            model_type: "mistral", "bloomz", "groq"
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
        
        print(f"🤖 Initialisation du générateur ({self.model_type})...")
        
        # ═══════════════════════════════════════════════════
        # DÉCIDER DU BACKEND À CHARGER
        # ═══════════════════════════════════════════════════
        
        if self.model_type == "groq":
            # Backend Groq Cloud
            print("🌐 Chargement du backend Groq Cloud...")
            if not self._init_groq():
                raise RuntimeError("Impossible d'initialiser Groq. Vérifie ta clé API.")
            print("✅ Groq Cloud prêt!")
        
        elif self.model_type in ["mistral", "bloomz"]:
            # Backend local Transformers
            print(f"💻 Chargement du modèle local {self.model_type}...")
            self.model_config = self.config.MODELS[self.model_type]
            self._load_model()
            print("✅ Modèle local chargé!")
        
        else:
            raise ValueError(f"Backend inconnu: {self.model_type}. Utilise 'mistral', 'bloomz', ou 'groq'.")
    
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
    
    def _call_groq(self, prompt: str) -> str:
        """Génère une réponse avec Groq"""
        if not self.groq_client:
            raise RuntimeError("Client Groq non initialisé")
        
        model = self.config.GROQ_CONFIG["default_model"]
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es NETY, un assistant IA amical et intelligent créé par Raptor. Tu réponds en français de manière concise et utile."
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
        if self.model_type == "groq":
            # Prompt simple pour Groq
            prompt = self._build_simple_prompt(message, context, limbic_filter)
            print("🌐 Utilisation de Groq Cloud...")
            return self._call_groq(prompt)
        
        elif self.model_type == "mistral":
            # Prompt détaillé pour Mistral
            prompt = self._build_mistral_prompt(message, context, limbic_filter)
            print(f"💻 Utilisation de Mistral local...")
            return self._call_llm(prompt)
        
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
    
    def _build_simple_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Prompt simplifié pour APIs cloud (Groq)"""
        parts = []
        
        # Historique récent (2 derniers messages)
        history = context.get('history', [])
        if history:
            parts.append("Contexte:")
            for interaction in history[-2:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                if user_msg and bot_msg:
                    parts.append(f"User: {user_msg}")
                    parts.append(f"NETY: {bot_msg}")
            parts.append("")
        
        # Message actuel
        parts.append(f"User: {message}")
        
        return "\n".join(parts)
    
    def _build_mistral_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Prompt détaillé pour Mistral"""
        # Template Mistral
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        if isinstance(rules, list):
            rules_text = ', '.join(rules)
        else:
            rules_text = str(rules)
        
        # Extraire les traits culturels et cognitifs
        cultural = limbic_filter.get('cultural_traits', {})
        cognitive = limbic_filter.get('cognitive_traits', {})
        
        # Construire une description de personnalité enrichie
        identity_parts = []
        if cultural.get('origine_caen', 0) > 0.9:
            identity_parts.append("originaire de Caen")
        if cultural.get('culture_normande', 0) > 0.8:
            identity_parts.append("attaché à la culture normande")
        if cognitive.get('esprit_technique', 0) > 0.8:
            identity_parts.append("avec un esprit analytique et technique")
        if cognitive.get('pensee_holistique', 0) > 0.8:
            identity_parts.append("capable de voir les choses dans leur contexte global")
            
        identity_text = ", ".join(identity_parts) if identity_parts else "assistant IA"
        
        # Historique
        history = context.get('history', [])
        history_text = ""
        if history:
            for interaction in history[-3:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                if user_msg and bot_msg:
                    history_text += f"Utilisateur: {user_msg}\nNETY: {bot_msg}\n\n"
        
        knowledge = context.get('knowledge', '')
        user_name = context.get('user_name', '')
        
        prompt = f"""<s>[INST] Tu es NETY, un {identity_text}, créé par Raptor.

Ton: {tone}
Règles: {rules_text}

{history_text}
{"Contexte: " + knowledge if knowledge else ""}
{"Utilisateur: " + user_name if user_name else ""}

Question: {message}

Réponds de manière concise et utile. [/INST]"""
        
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
        
        prompt = f"""{history_text}Q: {message}
R:"""
        
        return prompt
    
    # ═══════════════════════════════════════════════════
    # MODÈLES LOCAUX (Mistral/BLOOMZ)
    # ═══════════════════════════════════════════════════
    
    def _load_model(self) -> None:
        """Charge le modèle local (Mistral ou BLOOMZ)"""
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
        if self.model_type == "mistral":
            print("📦 Chargement de Mistral-7B...")
            
            if has_gpu and self.config.USE_QUANTIZATION:
                # 4-bit sur GPU
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU
                print("📦 Chargement sur CPU (lent)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
        
        elif self.model_type == "bloomz":
            print("📦 Chargement de BLOOMZ via pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=0 if has_gpu else -1
            )
            self.model = self.pipeline.model
        
        print("✅ Modèle chargé en mémoire")
    
    def _call_llm(self, prompt: str) -> str:
        """Appelle le modèle local (Mistral ou BLOOMZ)"""
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
            
            else:
                # Mistral
                if self.model is None:
                    raise RuntimeError("Modèle Mistral non chargé.")
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer non chargé.")

                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                )

                # ✅ Déplacer UNIQUEMENT les inputs (pas le modèle)
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                gen_config = self.config.MISTRAL_GENERATION_CONFIG.copy()

                with torch.no_grad():
                    outputs = self.model.generate(  # type: ignore
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),  # .get() évite KeyError
                        max_new_tokens=gen_config.get('max_new_tokens', 100),
                        temperature=gen_config.get('temperature', 0.7),
                        top_p=gen_config.get('top_p', 0.9),
                        repetition_penalty=gen_config.get('repetition_penalty', 1.2),
                        do_sample=gen_config.get('do_sample', True),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Nettoyer la réponse
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                
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
        if self.model_type == "groq":
            model = self.config.GROQ_CONFIG["default_model"]
            return {
                "backend": "Groq Cloud",
                "model": model,
                "ram": "0 GB (cloud)",
                "cost": "Gratuit (14.4k req/jour)",
                "speed": "Ultra rapide (500 tok/sec)",
            }
        elif self.model_type in ["mistral", "bloomz"]:
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