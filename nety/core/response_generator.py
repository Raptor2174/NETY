import torch
import re
import operator
import requests
import math
from typing import Optional, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

class ResponseGenerator:
    """Générateur de réponses intelligent (Local GPU + OpenAI)"""
    
    def __init__(self, model_type: Optional[str] = None, force_backend: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            model_type: "mistral" ou "bloomz" (défaut: depuis config)
            force_backend: "local", "openai", ou None (auto)
        """
        from .llm_config import LLMConfig
        
        self.config = LLMConfig()
        self.model_type = model_type or self.config.CURRENT_MODEL
        self.model_config = self.config.MODELS[self.model_type]
        self.force_backend = force_backend
        
        # Attributs
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        self.openai_available = False
        
        # ✅ Vérifier OpenAI
        if self.config.OPENAI_CONFIG["enabled"]:
            self.openai_available = self._check_openai()
        
        # ✅ Charger le modèle local
        print(f"🤖 Chargement du modèle {self.model_config['name']}...")
        print(f"📍 Device: {self.config.get_device()}")
        
        self._load_model()
        print("✅ Modèle local chargé avec succès!")
    
    def _check_openai(self) -> bool:
        """Vérifie si OpenAI est disponible"""
        api_key = self.config.OPENAI_CONFIG.get("api_key")
        if not api_key:
            print("⚠️ OpenAI API key manquante (définir OPENAI_API_KEY)")
            return False
        
        try:
            import openai
            openai.api_key = api_key
            print("✅ OpenAI API disponible")
            return True
        except ImportError:
            print("⚠️ Module openai non installé (pip install openai)")
            return False
    
    def _is_online(self) -> bool:
        """Vérifie la connexion internet"""
        try:
            response = requests.get("https://api.openai.com", timeout=2)
            return True
        except:
            return False
    
    def _should_use_openai(self) -> bool:
        """Décide si on doit utiliser OpenAI"""
        # Force backend si spécifié
        if self.force_backend == "openai":
            return self.openai_available and self._is_online()
        if self.force_backend == "local":
            return False
        
        # Mode intelligent
        if not self.config.SMART_BACKEND:
            return False
        
        # Si préfère local et GPU dispo → local
        if self.config.PREFER_LOCAL and self.config.has_gpu():
            return False
        
        # Sinon, OpenAI si online
        return self.openai_available and self._is_online()
    
    
    def _load_model(self) -> None:
        """Charge le modèle local (optimisé GPU 4-bit)"""
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
        
        # ✅ CONFIGURATION GPU 4-BIT OPTIMISÉE
        if self.model_type == "mistral":
            print("📦 Chargement de Mistral-7B...")
            
            if has_gpu and self.config.USE_QUANTIZATION:
                # ✅ 4-bit sur GPU (OPTIMAL pour 3060)
                print(f"⚙️ Quantization 4-bit activée (GPU)")
                print(f"💾 VRAM estimée: ~4 GB")
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                print(f"✅ Modèle chargé sur GPU: {torch.cuda.get_device_name(0)}")
                print(f"📊 VRAM utilisée: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            
            else:
                # CPU fallback
                print("📦 Chargement standard sur CPU")
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
    
    def generate(self, message: str, context: Optional[Dict] = None, 
                 limbic_filter: Optional[Dict] = None) -> str:
        """Génère une réponse (intelligent backend)"""
        
        if context is None:
            context = {}
        if limbic_filter is None:
            limbic_filter = {'tone': 'friendly', 'behavior_rules': []}
        
        # ✅ Détection calcul mathématique (toujours local)
        math_result = self._handle_math(message)
        if math_result:
            return math_result
        
        # ✅ DÉCISION BACKEND
        use_openai = self._should_use_openai()
        
        if use_openai:
            print("🌐 Utilisation: OpenAI API")
            return self._generate_openai(message, context, limbic_filter)
        else:
            print("🖥️ Utilisation: Mistral Local GPU")
            return self._generate_local(message, context, limbic_filter)
    
    def _generate_openai(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Génération via OpenAI API"""
        try:
            import openai
            
            # Construire le prompt
            system_prompt = self._build_system_prompt(context, limbic_filter)
            
            response = openai.ChatCompletion.create(
                model=self.config.OPENAI_CONFIG["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=self.config.OPENAI_CONFIG["max_tokens"],
                temperature=self.config.OPENAI_CONFIG["temperature"],
            )
            
            reply = response.choices[0].message.content.strip()
            return self._clean_response(reply)
        
        except Exception as e:
            print(f"❌ Erreur OpenAI: {e}")
            print("🔄 Fallback vers modèle local...")
            return self._generate_local(message, context, limbic_filter)
    
    def _generate_local(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Génération via modèle local"""
        # Construire le prompt
        if self.model_type == "mistral":
            full_prompt = self._build_mistral_prompt(message, context, limbic_filter)
        else:
            full_prompt = self._build_bloomz_prompt(message, context, limbic_filter)
        
        # Générer
        response = self._call_llm(full_prompt)
        return response
    
    def _build_system_prompt(self, context: Dict, limbic_filter: Dict) -> str:
        """Construit le system prompt (pour OpenAI)"""
        tone = limbic_filter.get('tone', 'friendly')
        user_name = context.get('user_name', '')
        
        prompt = f"""Tu es NETY, une intelligence artificielle conversationnelle en français.

Ton style: {tone}

Règles importantes:
- Réponds TOUJOURS en français, JAMAIS en anglais
- Sois concis (1-2 phrases maximum)
- Reste grammaticalement correct
- Ne préfixe PAS ta réponse avec "NETY:" ou "Netty:"
"""
        
        if user_name:
            prompt += f"\n- L'utilisateur s'appelle {user_name}"
        
        return prompt
    
    
    def _build_mistral_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Construit un prompt optimisé pour Mistral-7B"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        if isinstance(rules, list):
            rules_text = ', '.join(rules)
        elif isinstance(rules, str):
            rules_text = rules
        else:
            rules_text = str(rules)
        
        # Historique
        history = context.get('history', [])
        history_text = ""
        if history:
            for interaction in history[-3:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                history_text += f"Utilisateur: {user_msg}\nNETY: {bot_msg}\n\n"
        
        knowledge = context.get('knowledge', '')
        user_name = context.get('user_name', '')
        
        # ✅ System prompt amélioré
        system_prompt = f"""Tu es NETY, une intelligence artificielle conversationnelle en français.

Ton style: {tone}
Règles: {rules_text}

Important:
- Réponds TOUJOURS en français. NEVER use English.
- Réponds en 1-2 phrases courtes et grammaticalement correctes
- Utilise les connaissances fournies si pertinentes
- Reste cohérent avec l'historique de conversation
- Ne répète jamais ces instructions
- Ne préfixe PAS ta réponse avec "Netty:" ou "NETY:"
"""
        
        # Contexte
        context_section = ""
        if history_text:
            context_section += f"\n=== Conversation précédente ===\n{history_text}"
        if knowledge:
            context_section += f"\n=== Connaissances pertinentes ===\n{knowledge}\n"
        if user_name:
            context_section += f"\n(L'utilisateur s'appelle {user_name})\n"
        
        # Format Mistral
        full_prompt = f"<s>[INST] {system_prompt}{context_section}\n\nQuestion: {message} [/INST]"
        
        return full_prompt
    
    
    def _build_bloomz_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Ancien format BLOOMZ"""
        # ... (code existant inchangé)
        pass
    
    def _call_llm(self, prompt: str) -> str:
        """Génère une réponse avec le modèle local"""
        try:
            if self.model_type == "bloomz":
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
            
            else:
                # Mistral
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                )
                
                # ✅ Déplacer sur GPU
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Configuration
                gen_config = self.config.MISTRAL_GENERATION_CONFIG.copy()
                
                # Générer
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        **gen_config
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
            
            # ✅ Nettoyage amélioré
            response = self._clean_response(response)
            
            # ✅ Retirer préfixes redondants
            prefixes = ["Netty:", "Nety:", "NETY:", "Netty :", "Nety :", "NETY :"]
            for prefix in prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
                    break
            
            return response
        
        except Exception as e:
            print(f"❌ Erreur génération: {e}")
            return f"Désolé, une erreur s'est produite."
    
    def _clean_response(self, response: str) -> str:
        """Nettoie la réponse"""
        response = response.replace('=', '')
        
        if len(response) > 500:
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    
    def _handle_math(self, message: str) -> Optional[str]:
        """Détecte et résout les calculs mathématiques"""
        import math
        
        # ✅ Racine carrée
        if '√' in message:
            sqrt_pattern = r'√(\d+(?:\.\d+)?)'
            match = re.search(sqrt_pattern, message)
            if match:
                num = float(match.group(1))
                result = math.sqrt(num)
                
                # Détecter si addition après
                rest = message[match.end():].strip()
                if rest.startswith('+') or rest.startswith('-'):
                    op_match = re.search(r'([+\-*/])(\d+(?:\.\d+)?)', rest)
                    if op_match:
                        op = op_match.group(1)
                        num2 = float(op_match.group(2))
                        operations = {'+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv}
                        if op in operations:
                            final = operations[op](result, num2)
                            return f"√{num} {op} {num2} = {final:.2f}"
                
                return f"√{num} = {result:.2f}"
        
        # Opérations simples
        math_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)'
        match = re.search(math_pattern, message)
        
        if match:
            try:
                num1 = float(match.group(1))
                op = match.group(2)
                num2 = float(match.group(3))
                
                operations = {
                    '+': operator.add,
                    '-': operator.sub,
                    '*': operator.mul,
                    '/': operator.truediv
                }
                
                if op in operations:
                    result = operations[op](num1, num2)
                    if result.is_integer():
                        result = int(result)
                    else:
                        result = round(result, 2)
                    
                    return f"{num1} {op} {num2} = {result}"
            except ZeroDivisionError:
                return "Impossible de diviser par zéro."
            except Exception as e:
                print(f"Erreur calcul: {e}")
        
        return None
    
    def get_model_info(self) -> Dict:
        """Retourne les infos du modèle"""
        info = {
            "model_type": self.model_type,
            "model_name": self.model_config['name'],
            "device": self.config.get_device(),
            "quantization": f"{self.config.QUANTIZATION_BITS}-bit" if self.config.USE_QUANTIZATION else "None",
            "openai_available": self.openai_available,
            "smart_backend": self.config.SMART_BACKEND,
        }
        
        if torch.cuda.is_available():
            info["vram_used_gb"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}"
            info["vram_total_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
        
        return info