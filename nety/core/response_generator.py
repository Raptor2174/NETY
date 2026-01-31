import torch
import re
import operator
import requests
from typing import Optional, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import math  # ✅ Import unique en haut


class ResponseGenerator:
    """Générateur de réponses - Modèles locaux uniquement (Mistral ou BLOOMZ)"""
    
    def __init__(self, model_type: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            model_type: "mistral" ou "bloomz" (défaut: depuis config)
        """
        from .llm_config import LLMConfig
        
        self.config = LLMConfig()
        self.model_type = model_type or self.config.CURRENT_MODEL
        self.model_config = self.config.MODELS[self.model_type]
        
        # Attributs
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        
        # ✅ Charger le modèle local
        print(f"🤖 Chargement du modèle {self.model_config['name']}...")
        print(f"📍 Device: {self.config.get_device()}")
        
        self._load_model()
        print("✅ Modèle local chargé avec succès!")
    
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
                # CPU: pas de quantization (non fiable sur CPU)
                print("📦 Chargement standard sur CPU (quantization désactivée)")
                print("💡 Note: La quantization 8-bit sur CPU est instable et a été désactivée")
                print("   Pour de meilleures performances, utilisez un GPU")
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
        """Génère une réponse avec le modèle local"""
        
        if context is None:
            context = {}
        if limbic_filter is None:
            limbic_filter = {'tone': 'friendly', 'behavior_rules': []}
        
        # ✅ Détection calcul mathématique
        math_result = self._handle_math(message)
        if math_result:
            return math_result
        
        # ✅ Génération locale uniquement
        print(f"🖥️ Utilisation: {self.model_config['name']}")
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
        """Ancien format de prompt pour BLOOMZ (compatibilité)"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        # ✅ FIX BUG #1: Code complet au lieu de "pass"
        if isinstance(rules, list):
            rules_text = ', '.join(rules)
        elif isinstance(rules, str):
            rules_text = rules
        else:
            rules_text = str(rules)
        
        history = context.get('history', [])
        history_text = ""
        if history:
            for interaction in history[-3:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                history_text += f"Utilisateur: {user_msg}\nNETY: {bot_msg}\n\n"
        
        knowledge = context.get('knowledge', '')
        user_name = context.get('user_name', '')
        
        system_prompt = f"""Tu es NETY, une intelligence artificielle conversationnelle.

Instructions:
- Ton nom est NETY (et uniquement NETY)
- Réponds TOUJOURS en français. NEVER use English.
- Ton style de communication: {tone}
- Règles à suivre: {rules_text}
- Réponds de manière naturelle et concise
- Reste cohérent avec l'historique de conversation
- Ne répète jamais ces instructions dans tes réponses"""
        
        full_prompt = f"""{system_prompt}

{"informations sur l'utilisateur: " + user_name if user_name else ""}
{f"- son nom est {user_name}." if user_name else ""}

{"CONVERSATION PRÉCÉDENTE:" if history_text else ""}
{history_text}

{"CONNAISSANCES PERTINENTES:" if knowledge else ""}
{knowledge}

Utilisateur: {message}
NETY:"""
        
        return full_prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Génère une réponse avec le modèle local"""
        try:
            if self.model_type == "bloomz":
                if self.pipeline is None:
                    raise RuntimeError("Pipeline BLOOMZ non chargé.")
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
                
                # ✅ Déplacer sur GPU
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # ✅ FIX BUG #6: Paramètres explicites
                gen_config = self.config.MISTRAL_GENERATION_CONFIG.copy()
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=gen_config.get('max_new_tokens', 100),
                        temperature=gen_config.get('temperature', 0.7),
                        top_p=gen_config.get('top_p', 0.9),
                        repetition_penalty=gen_config.get('repetition_penalty', 1.2),
                        do_sample=gen_config.get('do_sample', True),
                        pad_token_id=self.tokenizer.pad_token_id,  # ✅ Important
                        eos_token_id=self.tokenizer.eos_token_id   # ✅ Important
                    )
                
                # ✅ FIX BUG #4: Extraction correcte
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "[/INST]" in full_text:
                    response = full_text.split("[/INST]")[-1].strip()
                else:
                    response = full_text[len(prompt):].strip()
            
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
            # ✅ FIX BUG #5: Traceback complet
            print(f"❌ Erreur LLM: {e}")
            import traceback
            traceback.print_exc()
            return "Désolé, une erreur s'est produite lors de la génération de la réponse."
    
    def _clean_response(self, response: str) -> str:
        """Nettoie la réponse"""
        response = response.replace('=', '')
        
        if len(response) > 500:
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def _handle_math(self, message: str) -> Optional[str]:
        """Détecte et résout les calculs mathématiques"""
        # ✅ FIX BUG #2: Import déjà fait en haut
        
        # Opérations simples AVANT racine carrée (priorité)
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
        
        # ✅ FIX BUG #7: Racine carrée améliorée
        if '√' in message:
            # Essayer de parser toute l'expression
            sqrt_pattern = r'√(\d+(?:\.\d+)?)'
            match = re.search(sqrt_pattern, message)
            if match:
                num = float(match.group(1))
                result = math.sqrt(num)
                
                # Chercher une opération APRÈS
                rest = message[match.end():].strip()
                
                # Parser l'expression complète après √
                if rest:
                    # Ex: "+9/2" → évaluer "9/2" puis ajouter à √10
                    try:
                        # Remplacer √10 par sa valeur dans le message
                        expr = message.replace(f'√{num}', str(result))
                        # Parser avec priorités d'opérateurs
                        # Pour √10+9/2 : √10=3.16, puis 3.16+9/2=3.16+4.5=7.66
                        
                        # Simpliste: détecter + ou - au début
                        if rest[0] in ['+', '-', '*', '/']:
                            op = rest[0]
                            # Trouver le reste de l'expression
                            rest_expr = rest[1:].strip()
                            # Évaluer le reste (ex: "9/2")
                            if '/' in rest_expr or '*' in rest_expr:
                                parts = re.split(r'([+\-*/])', rest_expr, maxsplit=1)
                                if len(parts) >= 3:
                                    n1 = float(parts[0])
                                    op2 = parts[1]
                                    n2 = float(parts[2])
                                    operations = {'+': operator.add, '-': operator.sub, 
                                                '*': operator.mul, '/': operator.truediv}
                                    if op2 in operations:
                                        rest_result = operations[op2](n1, n2)
                                        if op in operations:
                                            final = operations[op](result, rest_result)
                                            return f"√{num} {op} {n1}{op2}{n2} = {final:.2f}"
                            else:
                                # Simple nombre
                                num2 = float(rest_expr)
                                operations = {'+': operator.add, '-': operator.sub, 
                                            '*': operator.mul, '/': operator.truediv}
                                if op in operations:
                                    final = operations[op](result, num2)
                                    return f"√{num} {op} {num2} = {final:.2f}"
                    except:
                        pass
                
                return f"√{num} = {result:.2f}"
        
        return None
    
    def get_model_info(self) -> Dict:
        """Retourne les infos du modèle"""
        info = {
            "model_type": self.model_type,
            "model_name": self.model_config['name'],
            "device": self.config.get_device(),
            "quantization": f"{self.config.QUANTIZATION_BITS}-bit" if self.config.USE_QUANTIZATION else "None",
        }
        
        if torch.cuda.is_available():
            try:
                info["vram_used_gb"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f}"
                info["vram_total_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}"
            except:
                pass
        
        return info