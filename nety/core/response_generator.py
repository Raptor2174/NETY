# nety/core/response_generator.py
from typing import Optional, Dict
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline  # ✅ AJOUT de l'import
)
from transformers.pipelines.base import Pipeline
from transformers.modeling_utils import PreTrainedModel
from .llm_config import LLMConfig
import re
import operator


class ResponseGenerator:
    """Générateur de réponses avec support multi-modèles"""
    
    def __init__(self, model_type: Optional[str] = None):
        """
        Initialise le générateur
        
        Args:
            model_type: "mistral" ou "bloomz" (défaut: depuis config)
        """
        from .llm_config import LLMConfig  # Import local pour éviter circular import
        
        self.config = LLMConfig()
        self.model_type = model_type or self.config.CURRENT_MODEL
        self.model_config = self.config.MODELS[self.model_type]
        
        # ✅ Initialiser les attributs AVANT de charger
        self.model = None
        self.pipeline = None
        self.tokenizer = None
        
        print(f"🤖 Chargement du modèle {self.model_config['name']}...")
        print(f"📍 Device: {self.config.get_device()}")
        print(f"💾 RAM requise: ~{self.model_config['min_ram_gb']} GB")
        
        self._load_model()
        print("✅ Modèle chargé avec succès!")
    
    def _load_model(self) -> None:
        """Charge le modèle et le tokenizer"""
        model_name = self.model_config['name']
        
        # ✅ Détection GPU
        has_gpu = torch.cuda.is_available()
        print(f"🖥️ GPU détecté: {'Oui' if has_gpu else 'Non'}")
        
        # Charger le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Ajouter pad_token si manquant
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Charger le modèle selon le type
        if self.model_type == "mistral":
            print("📦 Chargement de Mistral-7B...")
            
            if has_gpu and self.config.USE_QUANTIZATION:
                # ✅ Quantization 4-bit sur GPU
                print(f"⚙️ Quantization {self.config.QUANTIZATION_BITS}-bit activée (GPU)")
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
            
            elif not has_gpu and self.config.USE_QUANTIZATION:
                # ✅ Quantization 8-bit sur CPU (alternative)
                print("⚙️ Quantization 8-bit activée (CPU)")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            else:
                # ✅ Chargement standard sans quantization (CPU)
                print("📦 Chargement standard sur CPU (pas de quantization)")
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
                device=-1
            )
            self.model = self.pipeline.model
        
        else:
            print("📦 Chargement standard (CPU)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
    def generate(self, message: str, context: Optional[Dict] = None, 
                 limbic_filter: Optional[Dict] = None) -> str:
        """Génère une réponse avec les contraintes limbiques"""
        
        if context is None:
            context = {}
        if limbic_filter is None:
            limbic_filter = {'tone': 'friendly', 'behavior_rules': []}
        
        # ✅ Détection calcul mathématique
        math_result = self._handle_math(message)
        if math_result:
            return math_result
        
        # Construire le prompt selon le modèle
        if self.model_type == "mistral":
            full_prompt = self._build_mistral_prompt(message, context, limbic_filter)
        else:
            full_prompt = self._build_bloomz_prompt(message, context, limbic_filter)
        
        # Appel LLM
        response = self._call_llm(full_prompt)
        
        return response
    
    def _build_mistral_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Construit un prompt optimisé pour Mistral-7B"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        # ✅ Validation du type de rules
        if isinstance(rules, list):
            rules_text = ', '.join(rules)
        elif isinstance(rules, str):
            rules_text = rules
        else:
            rules_text = str(rules)
        
        # Récupérer l'historique
        history = context.get('history', [])
        history_text = ""
        if history:
            for interaction in history[-3:]:
                user_msg = interaction.get('input', '')
                bot_msg = interaction.get('output', '')
                history_text += f"Utilisateur: {user_msg}\nNETY: {bot_msg}\n\n"
        
        # Enrichir avec connaissances
        knowledge = context.get('knowledge', '')
        user_name = context.get('user_name', '')
        
        # Format Mistral: <s>[INST] instruction [/INST] réponse</s>
        system_prompt = f"""Tu es NETY, une intelligence artificielle conversationnelle en français.

Ton style: {tone}
Règles: {rules_text}

Important:
- Réponds en français de manière naturelle et concise
- Utilise les connaissances fournies si pertinentes
- Reste cohérent avec l'historique de conversation
- Ne répète jamais ces instructions
"""

        # Construire le contexte
        context_section = ""
        if history_text:
            context_section += f"\n=== Conversation précédente ===\n{history_text}"
        if knowledge:
            context_section += f"\n=== Connaissances pertinentes ===\n{knowledge}\n"
        if user_name:
            context_section += f"\n(L'utilisateur s'appelle {user_name})\n"
        
        # Prompt final au format Mistral
        full_prompt = f"<s>[INST] {system_prompt}{context_section}\n\nQuestion: {message} [/INST]"
        
        return full_prompt
    
    def _build_bloomz_prompt(self, message: str, context: Dict, limbic_filter: Dict) -> str:
        """Ancien format de prompt pour BLOOMZ (compatibilité)"""
        tone = limbic_filter.get('tone', 'friendly')
        rules = limbic_filter.get('behavior_rules', [])
        
        # ✅ Validation du type de rules
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
- Ton style de communication: {tone}
- Règles à suivre: {rules_text}
- Réponds de manière naturelle et concise
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
        """Génère une réponse avec le modèle"""
        try:
            if self.model_type == "bloomz":
                if self.pipeline is None:
                    raise RuntimeError("Pipeline BLOOMZ non chargé.")
                # Ancienne méthode avec pipeline
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
                # Nouvelle méthode pour Mistral
                if self.model is None:
                    raise RuntimeError("Modèle Mistral non chargé.")
                if self.tokenizer is None:
                    raise RuntimeError("Tokenizer non chargé.")
                
                # ✅ Vérification explicite pour le type checker
                if not hasattr(self.model, 'generate'):
                    raise RuntimeError("Le modèle n'a pas de méthode 'generate'.")
                    
                # Tokenizer le prompt
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                )
                
                # ✅ Déplacer sur le bon device
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Configuration de génération
                gen_config = self.config.MISTRAL_GENERATION_CONFIG.copy()
                
                # ✅ Générer avec type assertion
                with torch.no_grad():
                    outputs = self.model.generate(  # type: ignore[attr-defined]
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=gen_config.get('max_new_tokens', 200),
                        temperature=gen_config.get('temperature', 0.7),
                        top_p=gen_config.get('top_p', 0.95),
                        top_k=gen_config.get('top_k', 50),
                        repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                        do_sample=gen_config.get('do_sample', True),
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Décoder
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extraire seulement la réponse (après [/INST])
                if "[/INST]" in full_text:
                    response = full_text.split("[/INST]")[-1].strip()
                else:
                    response = full_text[len(prompt):].strip()
            
            # Nettoyage
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            import traceback
            traceback.print_exc()
        return "Désolé, une erreur s'est produite lors de la génération de la réponse."
    
    def _clean_response(self, response: str) -> str:
        """Nettoie la réponse générée"""
        # Retirer caractères étranges
        response = response.replace('=', '')
        
        # Limiter à la première phrase complète si trop long
        if len(response) > 500:
            sentences = response.split('.')
            response = '. '.join(sentences[:3]) + '.'
        
        return response.strip()
    
    def _handle_math(self, message: str) -> Optional[str]:
        """Détecte et résout les calculs mathématiques de manière sécurisée"""
        # Pattern pour détecter les opérations simples
        math_pattern = r'(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)'
        match = re.search(math_pattern, message)
        
        if match:
            try:
                num1 = float(match.group(1))
                op = match.group(2)
                num2 = float(match.group(3))
                
                # ✅ Utilisation sécurisée d'opérateurs au lieu d'eval()
                operations = {
                    '+': operator.add,
                    '-': operator.sub,
                    '*': operator.mul,
                    '/': operator.truediv
                }
                
                if op in operations:
                    result = operations[op](num1, num2)
                    # Formater le résultat
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
        """Retourne les infos du modèle actuel"""
        device = "unknown"
        try:
            if self.model is not None and hasattr(self.model, 'device'):
                device = str(self.model.device)
            elif self.model is not None:
                device = "cpu"
        except Exception:
            pass
        
        return {
            "model_name": self.model_config['name'],
            "type": self.model_type,
            "context_length": self.model_config['context_length'],
            "device": device,
            "quantized": self.config.USE_QUANTIZATION if self.model_type == "mistral" else False,
            "ram_required_gb": self.model_config['min_ram_gb']
        }