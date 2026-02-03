"""
Module Brain - Cerveau central de NETY
"""
# nety/core/brain.py
from typing import Optional
import torch

# Imports locaux
from nety.cortex_limbic.emotion_engine import EmotionEngine
from nety.cortex_limbic.limbic_filter import LimbicFilter
from nety.cortex_limbic.memory_manager import MemoryManager
# Import lazy pour TextualCortex (évite les imports circulaires)
from nety.knowledge_base.knowledge_manager import KnowledgeManager
from nety.core.intent_analyzer import IntentAnalyzer
from nety.core.response_generator import ResponseGenerator
from nety.core.llm_config import LLMConfig
from nety.modules.machinelearning.ml_engine import MLEngine

# Import lazy de TextualCortex
TextualCortex = None



class Brain:
    """Le cerveau principal de NETYOrchestre tous les modules et gère l'interaction avec le système"""
    
    def __init__(self, model_type: Optional[str] = None):
        # Initialisation des modules
        self.limbic_filter = LimbicFilter()
        self.memory = MemoryManager()
        self.knowledge = KnowledgeManager()
        self.intent_analyzer = IntentAnalyzer()
        self.ml_engine = MLEngine()
        self.emotion_engine = EmotionEngine()
        
        # ✨ Initialiser le Cortex Textuel RNN - Cerveau Neuronal Textuel Autonome (lazy)
        print("🧠 Initialisation du Cortex Textuel (RNN bi-directionnel)...")
        try:
            from nety.cortex_limbic.textual_cortex import TextualCortex as _TextualCortex
            self.textual_cortex = _TextualCortex(
                hidden_size=256,
                output_size=512,
                num_layers=3,
                num_heads=4,
                dropout=0.3,
                emotion_engine=self.emotion_engine,
                memory_manager=self.memory
            )
        except ImportError as e:
            print(f"⚠️ Erreur d'importation du Cortex Textuel: {e}")
            self.textual_cortex = None
        
        # Déterminer le modèle à utiliser
        if model_type is None:
            model_type = LLMConfig().CURRENT_MODEL
        
        # Initialiser avec le modèle choisi
        print(f"🧠 Initialisation du cerveau NETY avec {model_type.upper()}...")
        self.response_generator = ResponseGenerator(model_type=model_type)
        
        # Afficher les infos
        print(f"✅ Modèle chargé: {model_type.upper()}")
        print(f"📊 Contexte max: 8192 tokens")
        
        # Historique des interactions pour get_context()
        self.context_history = []
        
        # État des modules
        self.modules_status = {
            "cortex_textuel": "actif",
            "cortex_limbic": "actif",
            "memory": "actif",
            "knowledge_base": "actif",
            "intent_analyzer": "actif",
            "ml_engine": "actif"
        }
        
        # Dictionnaire des modules pour compatibilité
        self.modules = {}
        self.context = {}
        self.state = "active"
    
    def think(self, message: str) -> str:
        """Méthode principale pour traiter un message"""
        
        # ✅ NETTOYER LE MESSAGE AVANT STOCKAGE
        # Retirer les préfixes même ici (au cas où)
        cleaned_message = message
        for prefix in ["CHAT: ", "PROMPT: ", "CHAT:", "PROMPT:"]:
            if cleaned_message.startswith(prefix):
                cleaned_message = cleaned_message[len(prefix):].strip()
                break
        
        # Stocker l'entrée NETTOYÉE
        interaction = {"input": cleaned_message}  # ✅ Version propre
        
        # Traiter le message via le pipeline complet
        response = self.process_message(cleaned_message)  # ✅ Ici aussi
        
        # Stocker la sortie
        interaction["output"] = response
        self.context_history.append(interaction)
        
        # Limiter l'historique à 100 interactions
        if len(self.context_history) > 100:
            self.context_history = self.context_history[-100:]
        
        return self.ml_engine.generate_response(message)
    
    def _identify_user(self) -> Optional[str]:
        """Identifie l'utilisateur basé sur les key_info.jsonl"""
        try:
            key_infos = self.ml_engine.load_key_info()
            if key_infos:
                # Chercher la dernière identité enregistrée
                for key_info in reversed(key_infos):
                    if key_info.get("type") == "user_identity":
                        return key_info.get("user_id")
        except Exception as e:
            print(f"⚠️ Erreur lors de l'identification utilisateur: {e}")
        return None

    def retrieve_context(self, message: str, intent: dict) -> dict:
        """Récupère le contexte basé sur le message et l'intention"""
        
        # Récupérer les connaissances pertinentes
        knowledge_data = {}
        if hasattr(self.knowledge, 'get_knowledge'):
            knowledge_data = self.knowledge.get_knowledge(intent.get('type', 'general'))
        
        # ✅ EXTRAIRE LES INFORMATIONS CLÉS DE L'HISTORIQUE
        user_name = None
        for interaction in reversed(self.context_history[-10:]):
            user_msg = interaction.get('input', '').lower()
            # Détecter "je m'appelle X" ou "je suis X"
            if "je m'appel" in user_msg or "je suis" in user_msg:
                # Extraire le nom (simpliste)
                words = user_msg.split()
                try:
                    if "m'appel" in user_msg:
                        idx = words.index("m'appel") if "m'appel" in words else words.index("m'appelle")
                        user_name = words[idx + 1].strip('.,!?')
                    elif "je suis" in user_msg:
                        idx = words.index("suis")
                        user_name = words[idx + 1].strip('.,!?')
                except:
                    pass
        
        ml_profile = self.ml_engine.get_user_profile()
        if not user_name:
            user_name = ml_profile.get("name")

        # ✨ UTILISER LA NOUVELLE MÉTHODE DE RÉCUPÉRATION DE MÉMOIRES AVEC CONTEXTE
        personal_memories = self.ml_engine.get_relevant_memories(message, limit=10)
        
        # ✅ CHARGER LES KEY_INFO (identité, rôles, etc.)
        key_infos = self.ml_engine.load_key_info()
        
        # ✅ DÉTECTER L'UTILISATEUR POUR LE USER_ID
        user_id = self._identify_user()

        context = {
            "message": message,
            "intent": intent,
            "history": self.context_history[-5:],
            "knowledge": knowledge_data,
            "user_name": user_name,  # ✅ Info clé extraite
            "personal_memory": personal_memories,  # ✨ Mémoires améliorées avec labels et corrélations
            "user_profile": ml_profile,
            "key_infos": key_infos,  # ✅ Infos clés (identité, rôles)
            "user_id": user_id,  # ✅ ID utilisateur détecté
            "memory_context": {  # ✨ Contexte de mémoire enrichi
                "recent_labels": self._extract_memory_labels(personal_memories),
                "memory_sentiment": self._extract_memory_sentiment(personal_memories),
            }
        }
        return context
    
    def _extract_memory_labels(self, memories: list) -> list:
        """Extrait les labels des souvenirs pour enrichir le contexte"""
        labels = set()
        for memory in memories:
            if isinstance(memory, dict) and "labels" in memory:
                labels.update(memory.get("labels", []))
        return list(labels)
    
    def _extract_memory_sentiment(self, memories: list) -> str:
        """Détermine le sentiment global des souvenirs récents"""
        sentiments = []
        for memory in memories:
            if isinstance(memory, dict) and "meta" in memory:
                sent = memory.get("meta", {}).get("sentiment", "neutral")
                sentiments.append(sent)
        
        if not sentiments:
            return "neutral"
        
        pos = sentiments.count("positive")
        neg = sentiments.count("negative")
        
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        else:
            return "neutral"
    
    def process_message(self, message: str) -> str:
        """Pipeline complet de traitement"""
        
        # [1] Analyse d'intention
        intent = self.intent_analyzer.analyze(message)
        
        # [2] Récupération contextuelle
        context = self.retrieve_context(message, intent)
        
        # [3] Filtrage limbique avancé ✨
        personality_filter = self.limbic_filter.apply_filter(context)
        
        # [3.5] ✨ TRAITEMENT RNN DU CORTEX TEXTUEL (Nouveau!)
        # Traiter le message via le cortex neuronal textuel autonome
        if self.textual_cortex is not None:
            try:
                # Convertir le message en embeddings pour le RNN
                message_embedding = self._get_message_embedding(message)
                if message_embedding is not None:
                    # Traiter via le cortex textuel avec modulation émotionnelle
                    neural_output, neural_metadata = self.textual_cortex.process_text_sequence(
                        message_embedding,
                        emotional_context={
                            "emotions": self.emotion_engine.emotions
                        },
                        use_persistent_state=True
                    )
                    
                    # Ajouter l'activation neuronal au contexte
                    context["neural_activation"] = neural_metadata["activation_level"]
                    context["neural_output"] = neural_output.detach().cpu() if isinstance(neural_output, torch.Tensor) else neural_output
                    
                    # Enregistrer dans la fenêtre contextuelle du cortex
                    self.textual_cortex.add_to_context_window({
                        "input": message,
                        "timestamp": neural_metadata["timestamp"],
                        "activation": neural_metadata["activation_level"]
                    })
            except Exception as e:
                print(f"⚠️ Cortex textuel processing: {e}")
        
        # [4] Génération de réponse
        response = self.response_generator.generate(
            message, context, personality_filter
        )

        # [4.5] Ingestion ML (mémoire personnelle) avec user_id détecté et labels
        try:
            user_id = context.get("user_id")
            if user_id:
                print(f"👤 Utilisateur identifié: {user_id}")
            
            # ✨ Utiliser la nouvelle méthode avec labels et corrélations
            memory_entry = self.ml_engine.assign_memory_labels(message, user_id=user_id)
            
            # Enregistrer dans la mémoire locale avec labels
            if self.memory:
                memory_id = self.memory.add_memory(
                    message[:100],  # Résumé court
                    labels=memory_entry.get("labels", ["other"]),
                    metadata={
                        "user_id": user_id,
                        "categories": memory_entry.get("categories", []),
                        "sentiment": memory_entry.get("meta", {}).get("sentiment", "neutral"),
                        "keywords": memory_entry.get("keywords", [])[:5]
                    }
                )
            
            stats = self.ml_engine.get_stats()
            if stats.get("total_entries", 0) % 20 == 0:
                self.ml_engine.train_from_memory()
        except Exception as exc:
            print(f"⚠️ ML Engine ingestion error: {exc}")
        
        # [5] Enregistrement de l'interaction pour apprentissage ✨
        user_sentiment = self._analyze_user_sentiment(message)
        interaction_data = {
            "message": message,
            "response": response,
            "user_id": context.get("user_id"),
            "emotional_state": personality_filter.get("emotional_state"),
            "user_sentiment": user_sentiment,
            "neural_activation": context.get("neural_activation")
        }
        # Store interaction in memory instead
        self.memory.add_memory(f"Interaction: {message[:50]} -> {response[:50]}")
        
        return response
    
    def _analyze_user_sentiment(self, message: str) -> str:
        """Analyse le sentiment de l'utilisateur à partir du message"""
        positive_words = ["merci", "super", "génial", "content", "heureux", "aime", "formidable", "excellent"]
        negative_words = ["triste", "nul", "mauvais", "déçu", "horrible", "déteste", "frustré"]
        
        message_lower = message.lower()
        
        # Compter les mots positifs et négatifs
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if positive_count > negative_count:
            return "positif"
        elif negative_count > positive_count:
            return "négatif"
        else:
            return "neutre"
    
    def _get_message_embedding(self, message: str) -> Optional[torch.Tensor]:
        """
        Convertit un message en embeddings pour le cortex textuel RNN.
        Utilise une dimension de 768 (standard pour les embeddings modernes).
        """
        try:
            import numpy as np
            
            # Pour la démo, créer un embedding basé sur le hash du message
            # En production, utiliser un vrai modèle d'embedding (FastText, BERT, etc.)
            hash_val = hash(message)
            np.random.seed(abs(hash_val) % (2**31))
            
            # Créer un embedding synthétique (768 dimensions)
            embedding = np.random.randn(1, 1, 768).astype(np.float32)
            
            # Normaliser
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return torch.from_numpy(embedding)
        except Exception as e:
            print(f"⚠️ Erreur lors de la création d'embedding: {e}")
            return None
    
    def get_modules_status(self):
        """Retourne l'état de tous les modules"""
        return self.modules_status.copy()
    
    def register_module(self, name, module):
        """Enregistre un nouveau module"""
        self.modules[name] = module
    
    def shutdown(self):
        """Arrête proprement le Brain"""
        self.state = "stopped"
        print("🧠 Brain arrêté")
    
    # ===== Méthodes supplémentaires pour compatibilité =====
    
    def process(self, data):
        """Alias pour think()"""
        return self.think(data)
    
    def add_to_memory(self, input_data, output_data):
        """Ajoute une interaction à la mémoire"""
        summary = f"Input: {str(input_data)[:50]} | Output: {str(output_data)[:50]}"
        self.memory.add_memory(summary)
    
    def get_memory(self):
        """Récupère la mémoire"""
        return self.memory
    
    def clear(self):
        """Vide le Brain (mémoire et contexte)"""
        self.clear_memory()
        self.context_history = []

    def clear_all(self):
        """Vide complètement le Brain"""
        self.clear_memory()
        self.context_history = []
        # Réinitialiser KnowledgeManager en ré-instanciant l'objet
        self.knowledge = KnowledgeManager()

    def reset(self):
        """Réinitialise le Brain"""
        self.clear_memory()
        self.context_history = []
        self.state = "active"

    def clear_memory(self):
        """Vide la mémoire"""
        # Réinitialiser MemoryManager en ré-instanciant l'objet
        self.memory = MemoryManager()
    
    def set_context(self, key, value):
        """Définit un élément de contexte"""
        self.context[key] = value
    
    def get_context_value(self, key=None):
        """Récupère le contexte ou un élément"""
        if key is None:
            return self.context
        return self.context.get(key)
    
    def get_module(self, name):
        """Récupère un module spécifique"""
        return self.modules.get(name)
    
    def activate_module(self, name):
        """Active un module"""
        if name in self.modules:
            module = self.modules[name]
            if hasattr(module, 'activate'):
                module.activate()
            return True
        return False
    
    def deactivate_module(self, name):
        """Désactive un module"""
        if name in self.modules:
            module = self.modules[name]
            if hasattr(module, 'deactivate'):
                module.deactivate()
            return True
        return False
    
    def get_state(self):
        """Retourne l'état actuel du Brain"""
        return self.state
    
    def set_state(self, new_state):
        """Définit un nouvel état"""
        self.state = new_state
    
    def update_memory(self, message: str, response: str):
        """Met à jour la mémoire"""
        summary = f"User: {message[:50]}... | Response: {response[:50]}..."
        self.memory.add_memory(summary)
    
    def get_context(self) -> list:
        """
        Retourne l'historique des interactions (contexte)
        Utilisé pour les tests et le debugging
        
        Returns:
            List of interactions with {input, output} keys
            Liste des interactions {input, output}
        """
        return self.context_history.copy()


# Alias pour compatibilité
NETYBrain = Brain
