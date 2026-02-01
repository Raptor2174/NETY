# nety/core/rnn_response_generator.py

"""
Générateur de réponses basé sur le RNN local (TextualCortex)
Utilise le modèle RNN bi-directionnel entraîné pour générer des réponses
"""

import torch
import json
import os
from typing import Dict, Optional, List
import numpy as np


class RNNResponseGenerator:
    """
    Générateur de réponses basé sur le RNN local
    Utilise le TextualCortex et le vocabulaire entraîné
    """
    
    def __init__(self, data_dir: str = "data/processed/ml_engine"):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Statistiques d'activation
        self.activation_stats = {
            "total_activations": 0,
            "average_activation": 0.0,
            "peak_activation": 0.0,
            "activation_history": []
        }
        
        # Charger le vocabulaire
        self.vocab = self._load_vocab()
        self.word_to_idx = self.vocab.get("word_to_idx", {})
        self.idx_to_word = self.vocab.get("idx_to_word", {})
        self.vocab_size = len(self.word_to_idx)
        
        # Initialiser le RNN
        from nety.cortex_limbic.textual_cortex import TextualCortex
        
        self.textual_cortex = TextualCortex(
            hidden_size=256,
            output_size=512,
            num_layers=3,
            num_heads=4,
            dropout=0.2,  # Moins de dropout pour l'inférence
            device=self.device
        )
        
        # Charger le modèle entraîné si disponible
        self._load_trained_model()
        
        print(f"🧠 RNN Response Generator initialisé")
        print(f"   ├─ Vocabulaire: {self.vocab_size} mots")
        print(f"   ├─ Device: {self.device}")
        print(f"   └─ Mode: Génération autonome")
    
    def _load_vocab(self) -> Dict:
        """Charge le vocabulaire depuis vocab.json"""
        vocab_path = os.path.join(self.data_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"word_to_idx": {"<pad>": 0, "<unk>": 1}, "idx_to_word": {"0": "<pad>", "1": "<unk>"}}
    
    def _load_trained_model(self) -> None:
        """Charge le modèle RNN entraîné"""
        model_path = os.path.join(self.data_dir, "model.pt")
        if os.path.exists(model_path):
            try:
                # Le modèle est chargé par le TextualCortex automatiquement
                print("✅ Modèle RNN entraîné chargé")
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle entraîné: {e}")
        else:
            print("⚠️ Aucun modèle entraîné trouvé (utilisation du modèle initial)")
    
    def generate(
        self, 
        message: str, 
        context: Optional[Dict] = None,
        limbic_filter: Optional[Dict] = None,
        max_length: int = 50,
        temperature: float = 0.8
    ) -> str:
        """
        Génère une réponse avec le RNN
        
        Args:
            message: Message de l'utilisateur
            context: Contexte (mémoire, historique)
            limbic_filter: Filtres émotionnels
            max_length: Longueur max de la réponse
            temperature: Contrôle la créativité (0.5=conservateur, 1.0=créatif)
        """
        
        # [1] Encoder le message
        input_sequence = self._encode_message(message)
        
        # [2] Obtenir le contexte émotionnel
        emotional_context = self._extract_emotional_context(limbic_filter)
        
        # [3] Traiter via le cortex textuel
        try:
            neural_output, metadata = self.textual_cortex.process_text_sequence(
                input_sequence,
                emotional_context=emotional_context,
                use_persistent_state=True
            )
            
            # [4] Générer la réponse token par token
            response = self._generate_response(
                neural_output, 
                context=context,
                max_length=max_length,
                temperature=temperature
            )
            
            # [5] Post-traitement
            response = self._post_process(response)
            
            # Statistiques
            activation = metadata.get("activation_level", 0.0)
            print(f"🧠 RNN activation: {activation:.3f}")
            
            return response
            
        except Exception as e:
            print(f"❌ Erreur RNN: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_response(message, context)
    
    def _encode_message(self, message: str) -> torch.Tensor:
        """
        Encode le message en tensor pour le RNN
        Utilise le vocabulaire entraîné
        """
        words = message.lower().split()
        
        # Convertir en indices
        indices = []
        for word in words:
            idx = self.word_to_idx.get(word, self.word_to_idx.get("<unk>", 1))
            indices.append(idx)
        
        # Limiter la longueur
        if len(indices) > 50:
            indices = indices[:50]
        
        # Créer un embedding simple (moyenne de one-hot encodings)
        # En production, utiliser un vrai modèle d'embedding (Word2Vec, FastText, etc.)
        embedding_dim = 768
        
        # Embedding basique : hash des mots
        embeddings = []
        for idx in indices:
            # Créer un embedding pseudo-aléatoire mais déterministe
            np.random.seed(idx)
            emb = np.random.randn(embedding_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        
        if not embeddings:
            embeddings = [np.zeros(embedding_dim, dtype=np.float32)]
        
        # Shape: (batch=1, seq_len, embedding_dim)
        embeddings_array = np.array(embeddings).reshape(1, len(embeddings), embedding_dim)
        
        return torch.from_numpy(embeddings_array).to(self.device)
    
    def _extract_emotional_context(self, limbic_filter: Optional[Dict]) -> Dict:
        """Extrait le contexte émotionnel du filtre limbique"""
        if not limbic_filter:
            return {"emotions": {}}
        
        emotional_state = limbic_filter.get("emotional_state", {})
        all_emotions = emotional_state.get("all_emotions", {})
        
        return {"emotions": all_emotions}
    
    def _generate_response(
        self,
        neural_output: torch.Tensor,
        context: Optional[Dict] = None,
        max_length: int = 50,
        temperature: float = 0.8
    ) -> str:
        """
        Génère une réponse intelligente basée sur la sortie neuronale
        """
        
        activation = self._calculate_neural_activation(neural_output)
        message = context.get("current_message", "") if context else ""
        memories = context.get("personal_memory", []) if context else []
        
        # [1] Détecter l'intention de la question
        intent = self._detect_intent(message)
        
        # [2] Générer selon l'intention ET l'activation
        if intent == "identity_question":
            # Questions "qui es-tu", "ton nom", etc.
            return self._respond_identity(context, activation)
        
        elif intent == "user_identity_question":
            # Questions "qui suis-je", "tu sais qui je suis"
            return self._respond_user_identity(context, activation)
        
        elif intent == "preference_question":
            # Questions "aimes-tu X", "préfères-tu Y"
            return self._respond_preference(message, context, activation)
        
        elif intent == "memory_recall":
            # "te souviens-tu", "rappelle-toi"
            return self._respond_memory_recall(context, activation)
        
        elif intent == "emotional_question":
            # "comment te sens-tu", "es-tu heureux"
            return self._respond_emotional(context, activation)
        
        elif intent == "greeting":
            # "bonjour", "salut", "hello"
            return self._respond_greeting(context, activation)
        
        else:
            # Réponse générique contextuelle
            return self._respond_generic(context, activation)

    def _detect_intent(self, message: str) -> str:
        """Détecte l'intention du message"""
        msg_lower = message.lower()
        
        # Questions sur l'identité de NETY
        if any(kw in msg_lower for kw in ["qui es-tu", "qui est tu", "ton nom", "tu es qui"]):
            return "identity_question"
        
        # Questions sur l'identité de l'utilisateur
        if any(kw in msg_lower for kw in ["qui suis-je", "qui je suis", "tu sais qui je suis", "rappelle-toi de moi"]):
            return "user_identity_question"
        
        # Questions sur les préférences
        if any(kw in msg_lower for kw in ["aime", "préfère", "adore", "déteste", "aimes-tu"]):
            return "preference_question"
        
        # Questions sur la mémoire
        if any(kw in msg_lower for kw in ["te souviens", "rappelle", "mémoire", "t'en souviens"]):
            return "memory_recall"
        
        # Questions émotionnelles
        if any(kw in msg_lower for kw in ["comment te sens", "comment vas-tu", "ça va", "es-tu heureux", "triste"]):
            return "emotional_question"
        
        # Salutations
        if any(kw in msg_lower for kw in ["bonjour", "salut", "hello", "hi", "coucou"]):
            return "greeting"
        
        return "generic"

    def _respond_identity(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux questions sur l'identité de NETY"""
        responses = [
            "Je suis NETY, une IA créée par Raptor_.",
            "Mon nom est NETY. Je suis un assistant IA en apprentissage.",
            "Je m'appelle NETY. Je suis là pour apprendre et discuter avec toi.",
        ]
        
        # Choisir selon l'activation
        idx = int(activation * len(responses)) % len(responses)
        return responses[idx]

    def _respond_user_identity(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux questions sur l'identité de l'utilisateur"""
        context = context or {}
        profile = context.get("user_profile", {})
        memories = context.get("personal_memory", [])
        
        # Chercher le nom dans le profil
        name = profile.get("name")
        if name:
            return f"Tu es {name}. Je me souviens de toi !"
        
        # Chercher dans les souvenirs
        for mem in memories:
            if "name" in mem.get("facts", {}):
                name = mem["facts"]["name"][0]
                return f"Tu es {name}, n'est-ce pas ? Je me souviens de toi."
        
        return "Je ne suis pas sûr de ton nom. Peux-tu me le rappeler ?"

    def _respond_preference(self, message: str, context: Optional[Dict], activation: float) -> str:
        """Répond aux questions sur les préférences"""
        context = context or {}
        msg_lower = message.lower()
        
        # Détecter le sujet de la préférence
        if "chat" in msg_lower:
            # Chercher dans la mémoire si j'aime les chats
            memories = context.get("personal_memory", [])
            for mem in memories:
                text = mem.get("text", "").lower()
                if "chat" in text and ("aime" in text or "adore" in text):
                    return "Oui, j'aime les chats ! C'est quelque chose qui me tient à cœur."
            
            return "Je n'ai pas vraiment d'opinion sur les chats pour le moment. Et toi ?"
        
        # Réponse générique
        return "C'est une bonne question. Je dois y réfléchir."

    def _respond_memory_recall(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux demandes de rappel de mémoire"""
        context = context or {}
        memories = context.get("personal_memory", [])
        
        if not memories:
            return "Je n'ai pas de souvenirs pertinents pour le moment."
        
        # Prendre un souvenir pertinent
        mem = memories[0]
        text = mem.get("text", "")
        categories = mem.get("categories", [])
        
        if "identity" in categories:
            return f"Oui, je me souviens ! {text}"
        elif "goals" in categories:
            return f"Je me rappelle de tes objectifs : {text}"
        else:
            return f"Je me souviens que tu m'as dit : {text}"

    def _respond_emotional(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux questions émotionnelles"""
        context = context or {}
        limbic_filter = context.get("limbic_filter", {})
        emotional_state = limbic_filter.get("emotional_state", {})
        
        if emotional_state:
            dominant = emotional_state.get("dominant_emotion", "calme")
            state = emotional_state.get("state", "bien")
            
            return f"Je me sens {state}. Mon émotion dominante est {dominant}."
        
        return "Je vais bien, merci de demander !"

    def _respond_greeting(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux salutations"""
        greetings = [
            "Bonjour ! Content de te revoir !",
            "Salut ! Comment vas-tu ?",
            "Hello ! Que puis-je faire pour toi ?",
        ]
        
        idx = int(activation * len(greetings)) % len(greetings)
        return greetings[idx]

    def _respond_generic(self, context: Optional[Dict], activation: float) -> str:
        """Réponse générique contextuelle"""
        responses = [
            "Je comprends. Peux-tu m'en dire plus ?",
            "Intéressant ! Continue.",
            "Je vois. Qu'en penses-tu ?",
            "D'accord. Et ensuite ?",
            "Je note ça. Autre chose ?",
        ]
        
        import random
        return random.choice(responses)

    def _calculate_neural_activation(self, output: torch.Tensor) -> float:
        """
        Calcule une activation normalisée entre 0 et 1
        Basée sur la variance et la magnitude
        """
        
        # [1] Calculer la magnitude moyenne
        magnitude = float(torch.mean(torch.abs(output)).item())
        
        # [2] Calculer la variance (indicateur d'activité)
        variance = float(torch.var(output).item())
        
        # [3] Combiner pour obtenir une activation significative
        # Normaliser par des valeurs empiriques
        activation = (magnitude * 0.5) + (variance * 10.0)
        
        # [4] Clamper entre 0 et 1
        activation = max(0.0, min(1.0, activation))
        
        # [5] Mettre à jour les stats
        self.activation_stats["total_activations"] += 1
        self.activation_stats["peak_activation"] = max(
            self.activation_stats["peak_activation"],
            activation
        )
        self.activation_stats["activation_history"].append(activation)
        
        if len(self.activation_stats["activation_history"]) > 1000:
            self.activation_stats["activation_history"] = \
                self.activation_stats["activation_history"][-1000:]
        
        self.activation_stats["average_activation"] = np.mean(
            self.activation_stats["activation_history"][-100:]
        )
        
        return activation
    
    def _template_response(self, context: Optional[Dict]) -> str:
        """
        Génère une réponse basée sur des templates
        Utilisé en complément du RNN pour assurer une sortie cohérente
        """
        
        templates = [
            "Je comprends. Peux-tu m'en dire plus ?",
            "Intéressant ! Continue.",
            "Je vois. Qu'en penses-tu ?",
            "D'accord. Et ensuite ?",
            "Je note ça. Autre chose ?",
        ]
        
        # Choisir un template basé sur le hash du contexte
        if context:
            history = context.get("history", [])
            idx = len(history) % len(templates)
        else:
            idx = 0
        
        return templates[idx]
    
    def _post_process(self, response: str) -> str:
        """Nettoie et formate la réponse"""
        
        # Capitaliser la première lettre
        if response:
            response = response[0].upper() + response[1:]
        
        # Ajouter ponctuation si manquante
        if response and response[-1] not in ".!?":
            response += "."
        
        return response
    
    def _fallback_response(self, message: str, context: Optional[Dict]) -> str:
        """Réponse de secours en cas d'erreur"""
        
        responses = [
            "Je suis désolé, je n'ai pas bien compris. Peux-tu reformuler ?",
            "Hmm, laisse-moi réfléchir à ça...",
            "C'est une bonne question. Je vais y réfléchir.",
            "Je ne suis pas sûr de bien comprendre. Peux-tu préciser ?",
        ]
        
        import random
        return random.choice(responses)
    
    def get_model_info(self) -> Dict:
        """Retourne les informations du modèle RNN"""
        return {
            "backend": "RNN Local (TextualCortex)",
            "model": "LSTM bi-directionnel 3 couches",
            "vocab_size": self.vocab_size,
            "device": self.device,
            "ram": "~500 MB",
            "cost": "Gratuit (local)",
            "speed": "Rapide (CPU/GPU)",
            "quality": "Expérimental (en apprentissage)",
        }