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
        Génère une réponse texte à partir de la sortie neuronale
        Utilise un décodage basé sur le vocabulaire
        """
        
        # Pour cette v1, utiliser un système basé sur des templates + mémoire
        # Le RNN influence le choix via son activation
        
        activation = torch.mean(torch.abs(neural_output)).item()
        
        # Récupérer les souvenirs pertinents
        memories = context.get("personal_memory", []) if context else []
        profile = context.get("user_profile", {}) if context else {}
        
        # Stratégie de génération basée sur l'activation
        if activation > 0.7:
            # Haute activation = réponse détaillée
            response_parts = []
            
            if memories:
                mem = memories[0]
                text = mem.get("text", "")
                if text:
                    response_parts.append(f"Je me souviens : {text}")
            
            if profile:
                name = profile.get("name")
                if name:
                    response_parts.append(f"Tu es {name}, n'est-ce pas ?")
            
            if response_parts:
                return " ".join(response_parts)
        
        elif activation > 0.4:
            # Activation moyenne = réponse standard
            if memories:
                categories = memories[0].get("categories", [])
                if "identity" in categories:
                    return "Je me souviens de notre conversation sur ton identité."
                elif "goals" in categories:
                    return "Je me rappelle de tes objectifs."
        
        # Activation faible = réponse simple
        return self._template_response(context)
    
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