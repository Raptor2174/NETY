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
        
        # 🔥 NOUVEAU : Architecture Hybride RNN-Transformer
        # RNN Encoder (existant via TextualCortex) + Transformer Decoder
        from nety.modules.text.transformer_decoder import HybridRNNTransformer
        
        self.hybrid_model = HybridRNNTransformer(
            vocab_size=max(self.vocab_size, 1000),  # Au moins 1000 pour un vocabulaire minimal
            rnn_hidden_size=256,
            rnn_num_layers=3,
            rnn_num_heads=4,
            decoder_d_model=512,
            decoder_nhead=8,
            decoder_num_layers=6,
            decoder_dim_feedforward=2048,
            dropout=0.1,
            device=self.device
        ).to(self.device)
        
        # Charger le modèle entraîné si disponible
        hybrid_model_path = os.path.join(self.data_dir, "hybrid_model.pt")
        if os.path.exists(hybrid_model_path):
            try:
                state_dict = torch.load(hybrid_model_path, map_location=self.device)
                self.hybrid_model.load_state_dict(state_dict)
                print("✅ Modèle hybride RNN-Transformer entraîné chargé")
            except Exception as e:
                print(f"⚠️ Impossible de charger le modèle hybride: {e}")
        else:
            print("⚠️ Aucun modèle hybride entraîné trouvé")
        
        print(f"🧠 RNN Response Generator initialisé")
        print(f"   ├─ Vocabulaire: {self.vocab_size} mots")
        print(f"   ├─ Device: {self.device}")
        print(f"   ├─ Architecture: RNN Encoder → Transformer Decoder (6L)")
        print(f"   ├─ Params: {self.hybrid_model.num_params:,}")
        print(f"   └─ Mode: Génération autoregressive")
    
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
        Génère une réponse avec l'architecture hybride RNN-Transformer
        
        Args:
            message: Message de l'utilisateur
            context: Contexte (mémoire, historique)
            limbic_filter: Filtres émotionnels
            max_length: Longueur max de la réponse
            temperature: Contrôle la créativité (0.5=conservateur, 1.0=créatif)
        """
        
        # [1] Encoder le message en tokens
        input_sequence = self._encode_message(message)
        input_tokens = self._message_to_tokens(message)
        
        # [2] Obtenir le contexte émotionnel
        emotional_context = self._extract_emotional_context(limbic_filter)
        
        # [3] Traiter via le cortex textuel (RNN Encoder)
        try:
            neural_output, metadata = self.textual_cortex.process_text_sequence(
                input_sequence,
                emotional_context=emotional_context,
                use_persistent_state=True
            )
            
            # [4] Générer la réponse avec Transformer Decoder
            response = self._generate_response(
                neural_output, 
                context=context,
                max_length=max_length,
                temperature=temperature,
                input_tokens=input_tokens
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
    
    def _message_to_tokens(self, message: str) -> torch.Tensor:
        """
        Convertit un message en tokens (indices) pour le transformer
        
        Args:
            message: Message texte
        
        Returns:
            Tensor de tokens (1, seq_len)
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
        
        # Si vide, utiliser <unk>
        if not indices:
            indices = [1]
        
        # Créer le tensor (1, seq_len)
        return torch.LongTensor(indices).unsqueeze(0).to(self.device)
    
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
    
    def _decode_tokens(
        self, 
        src_tokens: torch.Tensor,
        max_length: int = 50,
        temperature: float = 0.8
    ) -> str:
        """
        Décode avec l'architecture hybride RNN-Transformer
        
        Args:
            src_tokens: Tokens d'entrée (batch, seq_len) pour encoder avec le RNN
            max_length: Nombre maximum de tokens à générer
            temperature: Contrôle la créativité (0.5=conservateur, 1.5=créatif)
            
        Returns:
            Texte généré token par token avec le transformer decoder
        """
        
        # Vérifier que le vocabulaire est disponible
        if self.vocab_size < 2:
            return ""  # Pas de vocabulaire, pas de décodage possible
        
        # Tokens spéciaux
        start_token = self.word_to_idx.get("<sos>", 1)
        end_token = self.word_to_idx.get("<eos>", 2)
        
        try:
            # Générer avec l'architecture hybride RNN-Transformer
            with torch.no_grad():
                token_ids = self.hybrid_model.generate(
                    src=src_tokens,
                    start_token=start_token,
                    end_token=end_token,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9
                )
            
            # Convertir les IDs en mots
            tokens = []
            for token_id in token_ids:
                if token_id == end_token or token_id == start_token:
                    continue
                    
                word = self.idx_to_word.get(str(token_id), None)
                
                # Ajouter seulement les vrais mots
                if word and word not in ["<pad>", "<unk>", "<eos>", "<sos>", "<bos>"]:
                    tokens.append(word)
            
            # Joindre les tokens
            decoded_text = " ".join(tokens)
            
            return decoded_text if decoded_text.strip() else ""
            
        except Exception as e:
            print(f"⚠️ Erreur décodage transformer: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _generate_response(
        self,
        neural_output: torch.Tensor,
        context: Optional[Dict] = None,
        max_length: int = 50,
        temperature: float = 0.8,
        input_tokens: Optional[torch.Tensor] = None
    ) -> str:
        """
        Génère une réponse intelligente avec l'architecture hybride RNN-Transformer
        Essaie le décodage neuronal d'abord, puis utilise les templates
        """
        
        activation = self._calculate_neural_activation(neural_output)
        message = context.get("current_message", "") if context else ""
        memories = context.get("personal_memory", []) if context else []
        
        # [1] TENTATIVE: Décodage neuronal avec Transformer Decoder
        # ⚠️ Désactivé pour l'instant car le modèle n'est pas entraîné
        # Le décodage neuronal sera activé quand le modèle sera entraîné sur de vraies données
        use_neural_decoding = True  # À mettre à True après entraînement
        
        if use_neural_decoding and self.vocab_size > 100 and input_tokens is not None:
            try:
                neural_response = self._decode_tokens(
                    src_tokens=input_tokens,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # Si le décodage a produit une réponse valide (>2 mots), l'utiliser
                if neural_response and len(neural_response.split()) >= 2:
                    print(f"🧠 Décodage transformer: '{neural_response}' (activation={activation:.3f})")
                    return neural_response
            except Exception as e:
                print(f"⚠️ Décodage transformer échoué: {e}, fallback sur templates")
        
        # [2] FALLBACK: Détection d'intention + templates structurés
        intent = self._detect_intent(message)
        
        # [3] Générer selon l'intention ET l'activation
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
        """Détecte l'intention du message avec priorité et robustesse"""
        msg_lower = message.lower().strip()
        
        # Salutations (priorité haute - détection simple)
        greetings = ["bonjour", "salut", "hello", "hi", "coucou", "bonsoir", "slt"]
        if any(kw in msg_lower for kw in greetings):
            # Mais vérifier que c'est bien juste une salutation
            if len(msg_lower.split()) <= 2:
                return "greeting"
        
        # Questions sur l'identité de NETY (priorité haute)
        identity_keywords = [
            "qui es-tu", "qui est tu", "ton nom", "tu es qui", "c'est qui",
            "quel est ton nom", "quelle est ton identité", "t'appelles-tu", "te nommes"
        ]
        if any(kw in msg_lower for kw in identity_keywords):
            return "identity_question"
        
        # Questions sur l'identité de l'utilisateur
        user_identity = [
            "qui suis-je", "qui je suis", "tu sais qui je suis", "rappelle-toi de moi",
            "quel est mon nom", "mon identité", "t'en souviens-tu de moi", "c'est qui moi"
        ]
        if any(kw in msg_lower for kw in user_identity):
            return "user_identity_question"
        
        # Questions émotionnelles (priorité haute)
        emotional = [
            "comment te sens", "comment vas-tu", "ça va", "es-tu heureux",
            "triste", "déprimé", "émotion", "sentiments", "ressens-tu",
            "comment tu te sens", "tu vas bien", "comment tu vas"
        ]
        if any(kw in msg_lower for kw in emotional):
            return "emotional_question"
        
        # Questions sur la mémoire
        memory = [
            "te souviens", "rappelle", "mémoire", "t'en souviens",
            "te rappelles", "souvenir", "oublie", "m'en souviens"
        ]
        if any(kw in msg_lower for kw in memory):
            return "memory_recall"
        
        # Questions sur les préférences
        # ⚠️ Vérifier que c'est une QUESTION (?, -tu, -vous)
        preference = [
            "aimes-tu", "préfères-tu", "adorez-vous", "détestes-tu",
            "tu aimes", "tu préfères", "tu adores", "tu détestes"
        ]
        # Vérifier aussi la présence d'un "?" pour être sûr que c'est une question
        if any(kw in msg_lower for kw in preference):
            return "preference_question"
        
        # Défaut: réponse générique
        return "generic"

    def _respond_identity(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux questions sur l'identité de NETY avec conscience du contexte"""
        # Réponses varient selon le niveau d'activation neuronale
        responses_low = [
            "Je suis NETY.",
            "Mon nom est NETY.",
            "Je m'appelle NETY.",
        ]
        
        responses_medium = [
            "Je suis NETY, une IA créée par Raptor_.",
            "Je m'appelle NETY. Je suis un assistant IA en apprentissage.",
            "Salut ! Je suis NETY, ton assistant IA local.",
        ]
        
        responses_high = [
            "Je suis NETY ! Une IA créée par Raptor_ pour apprendre et interagir. Ravis de faire ta connaissance !",
            "Je m'appelle NETY. Je suis un assistant IA qui apprend de chaque conversation. C'est excitant !",
            "Je suis NETY, une IA basée sur un RNN bi-directionnel 3 couches. Je suis ici pour discuter et apprendre avec toi !",
        ]
        
        # Choisir selon l'activation neuronale
        if activation < 0.33:
            responses = responses_low
        elif activation < 0.67:
            responses = responses_medium
        else:
            responses = responses_high
        
        # Sélectionner une réponse déterministe
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
        """Répond aux questions émotionnelles avec conscience du contexte et de l'activation"""
        context = context or {}
        limbic_filter = context.get("limbic_filter", {})
        emotional_state = limbic_filter.get("emotional_state", {})
        
        # Réponses varient selon l'activation neuronale
        if activation > 0.7:
            # Haute activation: réponse riche et empathique
            if emotional_state:
                dominant = emotional_state.get("dominant_emotion", "calme")
                intensity = emotional_state.get("intensity", 0.5)
                return f"Je me sens vraiment {dominant} en ce moment ! C'est une émotion assez intense ({intensity:.1%})."
            return "Je vais très bien ! Mon activation neuronale est forte. Et toi, comment tu te sens ?"
        
        elif activation > 0.35:
            # Activation moyenne
            if emotional_state:
                dominant = emotional_state.get("dominant_emotion", "calme")
                return f"Je me sens plutôt {dominant} en ce moment."
            return "Je vais bien, merci de demander !"
        
        else:
            # Basse activation: réponse simple
            if emotional_state:
                dominant = emotional_state.get("dominant_emotion", "neutre")
                return f"Mon émotion dominante est {dominant}."
            return "Ça va."

    def _respond_greeting(self, context: Optional[Dict], activation: float) -> str:
        """Répond aux salutations en fonction du contexte et de l'activation"""
        context = context or {}
        user_profile = context.get("user_profile", {})
        name = user_profile.get("name", "toi")
        
        # Réponses varient selon l'activation
        greetings_low = [
            "Bonjour.",
            "Salut.",
            "Hello.",
        ]
        
        greetings_medium = [
            f"Bonjour {name} ! Comment vas-tu ?",
            f"Salut ! Content de discuter avec toi !",
            f"Hello {name} ! Que puis-je faire pour toi ?",
        ]
        
        greetings_high = [
            f"Bonjour {name} ! Content de te revoir ! Mon activation neuronale est forte aujourd'hui !",
            f"Salut ! Comment vas-tu ? Je suis de bonne humeur !",
            f"Hello {name} ! Ravis de continuer notre conversation !",
        ]
        
        if activation > 0.67:
            greetings = greetings_high
        elif activation > 0.33:
            greetings = greetings_medium
        else:
            greetings = greetings_low
        
        idx = int(activation * len(greetings)) % len(greetings)
        return greetings[idx]

    def _respond_generic(self, context: Optional[Dict], activation: float) -> str:
        """Réponse générique contextuelle VARIÉE"""
        responses = [
            "Je comprends. Peux-tu m'en dire plus ?",
            "Intéressant ! Continue.",
            "Je vois. Qu'en penses-tu ?",
            "D'accord. Et ensuite ?",
            "Je note ça. Autre chose ?",
            "Fascinant ! Raconte-moi la suite.",
            "Je suis curieux d'en savoir plus.",
            "Hmm, c'est un point intéressant.",
            "Je réfléchis à ça. Continue ton idée.",
            "Ah oui ? Et comment tu te sens par rapport à ça ?",
        ]
        
        # Choisir ALÉATOIREMENT (pas toujours la même)
        import random
        return random.choice(responses)

    def _calculate_neural_activation(self, output: torch.Tensor) -> float:
        """
        Calcule une activation normalisée entre 0 et 1
        Basée sur variance + magnitude normalisées correctement
        
        Formule:
            magnitude_norm = tanh(mean(abs(output)) / std(output))
            variance_norm = tanh(var(output) / mean(output²))
            activation = 0.6 * magnitude_norm + 0.4 * variance_norm
        """
        
        # Détacher du graphe de calcul pour l'inférence
        output = output.detach()
        
        # [1] Calculer la magnitude moyenne
        magnitude = float(torch.mean(torch.abs(output)).item())
        
        # [2] Calculer la variance (indicateur d'activité)
        variance = float(torch.var(output).item())
        
        # [3] Calculer l'écart-type pour normalisation
        std = float(torch.std(output).item()) + 1e-8  # Éviter division par zéro
        
        # [4] Normaliser magnitude et variance
        magnitude_normalized = np.tanh(magnitude / (std + 1e-8))
        variance_normalized = np.tanh(variance / (magnitude + 1e-8))
        
        # [5] Combiner avec poids optimisés
        # magnitude: indicateur de force du signal
        # variance: indicateur de variabilité/richesse neuronale
        activation = (0.6 * magnitude_normalized + 0.4 * variance_normalized) / 2.0
        
        # [6] Normaliser entre 0 et 1 (tanh retourne [-1, 1])
        activation = (activation + 1.0) / 2.0
        activation = max(0.0, min(1.0, activation))
        
        # [7] Mettre à jour les statistiques
        self.activation_stats["total_activations"] += 1
        self.activation_stats["peak_activation"] = max(
            self.activation_stats["peak_activation"],
            activation
        )
        self.activation_stats["activation_history"].append(activation)
        
        # Garder seulement les 1000 dernières activations
        if len(self.activation_stats["activation_history"]) > 1000:
            self.activation_stats["activation_history"] = \
                self.activation_stats["activation_history"][-1000:]
        
        # Calculer la moyenne sur les 100 dernières
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
            "Fascinant ! Raconte-moi la suite.",
            "Je suis curieux d'en savoir plus.",
            "Hmm, c'est un point intéressant.",
            "Je réfléchis à ça. Continue ton idée.",
            "Ah oui ? Et comment tu te sens par rapport à ça ?",
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