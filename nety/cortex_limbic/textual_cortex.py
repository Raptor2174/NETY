"""
Cortex Textuel Autonome - Réseau Neural Récurrent du Cerveau NETY

Ce module constitue le cortex cérébral textuel autonome de NETY.
Il utilise un RNN bi-directionnel avancé avec attention pour :
  - Traiter les séquences textuelles contextuelles
  - Maintenir un état neuronal persistant entre les interactions
  - Générer des représentations textuelles cohérentes
  - S'adapter émotionnellement via le limbic filter

Architecture:
  - Intégration RNN bi-directionnel avec multi-head attention
  - État caché persistant (mémoire court-terme neuronal)
  - Lien avec le système émotionnel limbique
  - Contexte mémoire long-terme via la knowledge base
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from nety.modules.text.modele_rnn import ModeleRNN
else:
    ModeleRNN = None


class TextualCortex:
    """
    Cortex Textuel Autonome - Cerveau Neuronal Textuel de NETY
    
    Gère:
      - Traitement séquentiel de texte via RNN
      - Maintien d'état neuronal persistant
      - Intégration émotionnelle avec le système limbique
      - Apprentissage contextuel entre interactions
    """
    
    def __init__(
        self,
        hidden_size: int = 256,
        output_size: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        device: Optional[str] = None,
        emotion_engine: Optional[object] = None,
        memory_manager: Optional[object] = None
    ):
        """
        Initialise le cortex textuel.
        
        Args:
            hidden_size: Taille de l'état caché LSTM
            output_size: Taille de la représentation de sortie
            num_layers: Nombre de couches LSTM
            num_heads: Nombre de têtes d'attention
            dropout: Taux de dropout
            device: Device PyTorch (cpu/cuda)
            emotion_engine: Moteur émotionnel du limbic system
            memory_manager: Gestionnaire mémoire du cerveau
        """
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_engine = emotion_engine
        self.memory_manager = memory_manager
        
        # Modèle RNN principal (lazy loading pour éviter les imports circulaires)
        self.rnn_model = None
        self._rnn_initialized = False
        self._rnn_params = {
            "hidden_size": hidden_size,
            "output_size": output_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "device": self.device
        }
        
        # État neuronal persistant
        self.neural_state = {
            "timestamp": datetime.now(),
            "activation_level": 0.5,  # Entre 0 et 1
            "attention_focus": None,
            "emotional_modulation": {}
        }
        
        # Historique contextuel court-terme (fenêtre glissante)
        self.context_window: List[Dict] = []
        self.max_context_window = 20  # 20 dernières interactions
        
        # Statistiques d'activation
        self.activation_stats = {
            "total_activations": 0,
            "average_activation": 0.0,
            "peak_activation": 0.0,
            "activation_history": []
        }
        
        # Paramètres d'adaptation
        self.adaptation_threshold = 0.7
        self.learning_rate_scale = 0.001
        
        print("🧠 Cortex Textuel Autonome initialisé")
        print(f"   ├─ RNN: {num_layers} couches LSTM bi-directionnelles")
        print(f"   ├─ Attention: {num_heads} têtes")
        print(f"   ├─ Device: {self.device}")
        print(f"   └─ État persistant: ACTIF")
    
    def _ensure_rnn_initialized(self) -> None:
        """Initialise le RNN model en lazy loading si nécessaire."""
        if self._rnn_initialized:
            return
        
        try:
            from nety.modules.text.modele_rnn import ModeleRNN
            
            self.rnn_model = ModeleRNN(
                input_size=768,  # Dimension typique des embeddings
                hidden_size=self._rnn_params["hidden_size"],
                output_size=self._rnn_params["output_size"],
                num_layers=self._rnn_params["num_layers"],
                num_heads=self._rnn_params["num_heads"],
                dropout=self._rnn_params["dropout"],
                bidirectional=True,
                use_attention=True,
                device=self._rnn_params["device"]
            ).to(self._rnn_params["device"])
            
            # Mode évaluation pour l'inférence
            self.rnn_model.eval()
            
            self._rnn_initialized = True
        except ImportError as e:
            print(f"⚠️ Erreur d'import du ModeleRNN: {e}")
            raise
    
    def process_text_sequence(
        self,
        text_embeddings: torch.Tensor,
        emotional_context: Optional[Dict] = None,
        use_persistent_state: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Traite une séquence textuelle via le cortex neuronal.
        
        Args:
            text_embeddings: Tensor (batch_size, seq_len, embedding_dim)
            emotional_context: Contexte émotionnel optionnel du limbic system
            use_persistent_state: Utiliser l'état caché persistant
            
        Returns:
            Tuple[représentation_neuronal, metadata_activation]
        """
        
        # Initialiser le RNN si nécessaire (lazy loading)
        self._ensure_rnn_initialized()
        
        if self.rnn_model is None:
            raise RuntimeError("RNN model failed to initialize")
        
        # Normaliser les embeddings
        text_embeddings = text_embeddings.to(self.device)
        
        # Traitement RNN avec état persistant
        neural_output = self.rnn_model(
            text_embeddings,
            use_persistent_state=use_persistent_state
        )
        
        # Calculer le niveau d'activation neuronal
        activation = self._calculate_neural_activation(neural_output)
        
        # Appliquer la modulation émotionnelle si disponible
        if emotional_context:
            neural_output = self._apply_emotional_modulation(
                neural_output,
                emotional_context
            )
        
        # Mettre à jour l'état neuronal
        self._update_neural_state(activation, emotional_context)
        
        # Créer les métadonnées d'activation
        metadata = {
            "timestamp": datetime.now(),
            "activation_level": activation,
            "state_depth": len(self.rnn_model.get_state_history()),
            "emotional_modulation": self.neural_state["emotional_modulation"]
        }
        
        return neural_output, metadata
    
    def _calculate_neural_activation(self, output: torch.Tensor) -> float:
        """Calcule le niveau d'activation neuronal."""
        # Activation = moyenne de la magnitude des activations
        activation = float(torch.mean(torch.abs(output)).item())
        activation = min(1.0, activation)  # Clamp [0, 1]
        
        # Mettre à jour les statistiques
        self.activation_stats["total_activations"] += 1
        self.activation_stats["peak_activation"] = max(
            self.activation_stats["peak_activation"],
            activation
        )
        self.activation_stats["activation_history"].append(activation)
        
        # Limiter l'historique
        if len(self.activation_stats["activation_history"]) > 1000:
            self.activation_stats["activation_history"] = \
                self.activation_stats["activation_history"][-1000:]
        
        # Calculer la moyenne
        self.activation_stats["average_activation"] = np.mean(
            self.activation_stats["activation_history"][-100:]
        )
        
        return activation
    
    def _apply_emotional_modulation(
        self,
        neural_output: torch.Tensor,
        emotional_context: Dict
    ) -> torch.Tensor:
        """
        Applique la modulation émotionnelle du système limbique.
        Les émotions positives augmentent l'activation, négatives la réduisent.
        """
        
        # Extraire les valeurs émotionnelles
        emotions = emotional_context.get("emotions", {})
        
        # Calculer le facteur de modulation émotionnelle
        positive_emotions = sum([
            emotions.get("joie", 0),
            emotions.get("confiance", 0),
            emotions.get("anticipation", 0)
        ]) / 3.0
        
        negative_emotions = sum([
            emotions.get("tristesse", 0),
            emotions.get("colère", 0),
            emotions.get("peur", 0),
            emotions.get("dégoût", 0)
        ]) / 4.0
        
        # Facteur de modulation [-1, 1]
        modulation_factor = positive_emotions - negative_emotions
        
        # Appliquer la modulation
        modulated_output = neural_output * (1.0 + modulation_factor * 0.3)
        
        # Stocker la modulation appliquée
        self.neural_state["emotional_modulation"] = {
            "positive": float(positive_emotions),
            "negative": float(negative_emotions),
            "factor": float(modulation_factor)
        }
        
        return modulated_output
    
    def _update_neural_state(
        self,
        activation: float,
        emotional_context: Optional[Dict] = None
    ) -> None:
        """Met à jour l'état neuronal du cortex."""
        
        # Décroissance naturelle de l'activation
        decay = 0.05
        self.neural_state["activation_level"] = \
            self.neural_state["activation_level"] * (1 - decay) + activation * decay
        
        # Mettre à jour le timestamp
        self.neural_state["timestamp"] = datetime.now()
        
        # Mettre à jour la modulation émotionnelle
        if emotional_context:
            self.neural_state["emotional_modulation"] = \
                emotional_context.get("emotions", {})
    
    def add_to_context_window(self, interaction: Dict) -> None:
        """
        Ajoute une interaction à la fenêtre contextuelle.
        
        Args:
            interaction: Dict avec 'input', 'output', 'timestamp', etc.
        """
        self.context_window.append(interaction)
        
        # Limiter la fenêtre
        if len(self.context_window) > self.max_context_window:
            self.context_window = self.context_window[-self.max_context_window:]
    
    def get_context_summary(self) -> Dict:
        """Récupère un résumé du contexte courant."""
        return {
            "window_size": len(self.context_window),
            "neural_state": self.neural_state,
            "activation_stats": self.activation_stats,
            "recent_interactions": [
                {
                    "timestamp": i.get("timestamp"),
                    "input": i.get("input", "")[:50] + "...",
                }
                for i in self.context_window[-5:]
            ]
        }
    
    def reset_state(self) -> None:
        """Réinitialise complètement l'état neuronal du cortex."""
        if self._rnn_initialized and self.rnn_model is not None:
            self.rnn_model.reset_persistent_state()
        self.context_window.clear()
        self.neural_state = {
            "timestamp": datetime.now(),
            "activation_level": 0.5,
            "attention_focus": None,
            "emotional_modulation": {}
        }
        print("🔄 État neuronal du cortex réinitialisé")
    
    def get_persistent_state(self) -> Optional[Tuple]:
        """Récupère l'état persistant du RNN pour sauvegarde."""
        if not self._rnn_initialized or self.rnn_model is None:
            return None
        if self.rnn_model.persistent_h is not None and self.rnn_model.persistent_c is not None:
            return (
                self.rnn_model.persistent_h.cpu(),
                self.rnn_model.persistent_c.cpu()
            )
        return None
    
    def load_persistent_state(self, state: Tuple) -> None:
        """Charge un état persistant sauvegardé."""
        if state is not None:
            self._ensure_rnn_initialized()
            if self.rnn_model is None:
                raise RuntimeError("RNN model failed to initialize")
            h, c = state
            self.rnn_model.set_persistent_state(
                h.to(self.device),
                c.to(self.device)
            )
            print("✅ État neuronal du cortex restauré")
    
    def get_neural_statistics(self) -> Dict:
        """Récupère les statistiques neurales détaillées."""
        context_depth = 0
        if self._rnn_initialized and self.rnn_model is not None:
            context_depth = len(self.rnn_model.get_state_history())
        
        return {
            "total_activations": self.activation_stats["total_activations"],
            "average_activation": float(self.activation_stats["average_activation"]),
            "peak_activation": float(self.activation_stats["peak_activation"]),
            "current_activation": float(self.neural_state["activation_level"]),
            "context_depth": context_depth,
            "last_update": self.neural_state["timestamp"].isoformat()
        }


if __name__ == "__main__":
    print("🧠 Initialisation du Cortex Textuel de NETY...\n")
    
    # Créer le cortex
    cortex = TextualCortex(
        hidden_size=256,
        output_size=512,
        num_layers=3,
        num_heads=4,
        dropout=0.3
    )
    
    # Exemple d'utilisation
    batch_size = 2
    seq_length = 10
    embedding_dim = 768
    
    # Créer des embeddings fictifs
    sample_embeddings = torch.randn(batch_size, seq_length, embedding_dim)
    
    # Traiter
    output, metadata = cortex.process_text_sequence(sample_embeddings)
    
    print(f"\n✅ Traitement réussi!")
    print(f"   ├─ Sortie shape: {output.shape}")
    print(f"   ├─ Activation level: {metadata['activation_level']:.3f}")
    print(f"   └─ Métadonnées: {metadata}")
    
    # Afficher les stats
    print(f"\n📊 Statistiques Neurales:")
    for key, val in cortex.get_neural_statistics().items():
        print(f"   ├─ {key}: {val}")
