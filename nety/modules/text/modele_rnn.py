"""
Module de réseau de neurones récurrent (RNN) pour le traitement de texte.

Ce module implémente un LSTM bi-directionnel avancé avec :
  - Attention mechanism multi-head
  - Couches LSTM avec dropout
  - Batch normalization
  - Gestion persistante de l'état caché

Architecture:
  - Embedding couches
  - Multi-head Attention
  - Bi-directional LSTM empilé
  - Couches fully-connected denses
  - Sortie avec normalisation

Amélioré et modernisé pour être le cortex cérébral textuel autonome de NETY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class MultiHeadAttention(nn.Module):
    """Mécanisme d'attention multi-tête pour le contexte textuel."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4) -> None:
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size doit être divisible par num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule l'attention multi-tête."""
        batch_size = query.shape[0]
        
        # Projection linéaire
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape pour les têtes multiples
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scores d'attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        # Valeurs pondérées
        context = torch.matmul(attention, V)
        
        # Combiner les têtes
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        output = self.fc_out(context)
        
        return output, attention


class ModeleRNN(nn.Module):
    """
    Réseau de neurones récurrent bi-directionnel avancé avec attention.
    
    Architecture:
        - Embedding optionnel
        - Multi-head Attention
        - Bi-directional LSTM empilé avec dropout
        - Batch normalization inter-couches
        - Couches entièrement connectées denses
        - Gestion persistante d'état caché
    
    Args:
        input_size: Dimension des features d'entrée (ou vocab_size si embedding)
        hidden_size: Dimension de l'état caché du LSTM (défaut: 256)
        output_size: Dimension de la sortie
        num_layers: Nombre de couches LSTM (défaut: 3)
        num_heads: Nombre de têtes d'attention (défaut: 4)
        dropout: Taux de dropout (défaut: 0.3)
        bidirectional: LSTM bi-directionnel (défaut: True)
        use_attention: Activer le mécanisme d'attention (défaut: True)
        device: Device PyTorch (cpu ou cuda, défaut: auto)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 256,
        output_size: int = 512,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        device: Optional[str] = None
    ) -> None:
        super(ModeleRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Déterminer le device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # État caché persistant pour interactions autonomes
        self.persistent_h = None
        self.persistent_c = None
        self.state_history: List[Tuple] = []
        
        # Couche d'embedding (optionnelle)
        # Peut être utilisée si input_size < 512
        if input_size < 512:
            self.embedding = nn.Embedding(input_size, 128)
            lstm_input_size = 128
        else:
            self.embedding = None
            lstm_input_size = input_size
        
        # Attention multi-tête
        if self.use_attention:
            self.attention = MultiHeadAttention(lstm_input_size, num_heads)
        
        # Couches LSTM bi-directionnelles
        self.lstm = nn.LSTM(
            lstm_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Batch normalization après LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.batch_norm1 = nn.BatchNorm1d(lstm_output_size)
        
        # Couches fully-connected denses
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Couches d'activation
        self.dropout_layer = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Layer normalization finale
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(
        self, 
        x: torch.Tensor, 
        use_persistent_state: bool = False
    ) -> torch.Tensor:
        """
        Propagation avant à travers le réseau.
        
        Args:
            x: Tensor d'entrée de forme (batch_size, sequence_length, input_size)
               ou (batch_size, sequence_length) si embedding
            use_persistent_state: Utiliser l'état caché persistant (défaut: False)
            
        Returns:
            Tensor de sortie de forme (batch_size, output_size)
        """
        # Déterminer le device réel du modèle (en se basant sur le premier paramètre)
        device = next(self.parameters()).device
        
        # Assurer que l'entrée est sur le bon device
        x = x.to(device)
        
        # Embedding optionnel
        if self.embedding is not None:
            x = self.embedding(x)
        
        # Attention optionnelle
        if self.use_attention:
            x, _ = self.attention(x, x, x)
        
        # États cachés initiaux ou persistants
        if use_persistent_state and self.persistent_h is not None and self.persistent_c is not None:
            h0 = self.persistent_h.to(device)
            c0 = self.persistent_c.to(device)
        else:
            batch_size = x.size(0)
            lstm_hidden_size = self.hidden_size * (2 if self.bidirectional else 1)
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size).to(device)
        
        # Propagation LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Sauvegarder l'état persistant
        self.persistent_h = hn.detach()
        self.persistent_c = cn.detach()
        self.state_history.append((hn.detach(), cn.detach()))
        
        # Limiter l'historique d'état
        if len(self.state_history) > 50:
            self.state_history = self.state_history[-50:]
        
        # Prendre la sortie de la dernière étape temporelle
        out = lstm_out[:, -1, :]
        
        # Batch norm
        out = self.batch_norm1(out)
        
        # Couches fully-connected avec activations
        out = self.activation(self.fc1(out))
        out = self.dropout_layer(out)
        out = self.batch_norm2(out)
        
        out = self.activation(self.fc2(out))
        out = self.dropout_layer(out)
        out = self.batch_norm3(out)
        
        # Sortie finale
        out = self.fc3(out)
        out = self.layer_norm(out)
        
        return out
    
    def reset_persistent_state(self) -> None:
        """Réinitialiser l'état caché persistant."""
        self.persistent_h = None
        self.persistent_c = None
        self.state_history.clear()
    
    def get_state_history(self) -> List[Tuple]:
        """Récupérer l'historique des états cachés."""
        return self.state_history
    
    def set_persistent_state(self, h: torch.Tensor, c: torch.Tensor) -> None:
        """Définir manuellement l'état caché persistant."""
        self.persistent_h = h
        self.persistent_c = c


if __name__ == "__main__":
    # Exemple de création et utilisation du modèle
    print("🧠 Initialisation du Cortex Textuel RNN de NETY...\n")
    
    modele_rnn = ModeleRNN(
        input_size=768,  # Embedding dimension (ex: FastText, Word2Vec)
        hidden_size=256,
        output_size=512,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    )
    
    print(modele_rnn)
    print(f"\n📊 Total de paramètres: {sum(p.numel() for p in modele_rnn.parameters()):,}")
    
    # Test avec une entrée aléatoire
    batch_size, seq_length = 2, 10
    test_input = torch.randn(batch_size, seq_length, 768)
    output = modele_rnn(test_input)
    print(f"✅ Sortie du modèle: {output.shape}")