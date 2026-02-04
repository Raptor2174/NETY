"""
NETY V2-Maxx Brain - Modèle Unifié
===================================

Architecture complète du cerveau de NETY V2-Maxx :
Input → Embedding → Cognitive Layer → Limbic System → RNN Encoder → RNN Decoder → Output

Ce modèle unifie :
- TextualCortex (RNN bi-directionnel)
- HybridRNNTransformer (RNN Encoder + Transformer Decoder)

En un seul modèle cohérent optimisé pour RTX 3060 (12GB VRAM).
Paramètres cibles : 100-200M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class NETYBrainConfig:
    """Configuration du cerveau NETY V2"""
    # Vocabulaire
    vocab_size: int = 50000
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2
    
    # Embeddings
    embedding_dim: int = 512
    max_seq_length: int = 256
    
    # Cognitive Layer (Transformer Encoder)
    cognitive_num_layers: int = 4
    cognitive_num_heads: int = 8
    cognitive_dim_feedforward: int = 2048
    cognitive_dropout: float = 0.1
    
    # Limbic System
    limbic_hidden_dim: int = 256
    limbic_num_emotions: int = 6
    limbic_dropout: float = 0.2
    
    # RNN Encoder
    rnn_encoder_hidden_dim: int = 512
    rnn_encoder_num_layers: int = 3
    rnn_encoder_bidirectional: bool = True
    rnn_encoder_dropout: float = 0.3
    
    # RNN Decoder
    rnn_decoder_hidden_dim: int = 512
    rnn_decoder_num_layers: int = 3
    rnn_decoder_dropout: float = 0.3
    rnn_decoder_attention: bool = True


class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal pour Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer matrice d'encodage positionnel
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de shape (seq_len, batch, embedding_dim) ou (batch, seq_len, embedding_dim)
        """
        if x.dim() == 3 and x.size(0) != self.pe.size(0):
            # Format (batch, seq_len, embedding_dim)
            x = x + self.pe[:x.size(1), 0, :].unsqueeze(0)
        else:
            # Format (seq_len, batch, embedding_dim)
            x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)


class CognitiveLayer(nn.Module):
    """
    Cognitive Layer - Transformer Encoder pour le raisonnement
    
    Traite l'input avec attention multi-head pour extraire des représentations
    sémantiques riches avant le traitement émotionnel et la génération.
    """
    
    def __init__(self, config: NETYBrainConfig):
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.embedding_dim,
            dropout=config.cognitive_dropout,
            max_len=config.max_seq_length
        )
        
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.cognitive_num_heads,
            dim_feedforward=config.cognitive_dim_feedforward,
            dropout=config.cognitive_dropout,
            batch_first=True,  # (batch, seq, feature)
            norm_first=False  # Post-norm (classique)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.cognitive_num_layers
        )
        
        # Layer normalization finale
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embedding_dim)
            src_key_padding_mask: (batch, seq_len) - True pour positions padding
        
        Returns:
            output: (batch, seq_len, embedding_dim) - représentation cognitive
        """
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Layer norm
        output = self.layer_norm(output)
        
        return output


class LimbicSystem(nn.Module):
    """
    Limbic System - Filtrage et modulation émotionnelle
    
    Analyse l'état émotionnel du contexte et module les représentations
    pour influencer la génération de manière émotionnellement cohérente.
    """
    
    def __init__(self, config: NETYBrainConfig):
        super().__init__()
        self.config = config
        
        # Encoder émotionnel
        self.emotion_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.limbic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.limbic_dropout),
            nn.Linear(config.limbic_hidden_dim, config.limbic_hidden_dim),
            nn.ReLU()
        )
        
        # Classifieur émotionnel (6 émotions : joie, tristesse, colère, peur, surprise, neutre)
        self.emotion_classifier = nn.Linear(config.limbic_hidden_dim, config.limbic_num_emotions)
        
        # Modulation gate (influence sur les features)
        self.modulation_gate = nn.Sequential(
            nn.Linear(config.limbic_hidden_dim, config.embedding_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, embedding_dim) - output de Cognitive Layer
        
        Returns:
            modulated: (batch, seq_len, embedding_dim) - représentation modulée
            emotion_logits: (batch, num_emotions) - prédiction émotionnelle
        """
        # Pooling pour obtenir représentation globale
        pooled = x.mean(dim=1)  # (batch, embedding_dim)
        
        # Encoder émotionnel
        emotion_features = self.emotion_encoder(pooled)  # (batch, limbic_hidden_dim)
        
        # Prédiction émotionnelle
        emotion_logits = self.emotion_classifier(emotion_features)  # (batch, num_emotions)
        
        # Gate de modulation
        gate = self.modulation_gate(emotion_features)  # (batch, embedding_dim)
        
        # Appliquer modulation à chaque timestep
        modulated = x * gate.unsqueeze(1)  # Broadcasting sur seq_len
        
        return modulated, emotion_logits


class AttentionMechanism(nn.Module):
    """Mécanisme d'attention pour le décodeur RNN"""
    
    def __init__(self, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (batch, decoder_hidden_dim)
            encoder_outputs: (batch, seq_len, encoder_hidden_dim)
            mask: (batch, seq_len) - True pour positions à masquer
        
        Returns:
            context: (batch, encoder_hidden_dim)
            attention_weights: (batch, seq_len)
        """
        batch_size, seq_len, encoder_dim = encoder_outputs.size()
        
        # Répéter hidden pour chaque position encoder
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
        
        # Concaténer et calculer scores
        energy = torch.tanh(self.attention(
            torch.cat([hidden_repeated, encoder_outputs], dim=2)
        ))  # (batch, seq_len, hidden_dim)
        
        attention_scores = self.v(energy).squeeze(2)  # (batch, seq_len)
        
        # Masquer si nécessaire
        if mask is not None:
            # S'assurer que le masque a la bonne shape (batch, seq_len)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # (seq_len,) -> (1, seq_len)
            # Broadcaster le masque si nécessaire
            if mask.size(0) == 1 and attention_scores.size(0) > 1:
                mask = mask.expand(attention_scores.size(0), -1)
            # Vérifier que les dimensions correspondent
            if mask.size(1) != attention_scores.size(1):
                # Pad ou truncate le masque
                if mask.size(1) < attention_scores.size(1):
                    pad_size = attention_scores.size(1) - mask.size(1)
                    mask = torch.cat([mask, torch.ones_like(mask[:, :pad_size])], dim=1)
                else:
                    mask = mask[:, :attention_scores.size(1)]
            attention_scores = attention_scores.masked_fill(mask, -1e10)
        
        # Softmax pour obtenir poids d'attention
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Contexte pondéré
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (batch, encoder_dim)
        
        return context, attention_weights


class NETYBrainV2(nn.Module):
    """
    NETY Brain V2-Maxx - Modèle Unifié Complet
    
    Pipeline :
    1. Input Embedding
    2. Cognitive Layer (Transformer Encoder) - Raisonnement
    3. Limbic System - Modulation émotionnelle
    4. RNN Encoder (Bi-LSTM) - Encodage séquentiel
    5. RNN Decoder (LSTM + Attention) - Génération autoregressive
    6. Output Projection
    
    Paramètres : ~122M
    Optimisé pour RTX 3060 (12GB VRAM)
    """
    
    def __init__(self, config: Optional[NETYBrainConfig] = None):
        super().__init__()
        
        self.config = config or NETYBrainConfig()
        
        # 1. Input Embedding
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.config.pad_token_id
        )
        
        # Dropout sur embeddings
        self.embedding_dropout = nn.Dropout(0.1)
        
        # 2. Cognitive Layer (Transformer Encoder)
        self.cognitive_layer = CognitiveLayer(self.config)
        
        # 3. Limbic System
        self.limbic_system = LimbicSystem(self.config)
        
        # 4. RNN Encoder (Bi-LSTM)
        self.rnn_encoder = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.rnn_encoder_hidden_dim,
            num_layers=self.config.rnn_encoder_num_layers,
            dropout=self.config.rnn_encoder_dropout if self.config.rnn_encoder_num_layers > 1 else 0,
            bidirectional=self.config.rnn_encoder_bidirectional,
            batch_first=True
        )
        
        # Projection pour combiner directions (si bidirectionnel)
        encoder_output_dim = (
            self.config.rnn_encoder_hidden_dim * 2
            if self.config.rnn_encoder_bidirectional
            else self.config.rnn_encoder_hidden_dim
        )
        
        self.encoder_projection = nn.Linear(encoder_output_dim, self.config.rnn_decoder_hidden_dim)
        
        # 5. RNN Decoder (LSTM)
        self.rnn_decoder = nn.LSTM(
            input_size=self.config.embedding_dim + (
                self.config.rnn_decoder_hidden_dim if self.config.rnn_decoder_attention else 0
            ),
            hidden_size=self.config.rnn_decoder_hidden_dim,
            num_layers=self.config.rnn_decoder_num_layers,
            dropout=self.config.rnn_decoder_dropout if self.config.rnn_decoder_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        if self.config.rnn_decoder_attention:
            self.attention = AttentionMechanism(
                hidden_dim=self.config.rnn_decoder_hidden_dim,
                encoder_dim=self.config.rnn_decoder_hidden_dim
            )
        
        # 6. Output Projection
        self.output_projection = nn.Linear(
            self.config.rnn_decoder_hidden_dim,
            self.config.vocab_size
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation Xavier pour meilleure convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Encode l'input à travers Cognitive Layer, Limbic System et RNN Encoder
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 pour tokens réels, 0 pour padding
        
        Returns:
            encoder_outputs: (batch, seq_len, rnn_decoder_hidden_dim)
            encoder_hidden: Tuple (h, c) pour LSTM decoder
            emotion_logits: (batch, num_emotions)
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        # 2. Cognitive Layer
        # Créer padding mask pour Transformer (True = padding)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None
        
        cognitive_output = self.cognitive_layer(embedded, padding_mask)  # (batch, seq_len, embedding_dim)
        
        # 3. Limbic System
        limbic_output, emotion_logits = self.limbic_system(cognitive_output)  # (batch, seq_len, embedding_dim)
        
        # 4. RNN Encoder
        if attention_mask is not None:
            # Pack pour efficacité (ignorer padding)
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                limbic_output, lengths, batch_first=True, enforce_sorted=False
            )
            encoder_outputs_packed, (h_n, c_n) = self.rnn_encoder(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs_packed, batch_first=True
            )
        else:
            encoder_outputs, (h_n, c_n) = self.rnn_encoder(limbic_output)
        
        # Projeter encoder outputs
        encoder_outputs = self.encoder_projection(encoder_outputs)  # (batch, seq_len, decoder_hidden)
        
        # Préparer hidden state pour decoder
        # h_n shape: (num_layers * num_directions, batch, hidden)
        if self.config.rnn_encoder_bidirectional:
            # Combiner forward et backward
            num_layers = self.config.rnn_encoder_num_layers
            h_n = h_n.view(num_layers, 2, batch_size, -1)  # (layers, 2, batch, hidden)
            c_n = c_n.view(num_layers, 2, batch_size, -1)
            
            # Concaténer et projeter
            h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)  # (layers, batch, hidden*2)
            c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)
            
            # Projeter vers decoder hidden dim si nécessaire
            if h_n.size(2) != self.config.rnn_decoder_hidden_dim:
                h_n = self.encoder_projection(h_n.transpose(0, 1)).transpose(0, 1)
                c_n = self.encoder_projection(c_n.transpose(0, 1)).transpose(0, 1)
        
        # Ajuster nombre de couches si nécessaire
        if self.config.rnn_encoder_num_layers != self.config.rnn_decoder_num_layers:
            # Répéter ou tronquer
            if self.config.rnn_encoder_num_layers < self.config.rnn_decoder_num_layers:
                repeat_factor = self.config.rnn_decoder_num_layers // self.config.rnn_encoder_num_layers
                h_n = h_n.repeat(repeat_factor, 1, 1)
                c_n = c_n.repeat(repeat_factor, 1, 1)
            else:
                h_n = h_n[:self.config.rnn_decoder_num_layers]
                c_n = c_n[:self.config.rnn_decoder_num_layers]
        
        return encoder_outputs, (h_n, c_n), emotion_logits
    
    def decode_step(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """
        Un step de décodage autorégressif
        
        Args:
            decoder_input: (batch, 1) - token à décoder
            decoder_hidden: (h, c) pour LSTM
            encoder_outputs: (batch, seq_len, decoder_hidden_dim)
            encoder_mask: (batch, seq_len) - 0 pour padding
        
        Returns:
            output_logits: (batch, vocab_size)
            decoder_hidden: (h, c) mis à jour
            attention_weights: (batch, seq_len) si attention activée
        """
        # Embed decoder input
        embedded = self.embedding(decoder_input)  # (batch, 1, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        
        attention_weights = None
        
        # Attention si activée
        if self.config.rnn_decoder_attention:
            # Utiliser dernier hidden state pour attention
            query = decoder_hidden[0][-1]  # (batch, decoder_hidden_dim)
            
            # Créer mask pour attention (True = padding)
            if encoder_mask is not None:
                # S'assurer que encoder_mask a la bonne shape (batch, seq_len)
                if encoder_mask.dim() == 1:
                    encoder_mask = encoder_mask.unsqueeze(0)
                attn_mask = (encoder_mask == 0)  # True pour les tokens à ignorer (padding)
            else:
                attn_mask = None
            
            context, attention_weights = self.attention(
                query, encoder_outputs, attn_mask
            )  # (batch, decoder_hidden_dim), (batch, seq_len)
            
            # Concaténer embedding et context
            decoder_input_combined = torch.cat([
                embedded,
                context.unsqueeze(1)
            ], dim=2)  # (batch, 1, embedding_dim + decoder_hidden_dim)
        else:
            decoder_input_combined = embedded
        
        # LSTM decoder
        output, decoder_hidden = self.rnn_decoder(decoder_input_combined, decoder_hidden)
        # output: (batch, 1, decoder_hidden_dim)
        
        # Project to vocabulary
        output_logits = self.output_projection(output.squeeze(1))  # (batch, vocab_size)
        
        return output_logits, decoder_hidden, attention_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        input_attention_mask: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        
        """
        Forward pass complet (mode training)
        
        Args:
            input_ids: (batch, input_seq_len)
            target_ids: (batch, target_seq_len) - pour teacher forcing
            input_attention_mask: (batch, input_seq_len)
            target_attention_mask: (batch, target_seq_len)
            teacher_forcing_ratio: Probabilité d'utiliser target réel vs prédiction
        
        Returns:
            dict avec:
                - logits: (batch, target_seq_len, vocab_size)
                - emotion_logits: (batch, num_emotions)
                - attention_weights: (batch, target_seq_len, input_seq_len) si attention
        """
        batch_size = input_ids.size(0)
        
        # Encoder
        encoder_outputs, encoder_hidden, emotion_logits = self.encode(
            input_ids, input_attention_mask
        )
        
        # Si pas de target, mode inference
        if target_ids is None:
            return {
                'encoder_outputs': encoder_outputs,
                'encoder_hidden': encoder_hidden,
                'emotion_logits': emotion_logits
            }
        
        target_seq_len = target_ids.size(1)
        
        # Préparer tenseur de sortie
        all_logits = []
        all_attention_weights = [] if self.config.rnn_decoder_attention else None
        
        # Initialiser decoder input avec SOS token
        decoder_input = torch.full(
            (batch_size, 1),
            self.config.sos_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        
        decoder_hidden = encoder_hidden
        
        # Boucle de décodage
        for t in range(target_seq_len):
            # Decode step
            output_logits, decoder_hidden, attention_weights = self.decode_step(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                input_attention_mask
            )
            
            all_logits.append(output_logits)
            if attention_weights is not None and all_attention_weights is not None:
                all_attention_weights.append(attention_weights)
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and t < target_seq_len - 1:
                decoder_input = target_ids[:, t:t+1]
            else:
                decoder_input = output_logits.argmax(dim=1, keepdim=True)
        
        # Stack results
        logits = torch.stack(all_logits, dim=1)  # (batch, target_seq_len, vocab_size)
        
        result = {
            'logits': logits,
            'emotion_logits': emotion_logits
        }
        
        if all_attention_weights is not None:
            result['attention_weights'] = torch.stack(all_attention_weights, dim=1)
        
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Génération autoregressive avec sampling
        
        Args:
            input_ids: (batch, seq_len)
            max_length: Longueur maximale à générer
            temperature: Contrôle de la créativité
            top_k: Top-K sampling
            top_p: Nucleus sampling
            repetition_penalty: Pénalité de répétition
            eos_token_id: Token de fin
            pad_token_id: Token de padding
        
        Returns:
            generated: (batch, max_length)
        """
        self.eval()
        
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Encode input
        with torch.no_grad():
            encoder_outputs, encoder_hidden, _ = self.encode(input_ids)
        
        # Initialiser génération
        generated = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
            device=device
        )
        
        decoder_input = torch.full(
            (batch_size, 1),
            self.config.sos_token_id,
            dtype=torch.long,
            device=device
        )
        
        decoder_hidden = encoder_hidden
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for t in range(max_length):
            with torch.no_grad():
                output_logits, decoder_hidden, _ = self.decode_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
            
            # Appliquer température
            logits = output_logits / temperature
            
            # Repetition penalty
            if t > 0:
                for i in range(batch_size):
                    for token_id in set(generated[i, :t].tolist()):
                        if token_id != pad_token_id:
                            logits[i, token_id] /= repetition_penalty
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Update generated
            generated[:, t] = next_token.squeeze(1)
            
            # Check for EOS
            finished |= (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break
            
            # Next input
            decoder_input = next_token
        
        return generated
    
    def count_parameters(self) -> int:
        """Compte le nombre total de paramètres"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Compte le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    print("=" * 80)
    print("Test de NETYBrainV2")
    print("=" * 80)
    
    # Créer config
    config = NETYBrainConfig()
    
    # Créer modèle
    model = NETYBrainV2(config)
    
    print(f"\n✓ Modèle créé")
    print(f"  Paramètres totaux: {model.count_parameters():,}")
    print(f"  Paramètres entraînables: {model.count_trainable_parameters():,}")
    
    # Test forward
    batch_size = 2
    input_seq_len = 10
    target_seq_len = 15
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, input_seq_len))
    target_ids = torch.randint(0, config.vocab_size, (batch_size, target_seq_len))
    
    print(f"\n✓ Test forward pass")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {target_ids.shape}")
    
    output = model(input_ids, target_ids)
    
    print(f"\n✓ Output")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Emotion logits shape: {output['emotion_logits'].shape}")
    if 'attention_weights' in output:
        print(f"  Attention weights shape: {output['attention_weights'].shape}")
    
    # Test génération
    print(f"\n✓ Test génération")
    generated = model.generate(input_ids, max_length=20)
    print(f"  Generated shape: {generated.shape}")
    
    print("\n" + "=" * 80)
    print("✓ Tous les tests passés!")
    print("=" * 80)
