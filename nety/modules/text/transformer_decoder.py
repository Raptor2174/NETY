import torch
import torch.nn as nn
import math
from typing import Optional


class MiniTransformerDecoder(nn.Module):
    """
    Mini-décodeur Transformer pour générer du texte
    Basé sur l'output du RNN encoder
    
    Architecture: 6 couches, 512 dims, 8 têtes d'attention
    Inspiré de: BART, T5 (Encoder-Decoder)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        # Embedding des tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Couches transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False  # Post-norm comme dans les transformers classiques
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Projection vers vocabulaire
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Initialisation des poids (Xavier uniform)
        self._init_weights()
    
        # Initialisation des poids (Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids pour une meilleure convergence"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        tgt: torch.Tensor,  # Tokens de sortie déjà générés
        memory: torch.Tensor,  # Output du RNN encoder (batch, 1, 512)
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass du décodeur
        
        Args:
            tgt: (batch, seq_len) - tokens déjà générés
            memory: (batch, 1, 512) - contexte du RNN encoder
            tgt_mask: Masque causal pour empêcher de voir le futur
            memory_mask: Masque pour le contexte encoder (généralement None)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Embed et scale
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Positional encoding
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Générer masque causal si non fourni
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Decoder transformer
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask
        )
        
        # Projection vers vocabulaire
        logits = self.fc_out(output)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Génère un masque causal triangulaire supérieur"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(
        self,
        memory: torch.Tensor,
        start_token: int = 1,  # <sos>
        end_token: int = 2,    # <eos>
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Génération autoregressive avec sampling amélioré
        
        Args:
            memory: Contexte du RNN encoder (batch, 1, 512)
            start_token: Token de début de séquence
            end_token: Token de fin de séquence
            max_length: Longueur maximale
            temperature: Contrôle de la diversité (0.1-2.0)
            top_k: Top-K sampling (0 = désactivé)
            top_p: Nucleus sampling (0-1, 0 = désactivé)
        
        Returns:
            tokens: Liste des tokens générés
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Initialiser avec le start token
        tokens = torch.LongTensor([[start_token]]).to(device)
        
        for _ in range(max_length):
            # Générer masque causal
            tgt_mask = self._generate_square_subsequent_mask(tokens.size(1)).to(device)
            
            # Forward pass
            with torch.no_grad():
                logits = self.forward(tokens, memory, tgt_mask)
            
            # Prendre le dernier token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Filtrage top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Filtrage top-p (nucleus)
            if top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Retirer les tokens avec cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Échantillonnage
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Ajouter le token
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Arrêter si EOS
            if next_token.item() == end_token:
                break
        
        return tokens[0].tolist()

class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoïdal standard (Vaswani et al., 2017)"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Créer la matrice d'encodage positionnel
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (max_len, d_model) -> buffer non-entraînable
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (batch, seq_len, d_model)
        Returns:
            Tensor avec positional encoding ajouté
        """
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe_slice = self.pe.data[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pe_slice
        return self.dropout(x)


class HybridRNNTransformer(nn.Module):
    """
    Architecture hybride: RNN Encoder + Transformer Decoder
    
    Architecture:
        Input Message
            ↓
        RNN Encoder (ModeleRNN - 3.5M params)
            ↓ (512 dims contextualisé)
        Mini-Transformer Decoder (6 couches, 512 dims)
            ↓
        Token Generation (autorégressif)
            ↓
        Response Text
    
    Inspiré de: BART, T5, MarianMT (Encoder-Decoder)
    """
    
    def __init__(
        self,
        vocab_size: int,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 3,
        rnn_num_heads: int = 4,
        decoder_d_model: int = 512,
        decoder_nhead: int = 8,
        decoder_num_layers: int = 6,
        decoder_dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        super().__init__()
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Importer ModeleRNN
        from nety.modules.text.modele_rnn import ModeleRNN
        
        # 1. Embedding pour le RNN (convertit tokens → embeddings)
        self.rnn_embedding = nn.Embedding(vocab_size, decoder_d_model)
        
        # 2. RNN Encoder (existant)
        self.rnn_encoder = ModeleRNN(
            input_size=decoder_d_model,  # Prend les embeddings, pas vocab_size
            hidden_size=rnn_hidden_size,
            output_size=decoder_d_model,  # 512 dims
            num_layers=rnn_num_layers,
            num_heads=rnn_num_heads,
            dropout=dropout,
            bidirectional=True,
            use_attention=True,
            device=str(self.device)
        )
        
        # 3. Transformer Decoder (nouveau)
        self.transformer_decoder = MiniTransformerDecoder(
            vocab_size=vocab_size,
            d_model=decoder_d_model,
            nhead=decoder_nhead,
            num_layers=decoder_num_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        self.decoder_d_model = decoder_d_model
        
        # Compter les paramètres
        self.num_params = sum(p.numel() for p in self.parameters())
        
        print(f"🔥 HybridRNNTransformer initialisé")
        print(f"   ├─ RNN Encoder: {sum(p.numel() for p in self.rnn_encoder.parameters()):,} params")
        print(f"   ├─ Transformer Decoder: {sum(p.numel() for p in self.transformer_decoder.parameters()):,} params")
        print(f"   ├─ Total: {self.num_params:,} params")
        print(f"   ├─ Device: {self.device}")
        print(f"   └─ Architecture: RNN(3.5M) → Transformer(6L, 512D, 8H)")
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encode le message d'entrée avec le RNN
        
        Args:
            src: (batch, seq_len) - tokens d'entrée (indices)
        
        Returns:
            memory: (batch, 1, 512) - contexte encodé
        """
        # Convertir tokens → embeddings
        src_emb = self.rnn_embedding(src)  # (batch, seq_len, 512)
        
        # RNN encode
        encoded = self.rnn_encoder(src_emb)  # (batch, 512)
        
        # Ajouter dimension séquentielle pour le transformer
        memory = encoded.unsqueeze(1)  # (batch, 1, 512)
        
        return memory
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Décode avec le transformer
        
        Args:
            tgt: (batch, seq_len) - tokens déjà générés
            memory: (batch, 1, 512) - contexte du RNN
            tgt_mask: Masque causal
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.transformer_decoder(tgt, memory, tgt_mask)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward complet: encode + decode
        
        Args:
            src: (batch, seq_len) - message d'entrée
            tgt: (batch, seq_len) - tokens de sortie (pour teacher forcing)
            tgt_mask: Masque causal
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        memory = self.encode(src)
        logits = self.decode(tgt, memory, tgt_mask)
        return logits
    
    def generate(
        self,
        src: torch.Tensor,
        start_token: int = 1,
        end_token: int = 2,
        max_length: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """
        Génération autoregressive complète
        
        Args:
            src: (batch, seq_len) - message d'entrée
            start_token: Token <sos>
            end_token: Token <eos>
            max_length: Longueur max
            temperature: Contrôle diversité
            top_k: Top-K sampling
            top_p: Nucleus sampling
        
        Returns:
            tokens: Liste des tokens générés
        """
        # Encoder le message
        memory = self.encode(src)
        
        # Générer avec le transformer decoder
        tokens = self.transformer_decoder.generate(
            memory=memory,
            start_token=start_token,
            end_token=end_token,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        return tokens
    
    def reset_rnn_state(self):
        """Réinitialiser l'état caché du RNN"""
        self.rnn_encoder.reset_persistent_state()


