"""
NETY V2-Maxx - Configuration Centralisée
=========================================

Configuration complète du système NETY V2-Maxx optimisée pour RTX 3060 (12GB VRAM).
Tous les hyperparamètres et contraintes matérielles sont centralisés ici.

Architecture: Input → Embedding → Cognitive Layer → Limbic System → RNN Encoder → RNN Decoder → Output
Modèle Unifié: 100-200M paramètres
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class HardwareConfig:
    """Configuration matérielle - RTX 3060 avec limitations mémoire"""
    
    # GPU Configuration
    gpu_name: str = "RTX 3060"
    total_vram_gb: int = 12
    max_vram_usage_gb: float = 10.0  # Limite de sécurité
    enable_mixed_precision: bool = True  # FP16 pour économiser VRAM
    
    # CPU/RAM Configuration
    cpu_name: str = "Intel Core i5"
    max_ram_usage_gb: float = 10.0
    num_workers: int = 4  # DataLoader workers
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Storage
    max_storage_gb: int = 30  # Projet + checkpoints + datasets
    
    def __post_init__(self):
        """Validation et ajustements automatiques"""
        if self.device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  CUDA non disponible, passage en CPU mode")
                self.device = "cpu"
                self.enable_mixed_precision = False
            else:
                gpu_props = torch.cuda.get_device_properties(0)
                actual_vram_gb = gpu_props.total_memory / (1024**3)
                print(f"✓ GPU détecté: {gpu_props.name} ({actual_vram_gb:.1f} GB VRAM)")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Statistiques mémoire GPU actuelle"""
        if self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
        return {}


@dataclass
class ModelConfig:
    """Configuration du modèle unifié NETYBrainV2"""
    
    # Vocabulaire
    vocab_size: int = 50000  # 50k tokens (compromis qualité/taille)
    pad_token_id: int = 0
    sos_token_id: int = 1  # Start of Sequence
    eos_token_id: int = 2  # End of Sequence
    unk_token_id: int = 3  # Unknown token
    
    # Embeddings
    embedding_dim: int = 512  # Dimension des embeddings de tokens
    max_seq_length: int = 256  # Longueur maximale de séquence
    
    # Cognitive Layer (Transformer Encoder pour raisonnement)
    cognitive_num_layers: int = 4  # 4 couches de Transformer Encoder
    cognitive_num_heads: int = 8  # 8 têtes d'attention multi-head
    cognitive_dim_feedforward: int = 2048  # FFN dimension
    cognitive_dropout: float = 0.1
    
    # Limbic System (Filtrage émotionnel)
    limbic_hidden_dim: int = 256  # Dimension cachée du système limbique
    limbic_num_emotions: int = 6  # Nombre d'émotions de base (joie, tristesse, colère, peur, surprise, neutre)
    limbic_dropout: float = 0.2
    
    # RNN Encoder (Bi-directional LSTM)
    rnn_encoder_hidden_dim: int = 512  # Dimension cachée RNN encoder
    rnn_encoder_num_layers: int = 3  # 3 couches LSTM
    rnn_encoder_bidirectional: bool = True  # Bi-directionnel
    rnn_encoder_dropout: float = 0.3
    
    # RNN Decoder (LSTM autorégressif pour génération)
    rnn_decoder_hidden_dim: int = 512  # Dimension cachée RNN decoder
    rnn_decoder_num_layers: int = 3  # 3 couches LSTM
    rnn_decoder_dropout: float = 0.3
    rnn_decoder_attention: bool = True  # Attention sur encoder outputs
    
    # Calcul du nombre de paramètres (estimation)
    def estimate_parameters(self) -> int:
        """Estime le nombre total de paramètres du modèle"""
        params = 0
        
        # Embeddings
        params += self.vocab_size * self.embedding_dim  # ~25M
        
        # Cognitive Layer (Transformer Encoder)
        # Formule: L * (12 * d^2 + 13 * d) où L=couches, d=embedding_dim
        d = self.embedding_dim
        L = self.cognitive_num_layers
        params += L * (12 * d**2 + 13 * d)  # ~25M
        
        # Limbic System
        params += (self.embedding_dim * self.limbic_hidden_dim * 2 +
                  self.limbic_hidden_dim * self.limbic_num_emotions)  # ~0.3M
        
        # RNN Encoder (Bi-LSTM)
        # Formule: 4 * num_layers * (hidden^2 + hidden * input + hidden)
        encoder_hidden = self.rnn_encoder_hidden_dim
        encoder_layers = self.rnn_encoder_num_layers
        encoder_params = 4 * encoder_layers * (encoder_hidden**2 + encoder_hidden * self.embedding_dim + encoder_hidden)
        if self.rnn_encoder_bidirectional:
            encoder_params *= 2  # Double params for bidirectional
        params += encoder_params  # ~20M
        
        # RNN Decoder (LSTM)
        decoder_hidden = self.rnn_decoder_hidden_dim
        decoder_layers = self.rnn_decoder_num_layers
        params += 4 * decoder_layers * (decoder_hidden**2 + decoder_hidden * self.embedding_dim + decoder_hidden)  # ~10M
        
        # Attention Mechanism
        if self.rnn_decoder_attention:
            params += encoder_hidden * decoder_hidden * 3  # Q, K, V projections (~1M)
        
        # Output Projection
        params += self.rnn_decoder_hidden_dim * self.vocab_size  # ~25M
        
        return params
    
    def estimate_vram_usage_gb(self, batch_size: int) -> float:
        """Estime l'usage VRAM en GB pour un batch donné"""
        # Paramètres du modèle en FP16
        params = self.estimate_parameters()
        model_size_gb = (params * 2) / (1024**3)  # 2 bytes per FP16 param
        
        # Activations (approximation)
        activations_per_sample = self.max_seq_length * self.embedding_dim * 10  # Factor 10 pour toutes les couches
        activations_size_gb = (batch_size * activations_per_sample * 2) / (1024**3)
        
        # Gradients (même taille que les paramètres)
        gradients_size_gb = model_size_gb
        
        # Optimizer states (Adam: 2x les paramètres)
        optimizer_size_gb = model_size_gb * 2
        
        # Total
        total_gb = model_size_gb + activations_size_gb + gradients_size_gb + optimizer_size_gb
        
        return total_gb


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement"""
    
    # Hyperparamètres
    learning_rate: float = 5e-4  # Learning rate initial
    min_learning_rate: float = 1e-6  # LR minimal pour scheduler
    weight_decay: float = 0.01  # L2 regularization
    
    # Batch et accumulation
    batch_size: int = 16  # Batch size optimal pour RTX 3060
    gradient_accumulation_steps: int = 2  # Effective batch = 32
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Epochs et steps
    num_epochs: int = 50  # Nombre d'epochs
    warmup_steps: int = 1000  # Warmup steps pour LR scheduler
    eval_steps: int = 500  # Évaluation toutes les N steps
    save_steps: int = 1000  # Sauvegarde toutes les N steps
    
    # Early stopping
    early_stopping_patience: int = 5  # Patience (en epochs)
    early_stopping_min_delta: float = 0.001  # Amélioration minimale
    
    # Optimizer
    optimizer_type: str = "adamw"  # AdamW optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler_type: str = "cosine_with_warmup"  # Cosine annealing with warmup
    
    # Mixed Precision
    use_amp: bool = True  # Automatic Mixed Precision (FP16)
    
    # Logging
    log_interval: int = 10  # Log toutes les N steps
    
    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3  # Garder les 3 derniers checkpoints
    
    def get_effective_batch_size(self) -> int:
        """Batch size effectif avec gradient accumulation"""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class MemoryConfig:
    """Configuration du système de mémoire pondérée"""
    
    # Capacité
    max_memories: int = 10000  # Maximum 10k souvenirs
    memory_embedding_dim: int = 512  # Dimension des embeddings de mémoire
    
    # Importance
    importance_threshold: float = 0.3  # Seuil minimum d'importance
    importance_decay_rate: float = 0.95  # Décroissance exponentielle (par jour)
    
    # Decay temporel
    temporal_decay_enabled: bool = True
    temporal_decay_halflife_days: float = 30.0  # Demi-vie de 30 jours
    
    # Consolidation
    consolidation_enabled: bool = True
    consolidation_similarity_threshold: float = 0.85  # Cosine similarity
    consolidation_interval_hours: float = 24.0  # Consolidation quotidienne
    
    # Retrieval
    retrieval_top_k: int = 5  # Nombre de souvenirs à récupérer
    retrieval_similarity_threshold: float = 0.5  # Seuil de similarité
    
    # Renforcement
    access_boost: float = 1.2  # Multiplier importance quand accédé
    max_importance: float = 10.0  # Importance maximum
    
    # Stockage
    memory_db_path: str = "data/memory/memories.db"
    memory_index_type: str = "faiss"  # FAISS pour recherche rapide


@dataclass
class GenerationConfig:
    """Configuration de la génération de texte"""
    
    # Stratégie de décodage
    decoding_strategy: str = "beam_search"  # "greedy", "beam_search", "sampling", "nucleus"
    
    # Beam Search
    beam_size: int = 5  # Nombre de beams
    beam_length_penalty: float = 0.6  # Pénalité de longueur
    
    # Sampling
    temperature: float = 0.8  # Température (0.0 = déterministe, >1.0 = créatif)
    top_k: int = 50  # Top-K sampling
    top_p: float = 0.9  # Nucleus sampling (top-p)
    
    # Longueur
    min_length: int = 5  # Longueur minimale de génération
    max_length: int = 128  # Longueur maximale de génération
    
    # Contrôle de répétition
    repetition_penalty: float = 1.2  # Pénalité de répétition
    no_repeat_ngram_size: int = 3  # Éviter répétition de n-grams
    
    # Early stopping
    early_stopping: bool = True  # Arrêt dès qu'un beam est complet
    
    # Diversité
    num_return_sequences: int = 1  # Nombre de séquences à retourner
    do_sample: bool = True  # Activer le sampling (vs greedy)
    
    # Performance
    max_inference_time_ms: float = 1000.0  # Max 1s par réponse


@dataclass
class DataConfig:
    """Configuration des datasets et preprocessing"""
    
    # Paths
    train_data_path: str = "data/training/conversations.json"
    val_data_path: str = "data/validation/conversations.json"
    test_data_path: str = "data/test/conversations.json"
    
    # Tokenizer
    tokenizer_type: str = "bpe"  # "bpe", "wordpiece", "sentencepiece"
    tokenizer_path: str = "data/tokenizer/nety_tokenizer.json"
    vocab_path: str = "data/tokenizer/vocab.txt"
    
    # Dataset splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Preprocessing
    lowercase: bool = True
    remove_accents: bool = False
    max_input_length: int = 256
    max_target_length: int = 128
    
    # Augmentation
    enable_augmentation: bool = False
    augmentation_prob: float = 0.1


@dataclass
class NETYSettings:
    """Configuration complète de NETY V2-Maxx"""
    
    # Sous-configurations
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Métadonnées
    version: str = "2.0-Maxx"
    project_name: str = "NETY V2-Maxx"
    description: str = "Modèle RNN local intelligent pour génération de conversations naturelles"
    
    def __post_init__(self):
        """Validation et ajustements automatiques après initialisation"""
        # Vérifier que le batch size ne dépasse pas la VRAM
        vram_usage = self.model.estimate_vram_usage_gb(self.training.batch_size)
        if vram_usage > self.hardware.max_vram_usage_gb:
            print(f"⚠️  VRAM usage estimé ({vram_usage:.1f} GB) dépasse la limite ({self.hardware.max_vram_usage_gb} GB)")
            print(f"   Réduction du batch size recommandée")
        
        # Créer les répertoires nécessaires
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.memory.memory_db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.data.train_data_path), exist_ok=True)
    
    def print_summary(self):
        """Affiche un résumé complet de la configuration"""
        print("=" * 80)
        print(f"  {self.project_name} v{self.version}")
        print("=" * 80)
        print(f"  {self.description}")
        print("=" * 80)
        
        print("\n📊 HARDWARE")
        print(f"  GPU: {self.hardware.gpu_name} ({self.hardware.total_vram_gb} GB VRAM)")
        print(f"  Device: {self.hardware.device}")
        print(f"  Mixed Precision: {'✓ Enabled (FP16)' if self.hardware.enable_mixed_precision else '✗ Disabled'}")
        print(f"  Max VRAM Usage: {self.hardware.max_vram_usage_gb} GB")
        print(f"  Max RAM Usage: {self.hardware.max_ram_usage_gb} GB")
        
        print("\n🧠 MODEL ARCHITECTURE")
        print(f"  Vocabulary: {self.model.vocab_size:,} tokens")
        print(f"  Embedding Dim: {self.model.embedding_dim}")
        print(f"  Max Sequence Length: {self.model.max_seq_length}")
        print(f"  Cognitive Layer: {self.model.cognitive_num_layers} Transformer layers, {self.model.cognitive_num_heads} heads")
        print(f"  Limbic System: {self.model.limbic_hidden_dim} hidden, {self.model.limbic_num_emotions} emotions")
        print(f"  RNN Encoder: {self.model.rnn_encoder_num_layers} layers, {self.model.rnn_encoder_hidden_dim} hidden, {'bi-directional' if self.model.rnn_encoder_bidirectional else 'uni-directional'}")
        print(f"  RNN Decoder: {self.model.rnn_decoder_num_layers} layers, {self.model.rnn_decoder_hidden_dim} hidden")
        
        total_params = self.model.estimate_parameters()
        print(f"\n  📈 Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        vram_usage = self.model.estimate_vram_usage_gb(self.training.batch_size)
        print(f"  💾 Estimated VRAM: {vram_usage:.2f} GB (batch_size={self.training.batch_size})")
        
        print("\n🎓 TRAINING")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Gradient Accumulation: {self.training.gradient_accumulation_steps} steps")
        print(f"  Effective Batch Size: {self.training.get_effective_batch_size()}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Optimizer: {self.training.optimizer_type}")
        print(f"  Scheduler: {self.training.scheduler_type}")
        print(f"  Mixed Precision: {'✓ Enabled' if self.training.use_amp else '✗ Disabled'}")
        
        print("\n🧠 MEMORY SYSTEM")
        print(f"  Max Memories: {self.memory.max_memories:,}")
        print(f"  Importance Threshold: {self.memory.importance_threshold}")
        print(f"  Temporal Decay: {'✓ Enabled' if self.memory.temporal_decay_enabled else '✗ Disabled'}")
        print(f"  Consolidation: {'✓ Enabled' if self.memory.consolidation_enabled else '✗ Disabled'}")
        print(f"  Retrieval Top-K: {self.memory.retrieval_top_k}")
        
        print("\n✨ GENERATION")
        print(f"  Strategy: {self.generation.decoding_strategy}")
        print(f"  Beam Size: {self.generation.beam_size}")
        print(f"  Temperature: {self.generation.temperature}")
        print(f"  Top-K: {self.generation.top_k}, Top-P: {self.generation.top_p}")
        print(f"  Max Length: {self.generation.max_length}")
        print(f"  Repetition Penalty: {self.generation.repetition_penalty}")
        print(f"  Max Inference Time: {self.generation.max_inference_time_ms} ms")
        
        print("\n📁 DATA")
        print(f"  Train: {self.data.train_data_path}")
        print(f"  Tokenizer: {self.data.tokenizer_type}")
        print(f"  Vocab Path: {self.data.vocab_path}")
        
        print("\n" + "=" * 80)
        
        # Warnings
        if vram_usage > self.hardware.max_vram_usage_gb:
            print("⚠️  WARNING: VRAM usage dépasse la limite!")
            print(f"   Réduire batch_size à {int(self.training.batch_size * self.hardware.max_vram_usage_gb / vram_usage)} recommandé")
        
        if total_params < 100e6:
            print(f"⚠️  WARNING: Nombre de paramètres ({total_params/1e6:.1f}M) < 100M cible")
        elif total_params > 200e6:
            print(f"⚠️  WARNING: Nombre de paramètres ({total_params/1e6:.1f}M) > 200M cible")
        else:
            print(f"✓ Nombre de paramètres dans la cible (100-200M)")
        
        print("=" * 80 + "\n")
    
    def save(self, path: str):
        """Sauvegarde la configuration en JSON"""
        import json
        from dataclasses import asdict
        
        config_dict = asdict(self)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Configuration sauvegardée: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NETYSettings':
        """Charge la configuration depuis JSON"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Reconstruire les dataclasses
        settings = cls(
            hardware=HardwareConfig(**config_dict['hardware']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            memory=MemoryConfig(**config_dict['memory']),
            generation=GenerationConfig(**config_dict['generation']),
            data=DataConfig(**config_dict['data'])
        )
        settings.version = config_dict['version']
        settings.project_name = config_dict['project_name']
        settings.description = config_dict['description']
        
        print(f"✓ Configuration chargée: {path}")
        return settings


# Instance globale par défaut
settings = NETYSettings()


if __name__ == "__main__":
    # Test et affichage de la configuration
    settings.print_summary()
    
    # Test de sauvegarde/chargement
    settings.save("config_test.json")
    loaded = NETYSettings.load("config_test.json")
    print("\n✓ Test de sauvegarde/chargement réussi")
