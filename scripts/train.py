"""
NETY V2-Maxx - Training Script
===============================

Script d'entraînement complet pour NETYBrainV2.
Utilise génération neuronale pure (pas de templates).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import json
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# Imports NETY
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nety.settings import NETYSettings
from nety.models.nety_brain_v2 import NETYBrainV2, NETYBrainConfig
from nety.preprocessing.text_preprocessor import Preprocessor
from nety.postprocessing.text_postprocessor import Postprocessor


class ConversationDataset(Dataset):
    """Dataset de conversations pour entraînement"""
    
    def __init__(
        self,
        conversations: List[Dict],
        preprocessor: Preprocessor,
        max_input_length: int = 256,
        max_target_length: int = 128
    ):
        self.conversations = conversations
        self.preprocessor = preprocessor
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conv = self.conversations[idx]
        
        # Encoder input
        input_enc = self.preprocessor(
            conv['input'],
            padding='max_length',
            truncation=True
        )
        
        # Encoder target
        target_enc = self.preprocessor(
            conv['output'],
            padding='max_length',
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(input_enc['input_ids'][:self.max_input_length]),
            'input_attention_mask': torch.tensor(input_enc['attention_mask'][:self.max_input_length]),
            'target_ids': torch.tensor(target_enc['input_ids'][:self.max_target_length]),
            'target_attention_mask': torch.tensor(target_enc['attention_mask'][:self.max_target_length])
        }


class NETYTrainer:
    """Trainer pour NETY V2-Maxx"""
    
    def __init__(
        self,
        model: NETYBrainV2,
        settings: NETYSettings,
        preprocessor: Preprocessor,
        postprocessor: Postprocessor
    ):
        self.model = model
        self.settings = settings
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
        self.device = torch.device(settings.hardware.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=settings.training.learning_rate,
            weight_decay=settings.training.weight_decay,
            betas=(settings.training.adam_beta1, settings.training.adam_beta2),
            eps=settings.training.adam_epsilon
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=settings.model.pad_token_id)
        
        # Mixed Precision
        self.scaler = GradScaler() if settings.training.use_amp else None
        
        # Stats
        self.global_step = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """Entraîne une epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            input_mask = batch['input_attention_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            target_mask = batch['target_attention_mask'].to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                # Mixed precision
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        target_ids=target_ids,
                        input_attention_mask=input_mask,
                        target_attention_mask=target_mask,
                        teacher_forcing_ratio=0.9  # 90% teacher forcing
                    )
                    
                    # Compute loss
                    logits = outputs['logits']  # (batch, seq_len, vocab_size)
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    targets_flat = target_ids.reshape(-1)
                    
                    loss = self.criterion(logits_flat, targets_flat)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.settings.training.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.settings.training.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
            else:
                # Standard precision
                outputs = self.model(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    input_attention_mask=input_mask,
                    target_attention_mask=target_mask,
                    teacher_forcing_ratio=0.9
                )
                
                logits = outputs['logits']
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = target_ids.reshape(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                
                # Backward
                loss.backward()
                
                if (batch_idx + 1) % self.settings.training.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.settings.training.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> float:
        """Évalue le modèle"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                input_mask = batch['input_attention_mask'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                target_mask = batch['target_attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    input_attention_mask=input_mask,
                    target_attention_mask=target_mask,
                    teacher_forcing_ratio=1.0  # Full teacher forcing for eval
                )
                
                logits = outputs['logits']
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = target_ids.reshape(-1)
                
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, path: str):
        """Sauvegarde un checkpoint"""
        from dataclasses import asdict
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'settings': asdict(self.settings)
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint sauvegardé: {path}")
    
    def load_checkpoint(self, path: str):
        """Charge un checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Checkpoint chargé: {path}")
        return checkpoint['epoch']


def load_conversations(path: str) -> List[Dict]:
    """Charge le dataset de conversations"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data['conversations']
    print(f"✓ Chargé {len(conversations)} conversations")
    
    return conversations


def prepare_datasets(
    conversations: List[Dict],
    preprocessor: Preprocessor,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[ConversationDataset, ConversationDataset, ConversationDataset]:
    """Prépare les datasets train/val/test"""
    import random
    random.shuffle(conversations)
    
    total = len(conversations)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    train_convs = conversations[:train_size]
    val_convs = conversations[train_size:train_size + val_size]
    test_convs = conversations[train_size + val_size:]
    
    train_dataset = ConversationDataset(train_convs, preprocessor)
    val_dataset = ConversationDataset(val_convs, preprocessor)
    test_dataset = ConversationDataset(test_convs, preprocessor)
    
    print(f"✓ Datasets préparés:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Script d'entraînement principal"""
    print("=" * 80)
    print("NETY V2-Maxx - Training")
    print("=" * 80)
    
    # 1. Charger settings
    settings = NETYSettings()
    settings.print_summary()
    
    # 2. Charger conversations
    conversations = load_conversations(settings.data.train_data_path)
    
    # 3. Préparer preprocessor
    print("\n📝 Préparation du preprocessor...")
    preprocessor = Preprocessor(
        vocab_size=settings.model.vocab_size,
        max_length=settings.model.max_seq_length,
        lowercase=settings.data.lowercase
    )
    
    # Fit sur corpus
    all_texts = []
    for conv in conversations:
        all_texts.append(conv['input'])
        all_texts.append(conv['output'])
    
    preprocessor.fit(all_texts, min_frequency=2)
    preprocessor.save("data/tokenizer")
    
    # 4. Préparer datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        conversations, preprocessor,
        settings.data.train_split,
        settings.data.val_split
    )
    
    # 5. Créer dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        num_workers=0,  # CPU workers
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 6. Créer modèle
    print("\n🧠 Création du modèle...")
    model_config = NETYBrainConfig(
        vocab_size=len(preprocessor.tokenizer.token_to_id),
        embedding_dim=settings.model.embedding_dim,
        max_seq_length=settings.model.max_seq_length,
        cognitive_num_layers=settings.model.cognitive_num_layers,
        cognitive_num_heads=settings.model.cognitive_num_heads,
        rnn_encoder_hidden_dim=settings.model.rnn_encoder_hidden_dim,
        rnn_decoder_hidden_dim=settings.model.rnn_decoder_hidden_dim
    )
    
    model = NETYBrainV2(model_config)
    print(f"✓ Modèle créé: {model.count_parameters():,} paramètres")
    
    # 7. Créer postprocessor
    postprocessor = Postprocessor()
    
    # 8. Créer trainer
    trainer = NETYTrainer(model, settings, preprocessor, postprocessor)
    
    # 9. Training loop
    print("\n🎓 Début de l'entraînement...\n")
    
    for epoch in range(1, settings.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{settings.training.num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = trainer.evaluate(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            checkpoint_path = f"{settings.training.checkpoint_dir}/best_model.pt"
            trainer.save_checkpoint(epoch, val_loss, checkpoint_path)
        
        # Save regular checkpoint
        if epoch % 5 == 0:
            checkpoint_path = f"{settings.training.checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(epoch, val_loss, checkpoint_path)
    
    print("\n" + "=" * 80)
    print("✓ Entraînement terminé!")
    print(f"  Meilleure val loss: {trainer.best_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
