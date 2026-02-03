#!/usr/bin/env python3
"""Test pour reproduire et vérifier le fix de l'erreur d'attention"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from nety.models.nety_brain_v2 import NETYBrainConfig, NETYBrainV2

def test_training_forward_pass():
    """Reproduit le forward pass du training qui avait l'erreur"""
    
    print("=" * 70)
    print("TEST: Forward pass d'entraînement (reproduit l'erreur originale)")
    print("=" * 70)
    
    # Configuration - utiliser les valeurs par défaut
    config = NETYBrainConfig()
    config.rnn_decoder_attention = True  # Attention activée
    
    # Créer le modèle
    device = torch.device("cpu")
    model = NETYBrainV2(config).to(device)
    model.train()  # Mode training
    
    print(f"\n✓ Modèle créé (attention={'ACTIVÉE' if config.rnn_decoder_attention else 'DÉSACTIVÉE'})")
    
    # Paramètres du batch
    batch_size = 2
    input_seq_len = 8
    target_seq_len = 6
    
    # Créer les inputs
    input_ids = torch.randint(3, config.vocab_size, (batch_size, input_seq_len), device=device)
    target_ids = torch.randint(3, config.vocab_size, (batch_size, target_seq_len), device=device)
    
    # Créer les masks (1 = token valide, 0 = padding)
    input_mask = torch.ones(batch_size, input_seq_len, device=device)
    input_mask[:, -2:] = 0  # Marquer les 2 derniers tokens comme padding
    target_mask = torch.ones(batch_size, target_seq_len, device=device)
    
    print(f"\n✓ Inputs créés:")
    print(f"  - input_ids shape: {input_ids.shape}")
    print(f"  - target_ids shape: {target_ids.shape}")
    print(f"  - input_mask shape: {input_mask.shape}")
    print(f"  - target_mask shape: {target_mask.shape}")
    
    # Forward pass
    print(f"\n[Forward pass...]")
    try:
        outputs = model(
            input_ids=input_ids,
            target_ids=target_ids,
            input_attention_mask=input_mask,
            target_attention_mask=target_mask,
            teacher_forcing_ratio=1.0
        )
        
        print(f"✓ Forward pass réussi!")
        print(f"\n✓ Outputs:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  - {key}: {type(value)}")
        
        # Vérifier les shapes
        assert outputs['logits'].shape == (batch_size, target_seq_len, config.vocab_size)
        assert outputs['emotion_logits'].shape == (batch_size, config.limbic_num_emotions)
        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
            print(f"  - attention_weights available: shape={outputs['attention_weights'].shape}")
        
        print(f"\n✓ Shapes correctes!")
        
        # Backward pass pour tester si le gradient s'écoule
        print(f"\n[Backward pass...]")
        loss_logits = outputs['logits'].mean()
        loss_emotion = outputs['emotion_logits'].mean()
        loss = loss_logits + 0.1 * loss_emotion
        
        loss.backward()
        print(f"✓ Backward pass réussi!")
        print(f"  - Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    success = test_training_forward_pass()
    print("\n" + "=" * 70)
    if success:
        print("✓✓✓ TEST PASSÉ - Le fix fonctionne correctement!")
    else:
        print("✗✗✗ TEST ÉCHOUÉ")
    print("=" * 70)
    exit(0 if success else 1)
