#!/usr/bin/env python3
"""Test unitaire pour vérifier que le fix d'attention fonctionne"""

import torch
import torch.nn as nn
from pathlib import Path

# Ajouter le repo à sys.path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nety.models.nety_brain_v2 import AttentionMechanism

def test_attention_with_mask():
    """Test l'attention avec un masque de dimensions variées"""
    
    print("=" * 60)
    print("TEST: AttentionMechanism avec masque")
    print("=" * 60)
    
    # Paramètres
    batch_size = 4
    seq_len = 8
    hidden_dim = 256
    encoder_dim = 512
    
    # Créer le mécanisme d'attention
    attention = AttentionMechanism(hidden_dim, encoder_dim)
    
    # Créer les tenseurs d'entrée
    hidden = torch.randn(batch_size, hidden_dim)  # (batch, hidden_dim)
    encoder_outputs = torch.randn(batch_size, seq_len, encoder_dim)  # (batch, seq_len, encoder_dim)
    
    print(f"\n✓ Tenseurs créés:")
    print(f"  - hidden shape: {hidden.shape}")
    print(f"  - encoder_outputs shape: {encoder_outputs.shape}")
    
    # Test 1: Sans masque
    print(f"\n--- Test 1: Sans masque ---")
    try:
        context, weights = attention(hidden, encoder_outputs, mask=None)
        print(f"✓ Succès!")
        print(f"  - context shape: {context.shape}")
        print(f"  - weights shape: {weights.shape}")
        assert context.shape == (batch_size, encoder_dim)
        assert weights.shape == (batch_size, seq_len)
        print(f"✓ Shapes correctes!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False
    
    # Test 2: Avec masque (batch, seq_len)
    print(f"\n--- Test 2: Masque (batch, seq_len) ---")
    try:
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True  # Masquer les 2 derniers tokens
        print(f"  - mask shape: {mask.shape}")
        
        context, weights = attention(hidden, encoder_outputs, mask=mask)
        print(f"✓ Succès!")
        print(f"  - context shape: {context.shape}")
        print(f"  - weights shape: {weights.shape}")
        print(f"  - max weight (devrait être sur tokens non-maskés): {weights.max():.4f}")
        print(f"  - attention sur tokens maskés (devrait être ~0): {weights[:, -2:].max():.6f}")
        assert context.shape == (batch_size, encoder_dim)
        assert weights.shape == (batch_size, seq_len)
        print(f"✓ Shapes correctes!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False
    
    # Test 3: Avec masque 1D
    print(f"\n--- Test 3: Masque 1D (seq_len,) ---")
    try:
        mask_1d = torch.zeros(seq_len, dtype=torch.bool)
        mask_1d[-2:] = True  # Masquer les 2 derniers tokens
        print(f"  - mask shape: {mask_1d.shape}")
        
        context, weights = attention(hidden, encoder_outputs, mask=mask_1d)
        print(f"✓ Succès!")
        print(f"  - context shape: {context.shape}")
        print(f"  - weights shape: {weights.shape}")
        assert context.shape == (batch_size, encoder_dim)
        assert weights.shape == (batch_size, seq_len)
        print(f"✓ Shapes correctes!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False
    
    # Test 4: Cas où seq_len du masque ne correspond pas (doit être géré)
    print(f"\n--- Test 4: Masque avec seq_len différente ---")
    try:
        mask_diff = torch.zeros(batch_size, 10, dtype=torch.bool)  # seq_len=10 vs 8
        print(f"  - mask shape: {mask_diff.shape}")
        
        context, weights = attention(hidden, encoder_outputs, mask=mask_diff)
        print(f"✓ Succès (masque a été truncaté)!")
        print(f"  - context shape: {context.shape}")
        print(f"  - weights shape: {weights.shape}")
        assert context.shape == (batch_size, encoder_dim)
        assert weights.shape == (batch_size, seq_len)
        print(f"✓ Shapes correctes!")
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ TOUS LES TESTS PASSENT!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_attention_with_mask()
    exit(0 if success else 1)
