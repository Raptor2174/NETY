"""
Demo de l'architecture hybride RNN-Transformer

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

import torch
import sys
import os

# Ajouter le path du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nety.modules.text.transformer_decoder import HybridRNNTransformer


def create_dummy_vocab(size: int = 1000):
    """Crée un vocabulaire factice pour la démo"""
    word_to_idx = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    
    # Ajouter des mots courants
    common_words = [
        "bonjour", "salut", "hello", "comment", "vas", "tu", "je", "suis", "bien",
        "merci", "de", "quoi", "pourquoi", "quand", "où", "qui", "est", "ton", "nom",
        "intelligence", "artificielle", "ia", "nety", "raptor", "conversation",
        "message", "réponse", "question", "répondre", "comprendre", "apprendre",
        "neural", "réseau", "transformer", "rnn", "lstm", "attention"
    ]
    
    for i, word in enumerate(common_words, start=4):
        word_to_idx[word] = i
    
    # Remplir le reste avec des mots factices
    for i in range(len(word_to_idx), size):
        word_to_idx[f"word_{i}"] = i
    
    # Créer idx_to_word
    idx_to_word = {str(v): k for k, v in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


def text_to_tokens(text: str, word_to_idx: dict) -> torch.Tensor:
    """Convertit du texte en tokens"""
    words = text.lower().split()
    indices = [word_to_idx.get(word, word_to_idx.get("<unk>", 3)) for word in words]
    return torch.LongTensor(indices).unsqueeze(0)


def tokens_to_text(tokens: list, idx_to_word: dict) -> str:
    """Convertit des tokens en texte"""
    words = []
    for token_id in tokens:
        word = idx_to_word.get(str(token_id), "<unk>")
        if word not in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            words.append(word)
    return " ".join(words)


def main():
    print("="*70)
    print("🔥 DEMO: Architecture Hybride RNN-Transformer")
    print("="*70)
    print()
    
    # Configuration
    vocab_size = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"📊 Configuration:")
    print(f"   ├─ Vocabulaire: {vocab_size} tokens")
    print(f"   ├─ Device: {device}")
    print(f"   └─ Architecture: RNN(3L) → Transformer(6L, 512D, 8H)")
    print()
    
    # Créer le vocabulaire
    print("📚 Création du vocabulaire...")
    word_to_idx, idx_to_word = create_dummy_vocab(vocab_size)
    print(f"   ✅ {len(word_to_idx)} tokens chargés")
    print()
    
    # Créer le modèle
    print("🏗️  Construction du modèle hybride...")
    model = HybridRNNTransformer(
        vocab_size=vocab_size,
        rnn_hidden_size=256,
        rnn_num_layers=3,
        rnn_num_heads=4,
        decoder_d_model=512,
        decoder_nhead=8,
        decoder_num_layers=6,
        decoder_dim_feedforward=2048,
        dropout=0.1,
        device=device
    )
    model.eval()
    print()
    
    # Messages de test
    test_messages = [
        "bonjour comment vas tu",
        "quel est ton nom",
        "tu es une intelligence artificielle",
        "parlons de transformer et rnn",
    ]
    
    print("="*70)
    print("🎯 TESTS DE GÉNÉRATION")
    print("="*70)
    print()
    
    for i, message in enumerate(test_messages, 1):
        print(f"[Test {i}/{len(test_messages)}]")
        print(f"📝 Input: '{message}'")
        
        # Convertir en tokens
        src_tokens = text_to_tokens(message, word_to_idx)
        print(f"   ├─ Tokens: {src_tokens.shape} = {src_tokens[0].tolist()[:10]}...")
        
        # Générer avec le modèle
        with torch.no_grad():
            try:
                # Test 1: Génération normale
                generated_tokens = model.generate(
                    src=src_tokens,
                    start_token=word_to_idx["<sos>"],
                    end_token=word_to_idx["<eos>"],
                    max_length=20,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                
                # Convertir en texte
                response = tokens_to_text(generated_tokens, idx_to_word)
                print(f"   ├─ Generated tokens: {generated_tokens[:10]}...")
                print(f"   └─ 🤖 Response: '{response}'")
                
            except Exception as e:
                print(f"   └─ ❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
        
        print()
    
    # Test de forward pass complet
    print("="*70)
    print("🔬 TEST FORWARD PASS COMPLET")
    print("="*70)
    print()
    
    src = text_to_tokens("bonjour nety", word_to_idx)
    tgt = text_to_tokens("salut comment puis-je aider", word_to_idx)
    
    print(f"Source: {src.shape} = {src[0].tolist()}")
    print(f"Target: {tgt.shape} = {tgt[0].tolist()}")
    print()
    
    with torch.no_grad():
        try:
            # Forward pass avec teacher forcing
            logits = model(src, tgt)
            print(f"✅ Logits shape: {logits.shape}")
            print(f"   └─ (batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab={logits.shape[2]})")
            
            # Probabilités du prochain token
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_k = 5
            top_probs, top_indices = torch.topk(probs, top_k)
            
            print()
            print(f"🎲 Top-{top_k} tokens prédits:")
            for j, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                word = idx_to_word.get(str(idx.item()), "<unk>")
                print(f"   {j}. {word:15s} {prob.item()*100:6.2f}%")
                
        except Exception as e:
            print(f"❌ Erreur forward: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("="*70)
    print("📊 STATISTIQUES DU MODÈLE")
    print("="*70)
    print()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    rnn_params = sum(p.numel() for p in model.rnn_encoder.parameters())
    transformer_params = sum(p.numel() for p in model.transformer_decoder.parameters())
    
    print(f"Total params:        {total_params:,}")
    print(f"Trainable params:    {trainable_params:,}")
    print(f"   ├─ RNN Encoder:   {rnn_params:,} ({rnn_params/total_params*100:.1f}%)")
    print(f"   └─ Transformer:   {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    print()
    
    # Taille en mémoire
    param_size = total_params * 4 / (1024**2)  # 4 bytes par float32
    print(f"Taille mémoire:      {param_size:.2f} MB")
    print()
    
    print("="*70)
    print("✅ DEMO TERMINÉE")
    print("="*70)


if __name__ == "__main__":
    main()
