import torch
import torch.nn as nn
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nety.modules.text.transformer_decoder import HybridRNNTransformer

# --- Configuration ---
DATASET_PATH = ROOT / "data" / "processed" / "ml_engine" / "dataset.jsonl"
VOCAB_PATH = ROOT / "data" / "processed" / "ml_engine" / "vocab.json"
MODEL_SAVE_PATH = ROOT / "data" / "processed" / "ml_engine" / "hybrid_model.pt"

EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.0001

def load_vocab():
    """Charge le vocabulaire"""
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab["word_to_idx"], vocab["idx_to_word"]

def load_dataset():
    """Charge le dataset question-réponse"""
    pairs = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            pairs.append((entry["input"], entry["target"]))
    return pairs

def tokenize(text, word_to_idx, max_len=50):
    """Convertit un texte en indices de tokens"""
    words = text.lower().split()
    indices = []
    for word in words[:max_len]:
        idx = word_to_idx.get(word, word_to_idx.get("<unk>", 1))
        indices.append(idx)
    
    # Padding
    while len(indices) < max_len:
        indices.append(word_to_idx.get("<pad>", 0))
    
    return torch.LongTensor(indices)

def train_hybrid_model():
    """Entraîne le modèle hybride RNN-Transformer"""
    
    # Charger vocabulaire et dataset
    word_to_idx, idx_to_word = load_vocab()
    pairs = load_dataset()
    vocab_size = len(word_to_idx)
    
    print(f"📚 Vocabulaire: {vocab_size} mots")
    print(f"📝 Dataset: {len(pairs)} paires question-réponse")
    
    # Créer le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        device=str(device)
    ).to(device)
    
    # Optimiseur et loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx.get("<pad>", 0))
    
    print(f"\n🚀 Début de l'entraînement ({EPOCHS} epochs)...\n")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Traiter par batch au lieu de 1 par 1
        for batch_start in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[batch_start:batch_start + BATCH_SIZE]
            
            # Skiper si batch trop petit (pour éviter BatchNorm error)
            if len(batch_pairs) < 2:
                continue
            
            batch_src = []
            batch_tgt = []
            
            for question, answer in batch_pairs:
                batch_src.append(tokenize(question, word_to_idx))
                batch_tgt.append(tokenize(answer, word_to_idx))
            
            src = torch.stack(batch_src).to(device)  # (batch, seq_len)
            tgt = torch.stack(batch_tgt).to(device)  # (batch, seq_len)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Shift target pour teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            
            # Calculer la loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_output.reshape(-1)
            )
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max((len(pairs) // BATCH_SIZE), 1)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Modèle sauvegardé : {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_hybrid_model()