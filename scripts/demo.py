"""
NETY V2-Maxx - Demo Rapide
===========================

Démonstration rapide de l'architecture complète sans entraînement.
Montre le flow complet : Input → Pipeline → Output
"""

import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nety.settings import NETYSettings
from nety.models.nety_brain_v2 import NETYBrainV2, NETYBrainConfig
from nety.preprocessing.text_preprocessor import Preprocessor
from nety.postprocessing.text_postprocessor import Postprocessor


def demo_pipeline():
    """Démontre le pipeline complet"""
    print("=" * 80)
    print("NETY V2-Maxx - Démonstration Pipeline Complet")
    print("=" * 80)
    
    # 1. Configuration
    print("\n1️⃣  CONFIGURATION")
    print("-" * 80)
    settings = NETYSettings()
    print(f"✓ Vocabulaire: {settings.model.vocab_size:,} tokens")
    print(f"✓ Embedding dim: {settings.model.embedding_dim}")
    print(f"✓ Architecture: Input → Cognitive (4 layers) → Limbic (6 emotions)")
    print(f"               → RNN Encoder (3 Bi-LSTM) → RNN Decoder (3 LSTM) → Output")
    print(f"✓ Paramètres estimés: {settings.model.estimate_parameters() / 1e6:.1f}M")
    print(f"✓ VRAM estimée: {settings.model.estimate_vram_usage_gb(settings.training.batch_size):.2f} GB")
    
    # 2. Preprocessing
    print("\n2️⃣  PREPROCESSING")
    print("-" * 80)
    
    # Charger ou créer preprocessor
    if os.path.exists("data/tokenizer/vocab.json"):
        preprocessor = Preprocessor.load("data/tokenizer")
    else:
        print("⚠️  Tokenizer non trouvé. Création d'un tokenizer de demo...")
        preprocessor = Preprocessor(vocab_size=1000, max_length=256)
        demo_corpus = [
            "Bonjour comment vas-tu",
            "Je vais bien merci",
            "Qu'est-ce que tu fais",
            "Je discute avec toi"
        ]
        preprocessor.fit(demo_corpus, min_frequency=1)
    
    # Test preprocessing
    user_input = "Bonjour, comment vas-tu aujourd'hui ?"
    print(f"✓ Input: {user_input}")
    
    encoded = preprocessor(user_input)
    print(f"✓ Tokenization: {len([id for id in encoded['input_ids'] if id != 0])} tokens (+ padding)")
    print(f"✓ IDs: {encoded['input_ids'][:20]}... (truncated)")
    
    # 3. Modèle
    print("\n3️⃣  MODÈLE NEURONAL")
    print("-" * 80)
    
    config = NETYBrainConfig(
        vocab_size=len(preprocessor.tokenizer.token_to_id),
        embedding_dim=settings.model.embedding_dim,
        max_seq_length=settings.model.max_seq_length,
        cognitive_num_layers=settings.model.cognitive_num_layers,
        rnn_encoder_hidden_dim=settings.model.rnn_encoder_hidden_dim,
        rnn_decoder_hidden_dim=settings.model.rnn_decoder_hidden_dim
    )
    
    model = NETYBrainV2(config)
    model.eval()
    
    print(f"✓ Modèle créé: {model.count_parameters():,} paramètres")
    print(f"✓ Architecture détaillée:")
    print(f"   - Embedding: {config.vocab_size} × {config.embedding_dim}")
    print(f"   - Cognitive Layer: {config.cognitive_num_layers} Transformer layers")
    print(f"   - Limbic System: {config.limbic_num_emotions} emotions")
    print(f"   - RNN Encoder: {config.rnn_encoder_num_layers} Bi-LSTM layers")
    print(f"   - RNN Decoder: {config.rnn_decoder_num_layers} LSTM layers + Attention")
    
    # 4. Forward Pass (Encoding)
    print("\n4️⃣  ENCODING (Input → Representations)")
    print("-" * 80)
    
    input_ids = torch.tensor([encoded['input_ids']])
    input_mask = torch.tensor([encoded['attention_mask']])
    
    with torch.no_grad():
        # Encoder
        encoder_outputs, encoder_hidden, emotion_logits = model.encode(
            input_ids, input_mask
        )
    
    print(f"✓ Encoder outputs shape: {encoder_outputs.shape}")
    print(f"✓ Emotion logits shape: {emotion_logits.shape}")
    
    # Afficher émotions prédites
    emotions = ['joie', 'tristesse', 'colère', 'peur', 'surprise', 'neutre']
    emotion_probs = torch.softmax(emotion_logits[0], dim=0)
    top_emotion_idx = torch.argmax(emotion_probs).item()
    print(f"✓ Émotion dominante: {emotions[top_emotion_idx]} ({emotion_probs[top_emotion_idx]:.2%})")
    
    # 5. Génération
    print("\n5️⃣  GÉNÉRATION (Decoding)")
    print("-" * 80)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=50,
            temperature=settings.generation.temperature,
            top_k=settings.generation.top_k,
            top_p=settings.generation.top_p
        )
    
    print(f"✓ Generated IDs shape: {generated_ids.shape}")
    print(f"✓ IDs: {generated_ids[0].tolist()[:20]}... (truncated)")
    
    # 6. Postprocessing
    print("\n6️⃣  POSTPROCESSING")
    print("-" * 80)
    
    # Détokenization
    raw_text = preprocessor.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print(f"✓ Raw output: {raw_text[:100]}...")
    
    # Postprocessing
    postprocessor = Postprocessor()
    cleaned_text = postprocessor(raw_text)
    
    if cleaned_text:
        print(f"✓ Cleaned output: {cleaned_text[:100]}...")
    else:
        print("✓ Output rejected by content filter (too short/inappropriate)")
    
    # 7. Pipeline Complet
    print("\n7️⃣  PIPELINE COMPLET RÉSUMÉ")
    print("-" * 80)
    print("Input (texte brut)")
    print("  ↓ Preprocessing (normalisation, tokenization, encoding)")
    print("Tokens IDs + Attention Mask")
    print("  ↓ Embedding Layer")
    print("Token Embeddings (512 dims)")
    print("  ↓ Cognitive Layer (4 Transformer Encoder layers)")
    print("Cognitive Representations")
    print("  ↓ Limbic System (emotional modulation)")
    print("Modulated Representations + Emotion Prediction")
    print("  ↓ RNN Encoder (3 Bi-LSTM layers)")
    print("Encoder Outputs + Hidden State")
    print("  ↓ RNN Decoder (3 LSTM layers + Attention)")
    print("Generated Token IDs (autoregressive)")
    print("  ↓ Postprocessing (detokenization, formatting, filtering)")
    print("Output (texte nettoyé)")
    print("-" * 80)
    
    # Stats finales
    print("\n📊 STATISTIQUES FINALES")
    print("-" * 80)
    print(f"✓ Modèle: {model.count_parameters() / 1e6:.1f}M paramètres")
    print(f"✓ Vocabulaire: {len(preprocessor.tokenizer.token_to_id):,} tokens")
    print(f"✓ Input tokens: {sum(encoded['attention_mask'])}")
    print(f"✓ Output tokens: {(generated_ids[0] != config.pad_token_id).sum().item()}")
    print(f"✓ Émotion: {emotions[top_emotion_idx]}")
    
    print("\n" + "=" * 80)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 80)
    print("\n💡 Note: Le modèle n'est pas entraîné, donc les sorties sont aléatoires.")
    print("   Après entraînement, NETY générera des réponses cohérentes et naturelles.")
    print("\n🚀 Pour entraîner: python scripts/train.py")
    print("🗣️  Pour chatbot: python scripts/inference.py --mode chat")


if __name__ == "__main__":
    demo_pipeline()
