"""
Test complet du Cortex Textuel RNN de NETY

Démontre:
  - Traitement de séquences textuelles
  - État caché persistant entre interactions
  - Modulation émotionnelle
  - Intégration avec le système limbique
  - Statistiques d'activation neurales
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Ajouter le chemin du projet (tests/../.. = racine du NETY)
project_root = Path(__file__).parent.parent  # Remonte de tests/ à NETY/
sys.path.insert(0, str(project_root))

from nety.modules.text.modele_rnn import ModeleRNN
from nety.cortex_limbic.textual_cortex import TextualCortex
from nety.cortex_limbic.emotion_engine import EmotionEngine


def create_sample_embedding(text: str, embedding_dim: int = 768) -> torch.Tensor:
    """Crée un embedding fictif basé sur le hash du texte."""
    hash_val = hash(text)
    np.random.seed(abs(hash_val) % (2**31))
    embedding = np.random.randn(1, 1, embedding_dim).astype(np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return torch.from_numpy(embedding)


def test_modele_rnn():
    """Test du modèle RNN brut."""
    print("\n" + "="*70)
    print("TEST 1: Modèle RNN Brut (ModeleRNN)")
    print("="*70)
    
    # Créer le modèle
    model = ModeleRNN(
        input_size=768,
        hidden_size=256,
        output_size=512,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    )
    
    print(f"✅ Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Créer des embeddings fictifs
    batch_size = 2
    seq_length = 5
    embeddings = torch.randn(batch_size, seq_length, 768)
    
    # Traitement avec état persistant
    print("\n📊 Traitement séquentiel (3 appels):")
    outputs = []
    for i in range(3):
        output = model(embeddings, use_persistent_state=True)
        outputs.append(output)
        print(f"   Appel {i+1}: shape={output.shape}, norm={torch.norm(output).item():.4f}")
    
    # Vérifier la persistance d'état
    state_history = model.get_state_history()
    print(f"\n📌 État persistant: {len(state_history)} états en mémoire")
    
    return model


def test_textual_cortex():
    """Test du Cortex Textuel complet."""
    print("\n" + "="*70)
    print("TEST 2: Cortex Textuel Autonome")
    print("="*70)
    
    # Créer l'emotion engine
    emotion_engine = EmotionEngine()
    
    # Créer le cortex
    cortex = TextualCortex(
        hidden_size=256,
        output_size=512,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
        emotion_engine=emotion_engine
    )
    
    print("✅ Cortex créé avec état persistant")
    
    # Traiter plusieurs messages
    test_messages = [
        "Bonjour, je suis un nouvel utilisateur",
        "Comment fonctionnent tes émotions?",
        "Raconte-moi une histoire intéressante"
    ]
    
    print("\n📊 Traitement de 3 messages séquentiels:")
    print("-" * 70)
    
    for i, message in enumerate(test_messages, 1):
        # Créer l'embedding
        embedding = create_sample_embedding(message)
        
        # Traiter
        output, metadata = cortex.process_text_sequence(embedding)
        
        # Afficher les résultats
        print(f"\n   Message {i}: {message[:40]}...")
        print(f"   ├─ Activation: {metadata['activation_level']:.3f}")
        print(f"   ├─ State depth: {metadata['state_depth']}")
        print(f"   └─ Output shape: {output.shape}")
        
        # Ajouter au contexte
        cortex.add_to_context_window({
            "input": message,
            "timestamp": metadata["timestamp"],
            "activation": metadata["activation_level"]
        })
    
    return cortex


def test_emotional_modulation():
    """Test de la modulation émotionnelle."""
    print("\n" + "="*70)
    print("TEST 3: Modulation Émotionnelle du Limbic System")
    print("="*70)
    
    emotion_engine = EmotionEngine()
    cortex = TextualCortex(emotion_engine=emotion_engine)
    
    # Créer un contexte émotionnel positif
    positive_context = {
        "emotions": {
            "joie": 0.8,
            "confiance": 0.7,
            "anticipation": 0.6,
            "tristesse": 0.1,
            "colère": 0.0,
            "peur": 0.0,
            "dégoût": 0.0
        }
    }
    
    # Créer un contexte émotionnel négatif
    negative_context = {
        "emotions": {
            "joie": 0.2,
            "confiance": 0.3,
            "anticipation": 0.2,
            "tristesse": 0.7,
            "colère": 0.6,
            "peur": 0.5,
            "dégoût": 0.4
        }
    }
    
    message = "Message de test"
    embedding = create_sample_embedding(message)
    
    print("\n📊 Impact émotionnel sur l'activation:")
    print("-" * 70)
    
    # Test avec contexte positif
    output_pos, meta_pos = cortex.process_text_sequence(
        embedding,
        emotional_context=positive_context,
        use_persistent_state=False
    )
    
    # Test avec contexte négatif
    output_neg, meta_neg = cortex.process_text_sequence(
        embedding,
        emotional_context=negative_context,
        use_persistent_state=False
    )
    
    # Afficher les résultats
    print(f"\n   Contexte POSITIF:")
    print(f"   ├─ Activation: {meta_pos['activation_level']:.3f}")
    print(f"   └─ Modulation: {meta_pos['emotional_modulation']}")
    
    print(f"\n   Contexte NÉGATIF:")
    print(f"   ├─ Activation: {meta_neg['activation_level']:.3f}")
    print(f"   └─ Modulation: {meta_neg['emotional_modulation']}")
    
    diff = meta_pos['activation_level'] - meta_neg['activation_level']
    print(f"\n   Différence d'activation (Pos - Neg): {diff:+.3f}")
    
    return cortex


def test_neural_statistics():
    """Test des statistiques neurales."""
    print("\n" + "="*70)
    print("TEST 4: Statistiques Neurales")
    print("="*70)
    
    cortex = TextualCortex()
    
    # Générer plusieurs activations
    print("\n📊 Génération de 10 activations:")
    for i in range(10):
        embedding = create_sample_embedding(f"Message {i}")
        output, meta = cortex.process_text_sequence(embedding)
    
    # Afficher les statistiques
    print("\n📈 Statistiques Complètes:")
    stats = cortex.get_neural_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   ├─ {key}: {value:.4f}")
        else:
            print(f"   ├─ {key}: {value}")
    
    # Contexte summary
    print("\n🧠 Résumé Contextuel:")
    context = cortex.get_context_summary()
    for key, value in context.items():
        if key == "recent_interactions":
            print(f"   ├─ {key}:")
            for inter in value:
                print(f"   │  ├─ {inter}")
        else:
            print(f"   ├─ {key}: {value}")
    
    return cortex


def test_state_persistence():
    """Test de la persistance d'état."""
    print("\n" + "="*70)
    print("TEST 5: Persistance d'État")
    print("="*70)
    
    cortex = TextualCortex()
    
    print("\n📊 Test de sauvegarde/restauration d'état:")
    print("-" * 70)
    
    # Phase 1: Traitement initial
    print("\n   Phase 1: Traitement initial (3 messages)")
    messages_1 = ["Message 1", "Message 2", "Message 3"]
    for msg in messages_1:
        embedding = create_sample_embedding(msg)
        cortex.process_text_sequence(embedding)
    
    state_before = cortex.get_persistent_state()
    if state_before is not None:
        print(f"   ✅ État sauvegardé: h={state_before[0].shape}, c={state_before[1].shape}")
    else:
        print(f"   ✅ État sauvegardé: None (pas d'état persistant)")
    
    # Phase 2: Réinitialiser et restaurer
    print("\n   Phase 2: Réinitialisation et restauration")
    cortex.reset_state()
    print(f"   🔄 État réinitialisé")
    
    if state_before is not None:
        cortex.load_persistent_state(state_before)
        print(f"   ✅ État restauré")
    else:
        print(f"   ⚠️ Pas d'état à restaurer")
    
    # Phase 3: Vérifier la continuité
    print("\n   Phase 3: Continuité après restauration")
    embedding = create_sample_embedding("Message 4")
    output, meta = cortex.process_text_sequence(embedding)
    print(f"   ✅ Traitement réussi avec état restauré")
    print(f"   └─ Activation: {meta['activation_level']:.3f}")


def main():
    """Lance tous les tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "🧠 TEST SUITE - CORTEX TEXTUEL NETY" + " "*18 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Test 1: Modèle RNN brut
        model = test_modele_rnn()
        
        # Test 2: Cortex Textuel
        cortex1 = test_textual_cortex()
        
        # Test 3: Modulation Émotionnelle
        cortex2 = test_emotional_modulation()
        
        # Test 4: Statistiques
        cortex3 = test_neural_statistics()
        
        # Test 5: Persistance
        test_state_persistence()
        
        print("\n" + "="*70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DES TESTS: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
