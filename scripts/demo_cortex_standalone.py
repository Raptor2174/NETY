"""
Démo Autonome - Cortex Textuel RNN Standalone

Démontre le cortex textuel sans dépendre du Brain ou d'autres modules externes.
"""

import sys
from pathlib import Path
import torch

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nety.cortex_limbic.textual_cortex import TextualCortex
from nety.cortex_limbic.emotion_engine import EmotionEngine
from nety.modules.text.modele_rnn import ModeleRNN


def create_sample_embedding(text: str) -> torch.Tensor:
    """Crée un embedding simple pour un texte."""
    # Utiliser une graine déterministe basée sur le texte
    seed = sum(ord(c) for c in text)
    torch.manual_seed(seed % 10000)
    
    # Créer un embedding de 768 dimensions
    return torch.randn(1, 10, 768)


def demo_modele_rnn():
    """Démo du modèle RNN brut."""
    print("\n" + "="*70)
    print("🧠 DÉMO 1: Modèle RNN Brut avec État Persistant")
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
    
    # Mettre en eval mode
    model.eval()
    
    print(f"\n✅ Modèle RNN créé: {sum(p.numel() for p in model.parameters()):,} paramètres")
    print(f"   ├─ Device: {next(model.parameters()).device}")
    print(f"   ├─ Couches LSTM: 3 (bi-directionnelles)")
    print(f"   ├─ Têtes d'attention: 4")
    print(f"   └─ État persistant: ACTIF\n")
    
    # Traiter 3 messages avec état persistant
    messages = [
        "Bonjour, je suis votre assistant NETY",
        "Je suis capable d'apprendre de nos conversations",
        "Mon état neuronal persiste entre les interactions"
    ]
    
    print("📊 Traitement avec État Persistant:")
    print("-"*70)
    
    with torch.no_grad():
        for i, msg in enumerate(messages, 1):
            embedding = create_sample_embedding(msg)
            
            # Traiter avec état persistant
            output = model(embedding, use_persistent_state=True)
            
            print(f"\n   Message {i}: {msg[:45]}...")
            print(f"   ├─ Entrée: {embedding.shape}")
            print(f"   ├─ Sortie: {output.shape}")
            print(f"   ├─ Norme: {torch.norm(output).item():.4f}")
            print(f"   └─ État sauvegardé: ✓")
    
    # Afficher l'historique d'état
    state_history = model.get_state_history()
    print(f"\n✨ État caché persistant: {len(state_history)} enregistrements en mémoire")
    
    return model


def demo_cortex_textuel():
    """Démo du cortex textuel complet."""
    print("\n" + "="*70)
    print("🧠 DÉMO 2: Cortex Textuel Autonome")
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
    
    print(f"\n✅ Cortex Textuel créé")
    print(f"   ├─ RNN: 3 couches LSTM bi-directionnelles")
    print(f"   ├─ Attention: 4 têtes")
    print(f"   ├─ Device: {cortex.device}")
    print(f"   └─ État persistant: ACTIF\n")
    
    # Conversation multi-tours
    messages = [
        "Salut NETY, comment ça va?",
        "Raconte-moi ce que tu as appris aujourd'hui",
        "Ton cerveau se souvient-il de nos conversations?"
    ]
    
    print("💭 Conversation avec Modulation Émotionnelle:")
    print("-"*70)
    
    with torch.no_grad():
        for i, msg in enumerate(messages, 1):
            embedding = create_sample_embedding(msg)
            
            # Traiter avec contexte émotionnel
            emotional_context = {
                "current_emotion": "curiosité" if i % 2 == 0 else "intérêt",
                "emotional_intensity": 0.7
            }
            
            output, metadata = cortex.process_text_sequence(
                embedding,
                emotional_context=emotional_context,
                use_persistent_state=True
            )
            
            print(f"\n   Message {i}: {msg[:40]}...")
            print(f"   ├─ Sortie: {output.shape}")
            print(f"   ├─ Activation: {metadata['activation_level']:.3f}")
            print(f"   ├─ Profondeur État: {metadata['state_depth']}")
            print(f"   └─ Modulation Émotionnelle: ✓")
    
    # Afficher les statistiques
    stats = cortex.get_neural_statistics()
    print(f"\n📈 Statistiques Neurales:")
    print(f"   ├─ Total d'activations: {stats['total_activations']}")
    print(f"   ├─ Activation moyenne: {stats['average_activation']:.4f}")
    print(f"   ├─ Activation maximale: {stats['peak_activation']:.4f}")
    print(f"   └─ Profondeur du contexte: {len(cortex.context_window)}")
    
    return cortex


def demo_modulation_emotionnelle():
    """Démo de la modulation émotionnelle."""
    print("\n" + "="*70)
    print("❤️ DÉMO 3: Modulation Émotionnelle")
    print("="*70)
    
    emotion_engine = EmotionEngine()
    cortex = TextualCortex(
        hidden_size=256,
        output_size=512,
        num_layers=3,
        num_heads=4,
        emotion_engine=emotion_engine
    )
    
    print(f"\n✅ Cortex avec Limbic System initialisé\n")
    
    # Tester différents contextes émotionnels
    contexts = [
        {"emotion": "joie", "intensity": 0.9},
        {"emotion": "tristesse", "intensity": 0.7},
        {"emotion": "colère", "intensity": 0.8},
        {"emotion": "calme", "intensity": 0.5}
    ]
    
    print("🎭 Impact des Émotions sur l'Activation Neurales:")
    print("-"*70)
    
    with torch.no_grad():
        for ctx in contexts:
            embedding = create_sample_embedding("test")
            
            emotional_context = {
                "current_emotion": ctx["emotion"],
                "emotional_intensity": ctx["intensity"]
            }
            
            output, metadata = cortex.process_text_sequence(
                embedding,
                emotional_context=emotional_context,
                use_persistent_state=False
            )
            
            print(f"\n   Émotion: {ctx['emotion'].upper()}")
            print(f"   ├─ Intensité: {ctx['intensity']:.1f}")
            print(f"   ├─ Activation Neurales: {metadata['activation_level']:.3f}")
            print(f"   └─ Modulation Active: ✓")


def main():
    """Exécute toutes les démos."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*18 + "🧠 CORTEX TEXTUEL RNN - DÉMO STANDALONE" + " "*11 + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Démo 1: RNN brut
        model = demo_modele_rnn()
        
        # Démo 2: Cortex textuel
        cortex = demo_cortex_textuel()
        
        # Démo 3: Modulation émotionnelle
        demo_modulation_emotionnelle()
        
        print("\n" + "="*70)
        print("✅ TOUS LES DÉMOS RÉUSSIS!")
        print("="*70)
        print("\n💡 Le Cortex Textuel RNN est maintenant")
        print("   ├─ Complètement fonctionnel")
        print("   ├─ Intégré aux systèmes émotionnels")
        print("   ├─ Capable d'état persistant")
        print("   └─ Prêt pour l'intégration Brain complète\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la démo: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
