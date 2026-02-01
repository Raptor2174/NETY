"""
Démo d'Intégration - Cortex Textuel RNN dans le Cerveau NETY

Démontre:
  - L'intégration complète du RNN autonome dans le cerveau
  - La conversation avec activation neuronal et modulation émotionnelle
  - L'apprentissage contextuel persistant
  - Les statistiques d'activation neurales
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from nety.core.brain import Brain
except ImportError as e:
    print(f"⚠️ Erreur d'import: {e}")
    print("Assurez-vous que vous êtes dans le répertoire du projet NETY")
    sys.exit(1)


def demo_cortex_integration():
    """Démo interactive du cortex textuel intégré."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*12 + "🧠 DÉMO - CORTEX TEXTUEL RNN DANS NETY" + " "*15 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Initialiser le cerveau
    print("🧠 Initialisation du cerveau NETY...\n")
    try:
        brain = Brain()
        print("\n✅ Cerveau initialisé avec succès\n")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return
    
    # Vérifier que le cortex textuel est présent
    if hasattr(brain, 'textual_cortex'):
        print("✨ Cortex Textuel détecté et actif!")
        print(f"   ├─ État persistant: ACTIVÉ")
        print(f"   ├─ Attention multi-tête: 4 têtes")
        print(f"   └─ LSTM bi-directionnel: 3 couches\n")
    else:
        print("⚠️ Cortex Textuel non trouvé")
        return
    
    # Messages de test
    test_messages = [
        "Bonjour! Comment ça va?",
        "Parle-moi de ton architecture neuronal",
        "Que penses-tu de l'apprentissage profond?",
        "C'est fascinant! Montre-moi tes statistiques"
    ]
    
    print("="*70)
    print("📊 TRAITEMENT DE MESSAGES AVEC ACTIVATION NEURONAL")
    print("="*70 + "\n")
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Message {i}: {message}")
        print("-" * 70)
        
        try:
            # Traiter le message via le cerveau
            response = brain.think(message)
            
            # Afficher la réponse (troncée)
            response_short = response[:100] + "..." if len(response) > 100 else response
            print(f"🤖 Réponse: {response_short}")
            
            # Afficher les statistiques neurales du cortex
            if hasattr(brain, 'textual_cortex'):
                stats = brain.textual_cortex.get_neural_statistics()
                print(f"\n📈 Activation Neurales:")
                print(f"   ├─ Activation courante: {stats['current_activation']:.4f}")
                print(f"   ├─ Moyenne (100 dernières): {stats['average_activation']:.4f}")
                print(f"   ├─ Pic: {stats['peak_activation']:.4f}")
                print(f"   ├─ Profondeur contextuelle: {stats['context_depth']}")
                print(f"   └─ Total d'activations: {stats['total_activations']}")
                
                # Résumé contextuel
                context_summary = brain.textual_cortex.get_context_summary()
                print(f"\n🧠 Contexte Neuronal:")
                print(f"   ├─ Fenêtre contextuelle: {context_summary['window_size']} interactions")
                print(f"   └─ État neuronal actif: ✓")
        
        except Exception as e:
            print(f"❌ Erreur lors du traitement: {e}")
    
    # Afficher le résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ FINAL - STATISTIQUES NEURALES GLOBALES")
    print("="*70 + "\n")
    
    if hasattr(brain, 'textual_cortex'):
        stats = brain.textual_cortex.get_neural_statistics()
        context = brain.textual_cortex.get_context_summary()
        
        print("📈 Statistiques Neurales:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   ├─ {key}: {value:.4f}")
            else:
                print(f"   ├─ {key}: {value}")
        
        print("\n🧠 État du Cortex:")
        print(f"   ├─ Interactions en mémoire: {len(brain.get_context())}")
        print(f"   ├─ Fenêtre contextuelle: {context['window_size']}")
        print(f"   ├─ État persistant: {'SAUVEGARDÉ' if brain.textual_cortex.get_persistent_state() else 'VIDE'}")
        print(f"   └─ Statut des modules:")
        for module, status in brain.get_modules_status().items():
            print(f"      ├─ {module}: {status}")
    
    print("\n" + "="*70)
    print("✅ DÉMONSTRATION COMPLÉTÉE!")
    print("="*70 + "\n")


def demo_interactive():
    """Démo interactive interactive."""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*18 + "💬 MODE INTERACTIF - NETY" + " "*24 + "║")
    print("╚" + "="*68 + "╝\n")
    
    # Initialiser le cerveau
    print("🧠 Initialisation du cerveau...\n")
    try:
        brain = Brain()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return
    
    print("✅ Cerveau initialisé!")
    print("💡 Tapez vos messages (tapez 'exit' ou 'quit' pour arrêter)\n")
    
    interaction_count = 0
    
    while True:
        try:
            user_input = input("📝 Vous: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sortir']:
                print("\n👋 Au revoir! Cerveau arrêté.\n")
                break
            
            if not user_input:
                continue
            
            interaction_count += 1
            
            # Traiter le message
            response = brain.think(user_input)
            print(f"\n🤖 NETY: {response}\n")
            
            # Afficher l'activation tous les 3 messages
            if interaction_count % 3 == 0 and hasattr(brain, 'textual_cortex'):
                stats = brain.textual_cortex.get_neural_statistics()
                print(f"📊 [Activation: {stats['current_activation']:.3f} | "
                      f"Avg: {stats['average_activation']:.3f} | "
                      f"Profondeur: {stats['context_depth']}]\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Arrêt du programme (Ctrl+C)\n")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Démo du Cortex Textuel RNN de NETY")
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive'],
        default='demo',
        help='Mode de démonstration (défaut: demo)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        demo_interactive()
    else:
        demo_cortex_integration()
