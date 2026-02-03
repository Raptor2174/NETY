"""
Test du flux complet RNN: Message → Réponse
Vérifie que tout fonctionne de bout en bout
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au sys.path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from nety.core.rnn_response_generator import RNNResponseGenerator
import torch

def test_rnn_generator():
    """Test du générateur RNN isolé"""
    print("=" * 60)
    print("TEST 1: RNN Response Generator (isolé)")
    print("=" * 60)
    
    try:
        generator = RNNResponseGenerator()
        print("\n✅ RNN Generator initialisé")
        
        # Test 1: Salutation simple
        print("\n📝 Test 1: Salutation")
        context = {
            "current_message": "bonjour",
            "user_profile": {"name": "Raptor"},
            "personal_memory": [],
            "limbic_filter": {
                "emotional_state": {
                    "dominant_emotion": "joyeux",
                    "intensity": 0.7
                }
            }
        }
        
        response = generator.generate(
            message="bonjour",
            context=context,
            limbic_filter=context["limbic_filter"]
        )
        print(f"   ├─ Message: bonjour")
        print(f"   └─ Réponse: {response}")
        
        # Test 2: Question identité
        print("\n📝 Test 2: Question identité")
        context["current_message"] = "qui es-tu ?"
        response = generator.generate(
            message="qui es-tu ?",
            context=context,
            limbic_filter=context["limbic_filter"]
        )
        print(f"   ├─ Message: qui es-tu ?")
        print(f"   └─ Réponse: {response}")
        
        # Test 3: Question émotionnelle
        print("\n📝 Test 3: Question émotionnelle")
        context["current_message"] = "comment tu te sens ?"
        response = generator.generate(
            message="comment tu te sens ?",
            context=context,
            limbic_filter=context["limbic_filter"]
        )
        print(f"   ├─ Message: comment tu te sens ?")
        print(f"   └─ Réponse: {response}")
        
        # Test 4: Message générique
        print("\n📝 Test 4: Message générique")
        context["current_message"] = "j'aime les chats"
        response = generator.generate(
            message="j'aime les chats",
            context=context,
            limbic_filter=context["limbic_filter"]
        )
        print(f"   ├─ Message: j'aime les chats")
        print(f"   └─ Réponse: {response}")
        
        print("\n✅ Tous les tests RNN passés!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur dans le test RNN: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_generator():
    """Test du ResponseGenerator complet"""
    print("=" * 60)
    print("TEST 2: Response Generator (avec sélection backend)")
    print("=" * 60)
    
    try:
        from nety.core.response_generator import ResponseGenerator
        
        print("\n🧠 Initialisation avec backend RNN...")
        gen = ResponseGenerator(model_type="rnn")
        print("✅ Response Generator initialisé")
        
        # Test avec contexte
        print("\n📝 Test avec contexte complet")
        context = {
            "user_profile": {"name": "Raptor"},
            "personal_memory": [],
            "history": []
        }
        
        limbic_filter = {
            "emotional_state": {
                "dominant_emotion": "calme",
                "state": "bien",
                "intensity": 0.5
            },
            "tone": "friendly"
        }
        
        response = gen.generate(
            message="salut NETY",
            context=context,
            limbic_filter=limbic_filter
        )
        
        print(f"   ├─ Message: salut NETY")
        print(f"   └─ Réponse: {response}")
        
        print("\n✅ Test Response Generator passé!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur dans le test Response Generator: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_activation():
    """Test du calcul d'activation neuronale"""
    print("=" * 60)
    print("TEST 3: Calcul d'activation neuronale")
    print("=" * 60)
    
    try:
        generator = RNNResponseGenerator()
        
        # Créer des outputs de test
        test_outputs = [
            torch.randn(1, 10, 512),  # Faible activation
            torch.randn(1, 10, 512) * 2,  # Moyenne activation
            torch.randn(1, 10, 512) * 5,  # Haute activation
        ]
        
        print("\n📊 Test des niveaux d'activation:")
        for i, output in enumerate(test_outputs, 1):
            activation = generator._calculate_neural_activation(output)
            print(f"   ├─ Test {i}: activation = {activation:.4f}")
            
            # Vérifier que c'est bien entre 0 et 1
            assert 0.0 <= activation <= 1.0, f"Activation hors limites: {activation}"
        
        print("\n✅ Calculs d'activation corrects!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur dans le test activation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_detection():
    """Test de la détection d'intention"""
    print("=" * 60)
    print("TEST 4: Détection d'intention")
    print("=" * 60)
    
    try:
        generator = RNNResponseGenerator()
        
        test_cases = [
            ("bonjour", "greeting"),
            ("qui es-tu", "identity_question"),
            ("comment tu te sens", "emotional_question"),
            ("tu te souviens de moi", "memory_recall"),
            ("qui suis-je", "user_identity_question"),
            ("aimes-tu les chats", "preference_question"),
            ("j'aime le chocolat", "generic"),
        ]
        
        print("\n🎯 Test de détection d'intention:")
        all_correct = True
        for message, expected_intent in test_cases:
            detected = generator._detect_intent(message)
            status = "✅" if detected == expected_intent else "❌"
            print(f"   {status} '{message}' → {detected} (attendu: {expected_intent})")
            if detected != expected_intent:
                all_correct = False
        
        if all_correct:
            print("\n✅ Toutes les intentions détectées correctement!\n")
        else:
            print("\n⚠️ Certaines intentions mal détectées\n")
        
        return all_correct
        
    except Exception as e:
        print(f"\n❌ Erreur dans le test intention: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TESTS FLUX COMPLET RNN - MESSAGE → RÉPONSE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: RNN Generator isolé
    results.append(("RNN Generator", test_rnn_generator()))
    
    # Test 2: Response Generator complet
    results.append(("Response Generator", test_response_generator()))
    
    # Test 3: Activation neuronale
    results.append(("Activation neuronale", test_neural_activation()))
    
    # Test 4: Détection d'intention
    results.append(("Détection intention", test_intent_detection()))
    
    # Résumé
    print("=" * 60)
    print("  RÉSUMÉ DES TESTS")
    print("=" * 60)
    for test_name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{status:12} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n📊 Score: {passed}/{total} tests passés ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS! Le flux est fonctionnel.")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
