# examples/test_mistral.py
"""
Script de test pour Mistral-7B
"""
from nety.core.response_generator import ResponseGenerator
from nety.core.llm_config import LLMConfig
import time

def test_basic_generation():
    """Test de génération basique"""
    print("=" * 70)
    print("🧪 TEST 1: Génération basique")
    print("=" * 70)
    
    generator = ResponseGenerator(model_type="mistral")
    
    # Afficher les infos du modèle
    info = generator.get_model_info()
    print("\n📊 Informations du modèle:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test simple
    questions = [
        "Bonjour, comment vas-tu ?",
        "Qu'est-ce qu'un réseau de neurones ?",
        "Explique-moi Python en une phrase.",
        "2 + 2 = ?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        start = time.time()
        response = generator.generate(question)
        elapsed = time.time() - start
        print(f"🤖 NETY: {response}")
        print(f"⏱️ Temps: {elapsed:.2f}s")
        print("-" * 70)


def test_with_context():
    """Test avec contexte et historique"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Génération avec contexte")
    print("=" * 70)
    
    generator = ResponseGenerator(model_type="mistral")
    
    # Simuler un historique
    context = {
        "user_name": "Raptor",
        "history": [
            {"input": "Je m'appelle Raptor", "output": "Enchanté Raptor !"},
            {"input": "J'aime Python", "output": "Python est un excellent langage !"}
        ],
        "knowledge": "Python est un langage de programmation créé par Guido van Rossum en 1991."
    }
    
    limbic_filter = {
        "tone": "friendly",
        "behavior_rules": ["être concis", "utiliser des emojis"]
    }
    
    question = "Rappelle-moi ce que j'aime ?"
    
    print(f"\n❓ Question: {question}")
    print(f"👤 Contexte: user_name={context['user_name']}, {len(context['history'])} messages")
    
    start = time.time()
    response = generator.generate(question, context=context, limbic_filter=limbic_filter)
    elapsed = time.time() - start
    
    print(f"🤖 NETY: {response}")
    print(f"⏱️ Temps: {elapsed:.2f}s")


def test_comparison():
    """Compare BLOOMZ vs Mistral"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Comparaison BLOOMZ vs Mistral")
    print("=" * 70)
    
    question = "Qu'est-ce que l'intelligence artificielle ?"
    
    # Test BLOOMZ
    print("\n📝 Test avec BLOOMZ-560M...")
    try:
        gen_bloomz = ResponseGenerator(model_type="bloomz")
        start = time.time()
        response_bloomz = gen_bloomz.generate(question)
        time_bloomz = time.time() - start
        print(f"🤖 BLOOMZ: {response_bloomz}")
        print(f"⏱️ Temps: {time_bloomz:.2f}s")
    except Exception as e:
        print(f"❌ Erreur BLOOMZ: {e}")
        response_bloomz = None
        time_bloomz = None
    
    # Test Mistral
    print("\n📝 Test avec Mistral-7B...")
    try:
        gen_mistral = ResponseGenerator(model_type="mistral")
        start = time.time()
        response_mistral = gen_mistral.generate(question)
        time_mistral = time.time() - start
        print(f"🤖 Mistral: {response_mistral}")
        print(f"⏱️ Temps: {time_mistral:.2f}s")
    except Exception as e:
        print(f"❌ Erreur Mistral: {e}")
        response_mistral = None
        time_mistral = None
    
    # Comparaison
    if response_bloomz and response_mistral and time_mistral is not None and time_bloomz is not None:
        print("\n📊 Comparaison:")
        print(f"  Longueur BLOOMZ: {len(response_bloomz)} caractères")
        print(f"  Longueur Mistral: {len(response_mistral)} caractères")
        print(f"  Différence vitesse: {time_mistral - time_bloomz:+.2f}s")


def main():
    """Exécute tous les tests"""
    print("🚀 Démarrage des tests Mistral-7B pour NETY")
    print()
    
    try:
        test_basic_generation()
        test_with_context()
        # test_comparison()  # Décommenter si tu veux comparer
        
        print("\n" + "=" * 70)
        print("✅ Tous les tests terminés avec succès!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur lors des tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()