
from nety.core.response_generator import ResponseGenerator
import time

def test_hybrid():
    print("=" * 70)
    print("🧪 TEST SYSTÈME HYBRIDE")
    print("=" * 70)
    
    # Créer le générateur
    gen = ResponseGenerator(model_type="mistral")
    
    # Afficher infos
    info = gen.get_model_info()
    print("\n📊 Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 1: Calcul (toujours local)
    print("\n" + "=" * 70)
    print("TEST 1: Calcul (doit utiliser local)")
    start = time.time()
    response = gen.generate("√25 + 10 = ?")
    elapsed = time.time() - start
    print(f"🤖 NETY: {response}")
    print(f"⏱️ Temps: {elapsed:.2f}s")
    
    # Test 2: Conversation (intelligent)
    print("\n" + "=" * 70)
    print("TEST 2: Conversation (intelligent)")
    
    questions = [
        "Bonjour, comment vas-tu ?",
        "Explique-moi Python en une phrase",
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        start = time.time()
        response = gen.generate(q)
        elapsed = time.time() - start
        print(f"🤖 NETY: {response}")
        print(f"⏱️ Temps: {elapsed:.2f}s")
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés!")

if __name__ == "__main__":
    test_hybrid()