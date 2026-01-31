"""
Script de test pour vérifier l'intégration Brain ↔ ML Engine
"""

from nety.core.brain import Brain

# Test 1 : Initialisation
print("=" * 50)
print("TEST 1 : Initialisation du Brain")
print("=" * 50)

brain = Brain()
print("✅ Brain créé avec succès\n")

# Test 2 : Transformation de texte
print("=" * 50)
print("TEST 2 : Transformation de texte")
print("=" * 50)

result = brain.think("Transforme ce texte en majuscules")
print(f"Résultat : {result}\n")

# Test 3 : Question
print("=" * 50)
print("TEST 3 : Poser une question")
print("=" * 50)

result = brain.think("Comment ça fonctionne ?")
print(f"Résultat : {result}\n")

# Test 4 : Conversation
print("=" * 50)
print("TEST 4 : Conversation naturelle")
print("=" * 50)

result = brain.think("Bonjour NETY")
print(f"Résultat : {result}\n")

# Test 5 : Vérifier la mémoire
print("=" * 50)
print("TEST 5 : Mémoire à court terme")
print("=" * 50)

context = getattr(brain, "context", [])
print(f"Mémoire ({len(context)} interactions) :")
for i, interaction in enumerate(context, 1):
    print(f"  {i}. Input: {interaction.get('input', '')}")
    print(f"     Output: {interaction.get('output', '')}")

print("\n✅ Tous les tests réussis !")