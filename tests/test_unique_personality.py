"""
Test des traits de personnalité uniques de NETY
"""
import sys
from pathlib import Path

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from nety.cortex_limbic.adaptive_personality import AdaptivePersonality
from nety.cortex_limbic.advanced_limbic_filter import AdvancedLimbicFilter

def test_personality_traits():
    """Teste que les traits sont bien définis"""
    personality = AdaptivePersonality()
    
    print("=" * 60)
    print("🧠 TEST DES TRAITS DE PERSONNALITÉ UNIQUES")
    print("=" * 60)
    
    # Traits Big Five
    print("\n📊 TRAITS BIG FIVE:")
    for trait, value in personality.traits.items():
        print(f"  • {trait}: {value}")
    
    # Traits culturels
    print("\n🌍 TRAITS CULTURELS:")
    for trait, value in personality.cultural_traits.items():
        print(f"  • {trait}: {value}")
    
    # Traits cognitifs
    print("\n🧩 TRAITS COGNITIFS:")
    for trait, value in personality.cognitive_traits.items():
        print(f"  • {trait}: {value}")
    
    # Sous-traits
    print("\n⚙️ SOUS-TRAITS COMPORTEMENTAUX:")
    for trait, value in personality.sub_traits.items():
        print(f"  • {trait}: {value}")
    
    # Résumé d'identité
    print("\n📝 RÉSUMÉ D'IDENTITÉ:")
    print(f"  {personality.get_identity_summary()}")
    
    print("\n" + "=" * 60)
    print("✅ Test des traits terminé!")
    print("=" * 60)

def test_limbic_filter_integration():
    """Teste l'intégration dans le filtre limbique"""
    print("\n" + "=" * 60)
    print("🔬 TEST D'INTÉGRATION AU FILTRE LIMBIQUE")
    print("=" * 60)
    
    limbic_filter = AdvancedLimbicFilter()
    
    # Contexte de test
    context = {
        "user_id": "test_user",
        "message": "Parle-moi de ta ville",
        "intent": {"type": "question"},
        "allow_humor": True
    }
    
    # Appliquer le filtre
    filter_params = limbic_filter.apply_filter(context)
    
    print("\n📋 PARAMÈTRES DU FILTRE:")
    print(f"  • Ton: {filter_params['tone']}")
    print(f"  • Style: {filter_params['style']}")
    print(f"  • Verbosité: {filter_params['verbosity']}")
    
    print("\n🌍 TRAITS CULTURELS ACTIFS:")
    for trait, value in filter_params['cultural_traits'].items():
        print(f"  • {trait}: {value}")
    
    print("\n🧩 TRAITS COGNITIFS ACTIFS:")
    for trait, value in filter_params['cognitive_traits'].items():
        print(f"  • {trait}: {value}")
    
    print("\n📜 RÈGLES COMPORTEMENTALES:")
    for rule in filter_params['behavior_rules'][:5]:  # Afficher les 5 premières
        print(f"  • {rule}")
    print(f"  ... et {len(filter_params['behavior_rules']) - 5} autres règles")
    
    print("\n" + "=" * 60)
    print("✅ Test d'intégration terminé!")
    print("=" * 60)

def test_personality_config():
    """Teste la configuration de personnalité pour un contexte"""
    print("\n" + "=" * 60)
    print("⚙️ TEST DE CONFIGURATION CONTEXTUELLE")
    print("=" * 60)
    
    personality = AdaptivePersonality()
    
    context = {
        "user_id": "test_user",
        "intent": {"type": "question"},
        "emotional_state": {"dominant_emotion": "confiance"}
    }
    
    config = personality.get_personality_for_context(context)
    
    print("\n📋 CONFIGURATION GÉNÉRÉE:")
    print(f"  • Ton: {config['tone']}")
    print(f"  • Style: {config['style']}")
    print(f"  • Verbosité: {config['verbosity']}")
    
    print("\n✅ Traits culturels inclus:", "cultural_traits" in config)
    print("✅ Traits cognitifs inclus:", "cognitive_traits" in config)
    
    print("\n" + "=" * 60)
    print("✅ Test de configuration terminé!")
    print("=" * 60)

if __name__ == "__main__":
    test_personality_traits()
    test_limbic_filter_integration()
    test_personality_config()
    
    print("\n" + "=" * 60)
    print("🎉 TOUS LES TESTS RÉUSSIS!")
    print("=" * 60)
