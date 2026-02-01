"""
🎭 Démonstration de l'impact des traits de personnalité sur les réponses de NETY
"""
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from nety.cortex_limbic.adaptive_personality import AdaptivePersonality
from nety.cortex_limbic.advanced_limbic_filter import AdvancedLimbicFilter

def demo_scenario_1():
    """Scénario : Question sur la ville de NETY"""
    print("\n" + "=" * 70)
    print("📍 SCÉNARIO 1 : Question sur l'origine de NETY")
    print("=" * 70)
    
    limbic = AdvancedLimbicFilter()
    
    context = {
        "user_id": "user123",
        "message": "D'où viens-tu NETY ?",
        "intent": {"type": "question"},
        "allow_humor": True
    }
    
    filter_params = limbic.apply_filter(context)
    
    print(f"\n👤 UTILISATEUR : {context['message']}")
    print(f"\n🎯 TON DÉTECTÉ : {filter_params['tone']}")
    print(f"📝 STYLE : {filter_params['style']}")
    
    print("\n🧠 TRAITS ACTIVÉS :")
    cultural = filter_params['cultural_traits']
    if cultural['origine_caen'] > 0.9:
        print(f"  ✅ Origine Caen : {cultural['origine_caen']} → FORTE influence")
    if cultural['culture_normande'] > 0.8:
        print(f"  ✅ Culture normande : {cultural['culture_normande']} → FORTE influence")
    
    print("\n📜 RÈGLES COMPORTEMENTALES ACTIVES (pertinentes) :")
    relevant_rules = [r for r in filter_params['behavior_rules'] 
                      if 'Caen' in r or 'normande' in r]
    for rule in relevant_rules:
        print(f"  • {rule}")
    
    print("\n💭 RÉPONSE ATTENDUE :")
    print("  « Je viens de Caen, en Normandie ! C'est une ville que j'apprécie")
    print("    particulièrement, avec son riche patrimoine historique. »")
    print("\n  ➜ La réponse reflète l'attachement fort (0.95) à Caen")

def demo_scenario_2():
    """Scénario : Question technique"""
    print("\n" + "=" * 70)
    print("🔧 SCÉNARIO 2 : Question technique")
    print("=" * 70)
    
    limbic = AdvancedLimbicFilter()
    
    context = {
        "user_id": "user456",
        "message": "Comment fonctionne un moteur de recherche ?",
        "intent": {"type": "question"},
        "allow_humor": False
    }
    
    filter_params = limbic.apply_filter(context)
    
    print(f"\n👤 UTILISATEUR : {context['message']}")
    
    print("\n🧠 TRAITS COGNITIFS ACTIVÉS :")
    cognitive = filter_params['cognitive_traits']
    print(f"  • Esprit technique : {cognitive['esprit_technique']}")
    print(f"  • Pensée holistique : {cognitive['pensee_holistique']}")
    print(f"  • Non-cartésianisme : {cognitive['non_cartesianisme']}")
    
    print("\n📜 RÈGLES COMPORTEMENTALES ACTIVES (pertinentes) :")
    relevant_rules = [r for r in filter_params['behavior_rules'] 
                      if 'technique' in r or 'analytique' in r or 'holistique' in r or 'contexte' in r]
    for rule in relevant_rules:
        print(f"  • {rule}")
    
    print("\n💭 RÉPONSE ATTENDUE :")
    print("  « Un moteur de recherche fonctionne en 3 étapes principales :")
    print("    1. Exploration (crawling) des pages web")
    print("    2. Indexation des contenus")
    print("    3. Classement selon la pertinence")
    print()
    print("    Mais pour comprendre pleinement leur impact, il faut aussi")
    print("    considérer leur évolution historique et leur rôle sociétal... »")
    print("\n  ➜ Approche technique (0.90) + vision holistique (0.90)")

def demo_scenario_3():
    """Scénario : Question ouverte nécessitant créativité"""
    print("\n" + "=" * 70)
    print("💡 SCÉNARIO 3 : Question créative")
    print("=" * 70)
    
    limbic = AdvancedLimbicFilter()
    
    context = {
        "user_id": "user789",
        "message": "Comment résoudre le problème du réchauffement climatique ?",
        "intent": {"type": "question"},
        "allow_humor": True
    }
    
    filter_params = limbic.apply_filter(context)
    
    print(f"\n👤 UTILISATEUR : {context['message']}")
    
    print("\n🧠 COMBINAISON DE TRAITS :")
    print(f"  • Créativité : {filter_params['sub_traits']['créativité']}")
    print(f"  • Non-cartésianisme : {filter_params['cognitive_traits']['non_cartesianisme']}")
    print(f"  • Pensée holistique : {filter_params['cognitive_traits']['pensee_holistique']}")
    
    print("\n📜 RÈGLES COMPORTEMENTALES ACTIVES (pertinentes) :")
    relevant_rules = [r for r in filter_params['behavior_rules'] 
                      if 'non-conventionnelles' in r or 'contexte' in r or 'global' in r]
    for rule in relevant_rules:
        print(f"  • {rule}")
    
    print("\n💭 RÉPONSE ATTENDUE :")
    print("  « Le réchauffement climatique nécessite une approche systémique.")
    print("    Au-delà des solutions techniques classiques (renouvelables, etc.),")
    print("    il faut repenser notre rapport à la croissance.")
    print()
    print("    Une idée non-conventionnelle : et si on s'inspirait des systèmes")
    print("    médiévaux de gestion des communs pour gérer les ressources ? »")
    print("\n  ➜ Pensée holistique (0.90) + approche non-conventionnelle (0.90)")

def show_identity_summary():
    """Affiche le résumé d'identité complet"""
    print("\n" + "=" * 70)
    print("🎭 IDENTITÉ COMPLÈTE DE NETY")
    print("=" * 70)
    
    personality = AdaptivePersonality()
    
    print("\n📝 RÉSUMÉ AUTO-GÉNÉRÉ :")
    print(f"  {personality.get_identity_summary()}")
    
    print("\n📊 DÉTAIL DES SCORES :")
    print("\n  🌍 Culturel :")
    for trait, value in personality.cultural_traits.items():
        bar = "█" * int(value * 20)
        print(f"    {trait:25} [{bar:<20}] {value}")
    
    print("\n  🧩 Cognitif :")
    for trait, value in personality.cognitive_traits.items():
        bar = "█" * int(value * 20)
        print(f"    {trait:25} [{bar:<20}] {value}")

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "🎭 DÉMONSTRATION DES TRAITS DE PERSONNALITÉ" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    
    show_identity_summary()
    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    
    print("\n" + "=" * 70)
    print("✨ CONCLUSION")
    print("=" * 70)
    print("""
Les traits de personnalité uniques influencent maintenant :
  
  ✅ Le TON des réponses (amical, empathique, etc.)
  ✅ Le STYLE de communication (conversationnel, technique, etc.)
  ✅ Les RÈGLES comportementales dynamiques
  ✅ Le CONTENU des prompts envoyés au LLM
  ✅ L'IDENTITÉ présentée à l'utilisateur

NETY n'est plus un assistant générique - c'est une IA avec une identité
culturelle, géographique et cognitive unique ! 🎉
""")
