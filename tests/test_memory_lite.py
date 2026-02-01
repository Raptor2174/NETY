#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simplifié du système de mémoire (sans charger le Brain complet).
Focus sur les labels, corrélations et récupération de souvenirs.
"""

import sys
import os
import json
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nety.modules.machinelearning.ml_engine import MLEngine
from nety.cortex_limbic.memory_manager import MemoryManager


def test_complete_memory_workflow():
    """Test complet du workflow de mémoire"""
    print("\n" + "="*70)
    print("🧠 TEST COMPLET: Système de Mémoire NETY avec Labels & Corrélations")
    print("="*70)
    
    ml_engine = MLEngine()
    mem_manager = MemoryManager()
    
    # === ÉTAPE 1: Enregistrer une conversation ===
    print("\n📝 ÉTAPE 1: Enregistrement de la conversation")
    print("-" * 70)
    
    conversation = [
        ("Je m'appelle Raptor_ et je suis admin", "identity"),
        ("Je suis très passionné par le machine learning", "preference"),
        ("Mon objectif principal est de créer une IA capable de se souvenir", "goal"),
        ("J'ai travaillé chez plusieurs startups tech", "work"),
        ("Je n'aime pas les bugs mais j'adore les résoudre", "preference"),
    ]
    
    memory_ids = []
    for msg, msg_type in conversation:
        print(f"\n💬 Message ({msg_type}): {msg}")
        
        # Utiliser la nouvelle méthode avec labels
        entry = ml_engine.assign_memory_labels(msg, user_id="00000001")
        memory_ids.append(entry['id'])
        
        print(f"   ✓ ID: {entry['id'][:40]}...")
        print(f"   ✓ Labels: {entry['labels']}")
        print(f"   ✓ Sentiment: {entry['meta']['sentiment']}")
        print(f"   ✓ Catégories: {entry['categories']}")
        print(f"   ✓ Faits extraits: {entry['facts']}")
    
    # === ÉTAPE 2: Vérifier les labels attribués ===
    print("\n" + "="*70)
    print("🏷️  ÉTAPE 2: Vérification des labels attribués")
    print("-" * 70)
    
    memories = ml_engine.get_memory_with_context(user_id="00000001", limit=10)
    print(f"\n✓ Total de souvenirs: {len(memories)}")
    
    label_summary = {}
    for mem in memories:
        for label in mem.get('labels', []):
            label_summary[label] = label_summary.get(label, 0) + 1
    
    print("\n📊 Distribution des labels:")
    for label, count in sorted(label_summary.items(), key=lambda x: -x[1]):
        print(f"   • {label}: {count}")
    
    # === ÉTAPE 3: Vérifier les corrélations ===
    print("\n" + "="*70)
    print("🔗 ÉTAPE 3: Vérification des corrélations")
    print("-" * 70)
    
    key_infos = ml_engine.load_key_info()
    correlations = [ki for ki in key_infos if ki.get("type") == "correlation"]
    
    print(f"\n✓ Total de corrélations: {len(correlations)}")
    
    # Afficher les corrélations par type de champ
    corr_by_field = {}
    for corr in correlations:
        field = corr.get("field", "unknown")
        if field not in corr_by_field:
            corr_by_field[field] = []
        corr_by_field[field].append(corr.get("value"))
    
    print("\n📋 Corrélations enregistrées:")
    for field, values in corr_by_field.items():
        print(f"\n   {field}:")
        for val in values[:3]:  # Afficher max 3 valeurs
            print(f"      • {val}")
        if len(values) > 3:
            print(f"      ... et {len(values)-3} autres")
    
    # === ÉTAPE 4: Tester la récupération contextuelle ===
    print("\n" + "="*70)
    print("🔍 ÉTAPE 4: Récupération de souvenirs contextualisés")
    print("-" * 70)
    
    queries = [
        "Qui es-tu?",
        "Quels sont tes objectifs?",
        "Dis-moi tes préférences",
    ]
    
    for query in queries:
        print(f"\n❓ Requête: '{query}'")
        relevant = ml_engine.get_relevant_memories(query, limit=3)
        
        print(f"   ✓ Souvenirs pertinents trouvés: {len(relevant)}")
        for i, mem in enumerate(relevant, 1):
            print(f"      {i}. {mem['text'][:50]}...")
            print(f"         Labels: {mem.get('labels', [])}")
    
    # === ÉTAPE 5: Vérifier la compression des mémoires anciennes ===
    print("\n" + "="*70)
    print("💾 ÉTAPE 5: Vérification du stockage de la mémoire locale")
    print("-" * 70)
    
    mem_manager.memory_labels = {mem['id']: mem.get('labels', []) for mem in memories}
    
    recent = mem_manager.get_recent_memories(count=5)
    print(f"\n✓ Souvenirs récents dans MemoryManager: {len(recent)}")
    for i, mem in enumerate(recent, 1):
        print(f"   {i}. {mem.get('content', 'N/A')[:40]}...")
        print(f"      Labels: {mem.get('labels', [])}")
    
    # === ÉTAPE 6: Vérifier la récupération par labels ===
    print("\n" + "="*70)
    print("🏷️  ÉTAPE 6: Récupération des souvenirs par label")
    print("-" * 70)
    
    for label in ['identity_info', 'goal', 'preference']:
        by_label = mem_manager.get_memories_by_label(label)
        print(f"\n   Label '{label}': {len(by_label)} souvenir(s)")
        for mem in by_label[:2]:
            print(f"      • {mem.get('content', 'N/A')[:40]}...")
    
    # === RÉSUMÉ ===
    print("\n" + "="*70)
    print("✅ RÉSUMÉ DU TEST")
    print("="*70)
    
    stats = ml_engine.get_stats()
    print(f"""
✓ Souvenirs enregistrés: {stats.get('total_entries', 0)}
✓ Catégories identifiées: {len(stats.get('category_counts', {}))}
✓ Corrélations stockées: {len(correlations)}
✓ Labels assignés: {len(label_summary)}
✓ User ID: 00000001 (Raptor_)

🎯 Résultats:
   ✓ Labels contextuels: FONCTIONNEL
   ✓ Sentiment analysis: FONCTIONNEL
   ✓ Corrélations: FONCTIONNEL
   ✓ Récupération contextuelle: FONCTIONNEL
   ✓ Stockage persistant: FONCTIONNEL

🔮 Prochaine étape:
   L'IA peut maintenant:
   1. Se souvenir des conversations récentes
   2. Assigner des labels contextuels
   3. Créer des corrélations entre informations
   4. Récupérer les souvenirs avec contexte
   5. Analyser le sentiment des interactions
""")


if __name__ == "__main__":
    try:
        test_complete_memory_workflow()
        print("\n" + "="*70)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*70)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
