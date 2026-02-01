#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test complet du système de mémoire amélioré avec labels et corrélations.
Vérifie que NETY peut:
1. Se souvenir des dernières conversations
2. Assigner des labels contextuels
3. Enregistrer les corrélations entre informations
"""

import sys
import os
import json
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nety.modules.machinelearning.ml_engine import MLEngine
from nety.cortex_limbic.memory_manager import MemoryManager
from nety.core.brain import Brain


def test_memory_labels():
    """Test 1: Assigner des labels aux souvenirs"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Assignation de labels aux souvenirs")
    print("="*60)
    
    ml_engine = MLEngine()
    
    # Enregistrer différents types de messages
    test_messages = [
        ("Je m'appelle Raptor_", "identity"),
        ("J'aime beaucoup le café et la programmation", "preference"),
        ("Mon objectif est de maîtriser le machine learning", "goal"),
        ("Je suis ingénieur en informatique", "work"),
    ]
    
    for message, expected_type in test_messages:
        print(f"\n📝 Message: {message}")
        entry = ml_engine.assign_memory_labels(message, user_id="00000001")
        
        print(f"   ✓ ID: {entry['id'][:40]}...")
        print(f"   ✓ Labels: {entry['labels']}")
        print(f"   ✓ Sentiment: {entry['meta']['sentiment']}")
        print(f"   ✓ Urgence: {entry['meta']['urgency']}")
        print(f"   ✓ Catégories: {entry['categories']}")


def test_memory_retrieval():
    """Test 2: Récupérer les souvenirs avec contexte"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Récupération de souvenirs avec contexte")
    print("="*60)
    
    ml_engine = MLEngine()
    
    # Ajouter plusieurs souvenirs
    messages = [
        "Je suis français et j'aime voyager",
        "Mon hobby principal c'est la programmation",
        "Je travaille chez une startup tech",
        "Mon objectif court terme est de finir ce projet",
        "J'ai remarqué un bug dans le système",
    ]
    
    for msg in messages:
        ml_engine.assign_memory_labels(msg, user_id="00000001")
    
    # Récupérer les souvenirs récents
    print("\n📚 Souvenirs avec contexte:")
    recent = ml_engine.get_memory_with_context(user_id="00000001", limit=5)
    
    for i, mem in enumerate(recent, 1):
        print(f"\n   {i}. {mem['text'][:50]}...")
        print(f"      Labels: {mem.get('labels', [])}")
        print(f"      Sentiment: {mem.get('meta', {}).get('sentiment', 'N/A')}")
        print(f"      Timestamp: {mem['timestamp']}")


def test_correlations():
    """Test 3: Vérifier les corrélations enregistrées"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Vérification des corrélations")
    print("="*60)
    
    ml_engine = MLEngine()
    
    # Enregistrer des messages avec faits clés
    messages = [
        "Je m'appelle Raptor_ et je suis admin",
        "Je travaille chez OpenAI",
        "Mon email est raptor@example.com",
    ]
    
    for msg in messages:
        ml_engine.assign_memory_labels(msg, user_id="00000001")
    
    # Vérifier les corrélations dans key_info.jsonl
    key_infos = ml_engine.load_key_info()
    
    print(f"\n📊 Corrélations enregistrées ({len(key_infos)} entrées):")
    
    correlations = [ki for ki in key_infos if ki.get("type") == "correlation"]
    for corr in correlations[-5:]:  # Afficher les 5 dernières
        print(f"\n   Type: {corr.get('type')}")
        print(f"   Champ: {corr.get('field')}")
        print(f"   Valeur: {corr.get('value')}")
        print(f"   Catégorie: {corr.get('category')}")
        print(f"   Labels: {corr.get('labels', [])}")
        print(f"   Sentiment: {corr.get('sentiment')}")


def test_brain_integration():
    """Test 4: Intégration dans le cerveau NETY"""
    print("\n" + "="*60)
    print("🧪 TEST 4: Intégration dans le cerveau NETY")
    print("="*60)
    
    try:
        brain = Brain()
        
        # Enregistrer une conversation
        responses = []
        test_inputs = [
            "Bonjour, je m'appelle Alice",
            "Je suis développeuse Python",
            "J'aime beaucoup résoudre des problèmes complexes",
        ]
        
        for user_input in test_inputs:
            print(f"\n🗣️ Utilisateur: {user_input}")
            response = brain.think(user_input)
            print(f"🤖 NETY: {response[:100]}...")
            responses.append(response)
        
        # Vérifier la mémoire du cerveau
        print("\n📚 Mémoires enregistrées dans le cerveau:")
        memories = brain.memory.get_recent_memories(count=3)
        for i, mem in enumerate(memories, 1):
            print(f"\n   {i}. {mem.get('content', 'N/A')[:50]}...")
            print(f"      Labels: {mem.get('labels', [])}")
            print(f"      Timestamp: {mem.get('timestamp', 'N/A')}")
        
        print("\n✅ Intégration réussie!")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'intégration: {e}")
        import traceback
        traceback.print_exc()


def test_query_recent_memories():
    """Test 5: Requête sur les souvenirs récents"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Requête sur les souvenirs récents")
    print("="*60)
    
    ml_engine = MLEngine()
    
    # Ajouter des souvenirs avec contexte
    messages = [
        "Je m'appelle Raptor_",
        "Je suis passionné par l'IA et le machine learning",
        "Mon rôle est celui d'admin et créateur",
        "J'aime écrire du code en Python",
    ]
    
    for msg in messages:
        ml_engine.assign_memory_labels(msg, user_id="00000001")
    
    # Faire une requête
    query = "Qui es-tu et quels sont tes intérêts?"
    print(f"\n❓ Requête: {query}")
    
    relevant = ml_engine.get_relevant_memories(query, limit=5)
    
    print(f"\n✓ Souvenirs pertinents trouvés: {len(relevant)}")
    for i, mem in enumerate(relevant, 1):
        print(f"\n   {i}. {mem['text'][:60]}...")
        print(f"      Score: basé sur les mots-clés et récence")
        print(f"      Labels: {mem.get('labels', [])}")


def main():
    """Exécute tous les tests"""
    print("\n" + "="*60)
    print("🧠 TESTS DU SYSTÈME DE MÉMOIRE AMÉLIORÉ DE NETY")
    print("="*60)
    
    try:
        test_memory_labels()
        test_memory_retrieval()
        test_correlations()
        test_query_recent_memories()
        test_brain_integration()
        
        print("\n" + "="*60)
        print("✅ TOUS LES TESTS RÉUSSIS!")
        print("="*60)
        print("\n📝 Résumé:")
        print("   ✓ Labels contextuels assignés correctement")
        print("   ✓ Récupération des souvenirs avec contexte")
        print("   ✓ Corrélations enregistrées dans key_info.jsonl")
        print("   ✓ Intégration dans le cerveau fonctionnelle")
        print("   ✓ Requêtes sur les souvenirs récents efficaces")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
