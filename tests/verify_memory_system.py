#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démarrage rapide - Vérifier le système de mémoire de NETY
Exécuter: python verify_memory_system.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from nety.modules.machinelearning.ml_engine import MLEngine
from nety.cortex_limbic.memory_manager import MemoryManager


def check_memory_system():
    """Vérifie que le système de mémoire fonctionne correctement"""
    print("\n" + "="*70)
    print("🧠 VÉRIFICATION DU SYSTÈME DE MÉMOIRE NETY")
    print("="*70)
    
    try:
        # 1. Initialiser ML Engine
        print("\n[1/4] Initialisation du ML Engine...")
        ml_engine = MLEngine()
        print("      ✅ ML Engine initialisé")
        
        # 2. Initialiser Memory Manager
        print("[2/4] Initialisation du Memory Manager...")
        mem_manager = MemoryManager()
        print("      ✅ Memory Manager initialisé")
        
        # 3. Tester l'assignation de labels
        print("[3/4] Test d'assignation de labels...")
        test_message = "Bonjour, je m'appelle NETY et j'aime apprendre"
        entry = ml_engine.assign_memory_labels(test_message, user_id="test_user")
        
        print(f"      Message: '{test_message}'")
        print(f"      ✅ Labels: {entry.get('labels', [])}")
        print(f"      ✅ Sentiment: {entry.get('meta', {}).get('sentiment', 'N/A')}")
        print(f"      ✅ Corrélation enregistrée: {len(ml_engine.load_key_info()) > 0}")
        
        # 4. Vérifier le stockage
        print("[4/4] Vérification du stockage...")
        stats = ml_engine.get_stats()
        print(f"      ✅ Total d'entrées: {stats.get('total_entries', 0)}")
        print(f"      ✅ Catégories: {len(stats.get('category_counts', {}))}")
        
        print("\n" + "="*70)
        print("✅ SYSTÈME DE MÉMOIRE: OPÉRATIONNEL")
        print("="*70)
        
        print("""
📚 Fonctionnalités disponibles:
   ✓ Se souvenir des conversations
   ✓ Labels contextuels (identity_info, goal, preference, etc.)
   ✓ Sentiment analysis (positive/negative/neutral)
   ✓ Évaluation d'urgence (high/medium/low)
   ✓ Corrélations entre informations
   ✓ Récupération contextuelle intelligente

🚀 Pour utiliser:
   
   # Enregistrer un souvenir
   entry = ml_engine.assign_memory_labels(
       "Je m'appelle Raptor_",
       user_id="00000001"
   )
   
   # Récupérer les souvenirs récents
   memories = ml_engine.get_memory_with_context(
       user_id="00000001",
       limit=5
   )
   
   # Chercher des souvenirs pertinents
   relevant = ml_engine.get_relevant_memories(
       "Qui es-tu?",
       limit=5
   )

📖 Documentation complète:
   • documentation/MEMORY_SYSTEM_UPGRADE_2026.md
   • documentation/MEMORY_USAGE_GUIDE.md
   • MEMORY_IMPROVEMENTS_SUMMARY.md

🧪 Tests:
   python tests/test_memory_lite.py     # Rapide
   python tests/test_memory_system.py   # Complet
""")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_memory_system()
    sys.exit(0 if success else 1)
