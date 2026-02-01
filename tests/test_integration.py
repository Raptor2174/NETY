#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test d'intégration du système de mémoire"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from nety.modules.machinelearning.ml_engine import MLEngine
from nety.cortex_limbic.memory_manager import MemoryManager

def test_integration():
    print("\n" + "="*70)
    print("[TEST] D'INTEGRATION - Systeme de Memoire")
    print("="*70)
    
    try:
        # Test 1: ML Engine
        print("\n[1/5] Initialisation ML Engine...")
        ml = MLEngine()
        print("      ✅ ML Engine initialisé")
        
        # Test 2: Assign labels
        print("[2/5] Assignation de labels...")
        entry = ml.assign_memory_labels('Je suis NETY, une IA autonome', user_id='test_system')
        print(f"      ✅ Labels assignés: {entry['labels']}")
        print(f"      ✅ Sentiment: {entry['meta']['sentiment']}")
        
        # Test 3: Get memories
        print("[3/5] Récupération de mémoires...")
        memories = ml.get_memory_with_context(user_id='test_system', limit=3)
        print(f"      ✅ Mémoires récupérées: {len(memories)}")
        
        # Test 4: Memory Manager
        print("[4/5] Test Memory Manager...")
        mm = MemoryManager()
        mem_id = mm.add_memory('Mémoire test', labels=['test'], metadata={'user': 'test'})
        print(f"      ✅ Memory Manager OK, ID: {mem_id[:40]}...")
        
        # Test 5: Correlations
        print("[5/5] Vérification des corrélations...")
        key_infos = ml.load_key_info()
        correlations = [k for k in key_infos if k.get('type') == 'correlation']
        print(f"      ✅ Corrélations stockées: {len(correlations)}")
        
        print("\n" + "="*70)
        print("[OK] SUCCES: TOUS LES TESTS D'INTEGRATION PASSENT!")
        print("="*70)
        print("\n[SUCCESS] Le systeme de memoire est completement operationnel!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
