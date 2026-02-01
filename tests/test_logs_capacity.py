"""
Script de test pour vérifier que les logs ne sont pas tronqués
Test que get_logs() retourne bien la totalité des logs
"""

import sys
sys.path.insert(0, r'c:\Users\trexm\Documents\NETY')

from app.services.nety_adapter import NetyAdapter
from nety.core.nety_bridge import bridge
import time

def test_logs_capacity():
    """Test que les logs gèrent bien la capacité sans truncation"""
    
    print("=" * 80)
    print("TEST: Vérification de la capacité des logs")
    print("=" * 80)
    
    adapter = NetyAdapter()
    
    # Ajouter beaucoup de logs
    print("\n📝 Ajout de 3000 logs de test...")
    for i in range(3000):
        bridge._add_log(f"Log test #{i+1:04d}")
        if (i + 1) % 500 == 0:
            print(f"  ✓ {i+1} logs ajoutés")
    
    # Vérifier les stats
    stats = adapter.get_logs_diagnostics()
    print("\n📊 STATISTIQUES DES LOGS:")
    print(f"  Total logs: {stats['total_logs']}")
    print(f"  Capacité max: {stats['max_capacity']}")
    print(f"  Pourcentage utilisé: {stats['capacity_used_percent']:.1f}%")
    print(f"  Rotations effectuées: {stats['rotation_count']}")
    print(f"  Premier log: {stats['first_log'][:60]}...")
    print(f"  Dernier log: {stats['last_log'][:60]}...")
    
    # Vérifier que get_logs() retourne la totalité
    all_logs = adapter.get_logs()
    print(f"\n✓ get_logs() retourne {len(all_logs)} logs")
    
    # Vérifier que aucun log n'est tronqué
    print("\n🔍 Vérification de l'intégrité des logs...")
    found_logs = set()
    for log in all_logs:
        # Extraire le numéro du log si c'est un log de test
        if "Log test #" in log:
            parts = log.split("Log test #")
            if len(parts) > 1:
                num_str = parts[1].split("]")[0]
                try:
                    num = int(num_str)
                    found_logs.add(num)
                except ValueError:
                    pass
    
    if found_logs:
        min_log = min(found_logs)
        max_log = max(found_logs)
        total_found = len(found_logs)
        print(f"  Logs trouvés: {total_found}")
        print(f"  Plage: #{min_log:04d} à #{max_log:04d}")
        
        # Vérifier la continuité
        missing = []
        for i in range(min_log, max_log + 1):
            if i not in found_logs:
                missing.append(i)
        
        if not missing:
            print(f"  ✓ Aucun log manquant dans la plage!")
        else:
            print(f"  ⚠️ {len(missing)} logs manquants: {missing[:10]}...")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLET")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    test_logs_capacity()
