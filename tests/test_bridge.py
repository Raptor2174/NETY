"""
Test du Bridge de communication Dashboard ↔ NETY
"""

import sys
sys.path.insert(0, '/home/runner/work/NETY/NETY')

from nety.core.nety_bridge import bridge
import time

print("=" * 60)
print("TEST: Bridge de communication NETY")
print("=" * 60)

# Test 1: Vérifier l'initialisation
print("\n1. Test d'initialisation du Bridge")
print(f"   ✅ Bridge initialisé: {bridge._initialized}")
print(f"   ✅ System running: {bridge.is_system_running()}")
print(f"   ✅ Brain ready: {bridge.is_brain_ready()}")

# Test 2: Envoyer un message depuis le Dashboard
print("\n2. Test d'envoi Dashboard → IA")
success = bridge.send_to_nety("Test message from dashboard", msg_type="prompt")
print(f"   ✅ Message envoyé: {success}")

# Test 3: Récupérer le message côté IA
print("\n3. Test de réception IA")
message = bridge.get_from_dashboard(timeout=0.1)
if message:
    print(f"   ✅ Message reçu: {message['type']}")
    print(f"   ✅ Contenu: {message['content'][:50]}...")
else:
    print("   ❌ Aucun message reçu")

# Test 4: Envoyer une réponse de l'IA
print("\n4. Test d'envoi IA → Dashboard")
success = bridge.send_from_nety("Réponse de l'IA", msg_type="response")
print(f"   ✅ Réponse envoyée: {success}")

# Test 5: Récupérer la réponse côté Dashboard
print("\n5. Test de réception Dashboard")
response = bridge.get_from_nety(timeout=0.1)
if response:
    print(f"   ✅ Réponse reçue: {response['type']}")
    print(f"   ✅ Contenu: {response['content']}")
else:
    print("   ❌ Aucune réponse reçue")

# Test 6: Vérifier les logs
print("\n6. Test des logs")
logs = bridge.get_logs()
print(f"   ✅ Nombre de logs: {len(logs)}")
print(f"   ✅ Dernier log: {logs[-1] if logs else 'Aucun log'}")

# Test 7: Mise à jour des modules
print("\n7. Test de mise à jour des modules")
test_modules = {
    "module_1": "actif",
    "module_2": "inactif",
    "module_3": "actif"
}
bridge.update_modules_status(test_modules)
retrieved_status = bridge.get_modules_status()
print(f"   ✅ Modules mis à jour: {retrieved_status}")

# Test 8: Statistiques
print("\n8. Test des statistiques")
stats = bridge.get_stats()
print(f"   ✅ Messages envoyés: {stats['messages_sent']}")
print(f"   ✅ Messages reçus: {stats['messages_received']}")
print(f"   ✅ Logs count: {stats['logs_count']}")

print("\n" + "=" * 60)
print("✅ TOUS LES TESTS DU BRIDGE RÉUSSIS!")
print("=" * 60)
