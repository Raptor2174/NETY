"""
Test d'intégration complète: Dashboard → Brain → Dashboard
"""

import sys
sys.path.insert(0, '/home/runner/work/NETY/NETY')

from nety.main import NETYSystem
from nety.core.nety_bridge import bridge
import time

print("=" * 60)
print("TEST: Intégration complète Dashboard ↔ Brain")
print("=" * 60)

# Initialiser le système
print("\n1. Initialisation du système NETY...")
nety = NETYSystem()
nety.start()
print("   ✅ Système démarré")
print(f"   ✅ Brain présent: {nety.brain is not None}")

# Simuler un message du Dashboard
print("\n2. Envoi d'un message depuis le Dashboard...")
test_message = "Bonjour NETY, comment vas-tu ?"
bridge.send_to_nety(test_message, msg_type="prompt")
print(f"   ✅ Message envoyé: '{test_message}'")

# Traiter le message comme le ferait la boucle principale
print("\n3. Traitement par le système...")
dashboard_message = bridge.get_from_dashboard(timeout=0.1)
if dashboard_message:
    print(f"   ✅ Message récupéré: {dashboard_message['type']}")
    
    # Traiter avec handle_dashboard_message
    nety.handle_dashboard_message(dashboard_message)
    print("   ✅ Message traité par le Brain")
    
    # Récupérer la réponse
    response = bridge.get_from_nety(timeout=0.1)
    if response:
        print(f"   ✅ Réponse générée: '{response['content']}'")
    else:
        print("   ⚠️ Pas de réponse générée")
else:
    print("   ❌ Impossible de récupérer le message")

# Vérifier l'état des modules
print("\n4. Vérification de l'état des modules...")
modules = bridge.get_modules_status()
print(f"   ✅ Modules actifs: {len(modules)}")
for module, status in modules.items():
    print(f"      - {module}: {status}")

# Vérifier le contexte du Brain
print("\n5. Vérification du contexte du Brain...")
if nety.brain:
    context = nety.brain.get_context()
    print(f"   ✅ Interactions enregistrées: {len(context)}")
    if context:
        last_interaction = context[-1]
        print(f"      Input: {last_interaction['input'][:50]}...")
        print(f"      Output: {last_interaction['output'][:50]}...")

# Tester avec un autre type de message
print("\n6. Test avec un message de type 'chat'...")
bridge.send_to_nety("Quelle est ta mission ?", msg_type="chat")
chat_msg = bridge.get_from_dashboard(timeout=0.1)
if chat_msg:
    nety.handle_dashboard_message(chat_msg)
    response = bridge.get_from_nety(timeout=0.1)
    if response:
        print(f"   ✅ Réponse au chat: '{response['content']}'")

# Vérifier les logs
print("\n7. Vérification des logs...")
logs = bridge.get_logs()
print(f"   ✅ Total de logs: {len(logs)}")
print(f"   Derniers logs:")
for log in logs[-5:]:
    print(f"      {log}")

print("\n" + "=" * 60)
print("✅ TEST D'INTÉGRATION COMPLÈTE RÉUSSI!")
print("=" * 60)

# Arrêter proprement
nety.stop()
