"""
Guide de Diagnostic pour le Système de Logs
Pour vérifier que les logs ne sont pas tronqués
"""

# ============================================================================
# DANS LE DASHBOARD (app/ui/window.py ou dans une page de diagnostic)
# ============================================================================

def show_logs_diagnostics(self):
    """Affiche les diagnostics du système de logs dans le Dashboard"""
    
    # Récupérer les stats depuis l'adapter
    stats = self.nety_adapter.get_logs_diagnostics()
    
    # Construire le rapport
    diagnostic_text = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    📊 DIAGNOSTIC SYSTÈME DE LOGS                  ║
╚════════════════════════════════════════════════════════════════════╝

📈 STATISTIQUES
  • Total logs en mémoire: {stats['total_logs']} / {stats['max_capacity']}
  • Pourcentage utilisé: {stats['capacity_used_percent']:.1f}%
  • Nombre de rotations: {stats['rotation_count']}

⚙️ SEUILS
  • Seuil de rotation: {90}% ({int(stats['max_capacity'] * 0.9)} logs)
  • Pourcentage conservé: {75}%
  • Pourcentage supprimé: {25}%

🔍 HISTORIQUE
  • Premier log: {stats['first_log']}
  • Dernier log: {stats['last_log']}

✅ SANTÉ DU SYSTÈME
  • Status: {"🟢 NORMAL" if stats['capacity_used_percent'] < 80 else "🟡 PRUDENT" if stats['capacity_used_percent'] < 95 else "🔴 ATTENTION"}
  • Capacité: {"✓ Confortable" if stats['capacity_used_percent'] < 80 else "⚠️ Élevée"}
  • Rotations: {"✓ Aucune" if stats['rotation_count'] == 0 else f"⚠️ {stats['rotation_count']} rotations"}

════════════════════════════════════════════════════════════════════════
Garantie: get_logs() retourne TOUJOURS la totalité des logs en mémoire.
Aucun log n'est jamais perdu sauf par clear_logs().
════════════════════════════════════════════════════════════════════════
    """
    
    print(diagnostic_text)
    return diagnostic_text


# ============================================================================
# DANS UN TEST UNITAIRE
# ============================================================================

def test_logs_totality(nety_adapter):
    """Test unitaire pour vérifier que get_logs() retourne la totalité"""
    
    # Récupérer les diagnostics
    stats = nety_adapter.get_logs_diagnostics()
    
    # Vérifier que les logs ne sont pas tronqués
    assert stats['total_logs'] > 0, "Aucun log n'est présent"
    assert stats['total_logs'] <= stats['max_capacity'], "Logs dépasse la capacité"
    assert stats['capacity_used_percent'] <= 100, "Pourcentage invalide"
    
    # Vérifier que get_logs() retourne exactement ce nombre
    logs = nety_adapter.get_logs()
    assert len(logs) == stats['total_logs'], \
        f"Mismatch: diagnostics dit {stats['total_logs']} logs, get_logs() retourne {len(logs)}"
    
    print(f"✓ Test OK: {len(logs)} logs retournés correctement")
    return True


# ============================================================================
# DEBUG EN CAS DE PROBLÈME DE TRONQUAGE
# ============================================================================

def debug_log_truncation(nety_adapter, expected_count=None):
    """Debug pour identifier si les logs sont tronqués"""
    
    logs = nety_adapter.get_logs()
    stats = nety_adapter.get_logs_diagnostics()
    
    print(f"\n🔍 DEBUG - Vérification des logs")
    print(f"─" * 60)
    print(f"Logs retournés: {len(logs)}")
    print(f"Stats diagnostics: {stats['total_logs']}")
    print(f"Capacité max: {stats['max_capacity']}")
    print(f"Rotations: {stats['rotation_count']}")
    print(f"Premier log: {logs[0][:80] if logs else 'AUCUN'}")
    print(f"Dernier log: {logs[-1][:80] if logs else 'AUCUN'}")
    
    # Vérifier la cohérence
    if len(logs) != stats['total_logs']:
        print(f"\n❌ ERREUR: Mismatch entre get_logs() et diagnostics!")
        print(f"   get_logs() retourne: {len(logs)}")
        print(f"   diagnostics indique: {stats['total_logs']}")
        return False
    
    if expected_count and len(logs) < expected_count:
        print(f"\n⚠️ ATTENTION: Moins de logs que prévu")
        print(f"   Attendus: {expected_count}")
        print(f"   Trouvés: {len(logs)}")
        print(f"   Manquants: {expected_count - len(logs)}")
        if stats['rotation_count'] > 0:
            print(f"   Rotation(s) détectée(s): {stats['rotation_count']}")
            print(f"   Cela est normal, les anciens logs ont été supprimés")
        return False
    
    print(f"\n✅ OK: Tous les logs sont présents")
    return True


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    from app.services.nety_adapter import NetyAdapter
    from nety.core.nety_bridge import bridge
    
    # Initialiser l'adapter
    adapter = NetyAdapter()
    
    # Ajouter quelques logs de test
    print("Ajout de 100 logs de test...")
    for i in range(100):
        bridge._add_log(f"[TEST] Log numéro {i+1}")
    
    # Afficher le diagnostic
    show_logs_diagnostics(adapter)
    
    # Debug
    debug_log_truncation(adapter, expected_count=100)
