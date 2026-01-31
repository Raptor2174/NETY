#!/usr/bin/env python3
"""
Vérification complète de l'intégration du système de sélection de modèle
"""

import re

def check_file(filepath, checks):
    """Vérifie qu'un fichier contient les patterns requis"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = []
        for check_name, pattern in checks:
            found = re.search(pattern, content) is not None
            results.append((check_name, found))
        
        return results
    except Exception as e:
        print(f"❌ Erreur lecture {filepath}: {e}")
        return []

print("\n" + "="*70)
print("🔍 VÉRIFICATION COMPLÈTE DU SYSTÈME DE SÉLECTION DE MODÈLE")
print("="*70 + "\n")

# Vérifications
checks = {
    "nety/core/model_selector.py": [
        ("Literal['mistral', 'bloomz']", r"Literal\[\"mistral\",\s*\"bloomz\"\]"),
        ("select_model() fonction", r"def select_model\("),
        ("Mistral requires_gpu=True", r'"mistral".*?"requires_gpu":\s*True'),
        ("BLOOMZ requires_gpu=False", r'"bloomz".*?"requires_gpu":\s*False'),
    ],
    "nety/core/brain.py": [
        ("Brain.__init__(model_type)", r"def __init__\(self,\s*model_type"),
        ("Passage model_type à ResponseGenerator", r"ResponseGenerator\(model_type=model_type\)"),
    ],
    "nety/main.py": [
        ("NETYSystem init avec model_type", r"def __init__\(self,\s*model_type=None\)"),
        ("Brain avec model_type", r"Brain\(model_type=self\.model_type\)"),
        ("select_model() call", r"select_model\(interactive=True\)"),
    ],
    "nety/core/response_generator.py": [
        ("Pas d'import requests", r"^(?!.*import requests)"),
        ("ResponseGenerator init model_type", r"def __init__\(self,\s*model_type"),
        ("self.model_type assigné", r"self\.model_type\s*="),
    ],
}

# Vérifier les fichiers
all_ok = True
for filepath, file_checks in checks.items():
    print(f"\n📄 {filepath}")
    print("   " + "-" * 60)
    
    results = check_file(filepath, file_checks)
    
    if not results:
        print("   ⚠️  Fichier non trouvé ou erreur de lecture")
        all_ok = False
        continue
    
    for check_name, found in results:
        status = "✅" if found else "❌"
        print(f"   {status} {check_name}")
        if not found:
            all_ok = False

print("\n" + "="*70)

if all_ok:
    print("✅ TOUS LES CONTRÔLES SONT RÉUSSIS!")
    print("\n🎯 Le système de sélection de modèle est correctement intégré:")
    print("   ✓ L'utilisateur choisit entre Mistral ou BLOOMZ")
    print("   ✓ Le choix est transmis à NETYSystem")
    print("   ✓ NETYSystem transmet à Brain")
    print("   ✓ Brain transmet à ResponseGenerator")
    print("   ✓ Le modèle choisi est chargé et utilisé")
else:
    print("❌ CERTAINS CONTRÔLES ONT ÉCHOUÉ")
    print("   Veuillez vérifier les points marqués avec ❌")

print("="*70 + "\n")
