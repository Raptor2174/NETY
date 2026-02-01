# 🧠 Système de Mémoire NETY - Démarrage Rapide

## ⚡ En 30 secondes

NETY peut maintenant **SE SOUVENIR** des conversations avec:
- ✅ **Labels** (qui es-tu, tes objectifs, préférences)
- ✅ **Sentiment** (heureux/triste/neutre)
- ✅ **Corrélations** (liens entre informations)
- ✅ **Contexte** (récupération intelligente)

---

## 🚀 Test Rapide

```bash
# Vérifier que tout fonctionne (30 secondes)
python verify_memory_system.py
```

**Résultat attendu:**
```
✅ SYSTÈME DE MÉMOIRE: OPÉRATIONNEL
✓ Se souvenir des conversations
✓ Labels contextuels
✓ Sentiment analysis
✓ Évaluation d'urgence
✓ Corrélations entre informations
✓ Récupération contextuelle intelligente
```

---

## 💻 Utilisation Basique

### 1. Enregistrer un Souvenir
```python
from nety.modules.machinelearning.ml_engine import MLEngine

ml_engine = MLEngine()

# Enregistrer
entry = ml_engine.assign_memory_labels(
    "Je m'appelle Raptor_ et j'aime la programmation",
    user_id="00000001"
)

# Résultat
print(entry['labels'])      # ['identity_info', 'preference', 'short_interaction']
print(entry['meta']['sentiment'])  # 'positive'
print(entry['facts'])       # {'name': ['Raptor_'], 'likes': ['la programmation']}
```

### 2. Récupérer les Souvenirs Récents
```python
# Récupérer les 5 derniers souvenirs
memories = ml_engine.get_memory_with_context(
    user_id="00000001",
    limit=5
)

for mem in memories:
    print(f"- {mem['text']}")
    print(f"  Labels: {mem['labels']}")
    print(f"  Sentiment: {mem['meta']['sentiment']}")
```

### 3. Chercher des Souvenirs Pertinents
```python
# Poser une question
query = "Dis-moi qui tu es?"
relevant = ml_engine.get_relevant_memories(query, limit=3)

for mem in relevant:
    print(f"- {mem['text'][:50]}...")
    print(f"  Labels: {mem['labels']}")
```

---

## 📊 Exemple Complet

```python
from nety.modules.machinelearning.ml_engine import MLEngine
from nety.cortex_limbic.memory_manager import MemoryManager

ml_engine = MLEngine()
mem_manager = MemoryManager()

# Conversation
messages = [
    "Bonjour, je m'appelle Alice",
    "Je suis développeuse Python",
    "J'aime les défis complexes",
    "Mon objectif est de maîtriser le machine learning"
]

# Enregistrer chaque message
for msg in messages:
    entry = ml_engine.assign_memory_labels(msg, user_id="alice")
    print(f"✓ Enregistré: {msg}")
    print(f"  Labels: {entry['labels']}\n")

# Chercher des informations
print("\n--- Récupération ---")
identity = ml_engine.get_relevant_memories("Qui es-tu?", limit=3)
for mem in identity:
    print(f"- {mem['text']}")
```

---

## 🎯 Types de Labels

| Label | Signification | Exemple |
|-------|---------------|---------|
| `identity_info` | Information d'identité | "Je suis Alice" |
| `preference` | Préférence | "J'aime Python" |
| `goal` | Objectif | "Je veux apprendre le ML" |
| `health_update` | Santé | "J'ai mal à la tête" |
| `detailed_context` | Message long | Description détaillée |
| `short_interaction` | Message court | Messages brefs |

---

## 📁 Fichiers de Stockage

```
~/data/processed/ml_engine/
├── memory.jsonl         ← Souvenirs avec labels
├── key_info.jsonl       ← Corrélations
└── stats.json           ← Statistiques
```

---

## 🧪 Tests

```bash
# Test allégé (rapide, recommandé)
python tests/test_memory_lite.py

# Test complet (inclut Brain)
python tests/test_memory_system.py
```

---

## 📚 Documentation Complète

1. **MEMORY_SYSTEM_UPGRADE_2026.md** - Rapport technique
2. **MEMORY_USAGE_GUIDE.md** - Guide d'utilisation détaillé
3. **MEMORY_IMPROVEMENTS_SUMMARY.md** - Résumé des changements
4. **RAPPORT_FINAL_MEMORY_SYSTEM.md** - Rapport final

---

## ❓ FAQ Rapide

### Q: Où sont stockés les souvenirs?
**R:** Dans `data/processed/ml_engine/memory.jsonl`

### Q: Comment NETY se souvient?
**R:** Avec labels contextuels, sentiment, et corrélations

### Q: Les vieux souvenirs sont supprimés?
**R:** Non, tous sont gardés mais les récents sont priorisés

### Q: Comment attribuer un user_id?
**R:** Automatiquement détecté dans le Brain, ou spécifié en parameter

### Q: Puis-je filtrer par labels?
**R:** Oui! `mem_manager.get_memories_by_label("goal")`

---

## ✅ Checklist de Démarrage

- [ ] Exécuter `python verify_memory_system.py`
- [ ] Lire `MEMORY_USAGE_GUIDE.md`
- [ ] Exécuter `python tests/test_memory_lite.py`
- [ ] Essayer les exemples ci-dessus
- [ ] Lire `MEMORY_SYSTEM_UPGRADE_2026.md`

---

## 🎉 Résultat Final

```
NETY se souvient maintenant! 🧠✨

✓ Reconnaît qui tu es
✓ Comprend ce que tu aimes
✓ Connaît tes objectifs
✓ Retient les informations clés
✓ Crée des corrélations
✓ Récupère le contexte approprié
```

---

**Status:** ✅ Production Ready  
**Version:** 2.0  
**Date:** 1er Février 2026

🚀 **PRÊT À UTILISER!**
