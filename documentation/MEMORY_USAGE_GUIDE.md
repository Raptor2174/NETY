# 📚 Guide d'Utilisation - Système de Mémoire NETY

## 🎯 Vue d'ensemble

NETY utilise maintenant un **système de mémoire intelligent** avec:
- ✅ **Labels contextuels** pour catégoriser les souvenirs
- ✅ **Corrélations** entre les informations
- ✅ **Sentiment analysis** pour l'émotionnel
- ✅ **Récupération intelligente** des souvenirs récents

---

## 🔧 API Principale

### 1. Enregistrer un Souvenir avec Labels

```python
from nety.modules.machinelearning.ml_engine import MLEngine

ml_engine = MLEngine()

# Enregistrer un message utilisateur
memory_entry = ml_engine.assign_memory_labels(
    text="Je m'appelle Raptor_ et j'adore la programmation",
    user_id="00000001"
)

# Le résultat contient:
print(memory_entry)
# {
#     "id": "2026-02-01T19:31:36.623256-37",
#     "text": "Je m'appelle Raptor_ et j'adore la programmation",
#     "labels": ["identity_info", "preference", "short_interaction"],
#     "sentiment": "positive",
#     "facts": {
#         "name": ["Raptor_"],
#         "likes": ["programmation"]
#     },
#     "categories": ["identity", "preferences"],
#     "keywords": ["raptor", "programmation", "adore"],
#     "meta": {
#         "sentiment": "positive",
#         "urgency": "low"
#     }
# }
```

### 2. Récupérer les Souvenirs Récents

```python
# Récupérer les 5 derniers souvenirs d'un utilisateur
recent = ml_engine.get_memory_with_context(
    user_id="00000001",
    limit=5
)

for memory in recent:
    print(f"Texte: {memory['text']}")
    print(f"Labels: {memory['labels']}")
    print(f"Timestamp: {memory['timestamp']}")
```

### 3. Chercher des Souvenirs Pertinents

```python
# Trouver les souvenirs pertinents pour une requête
query = "Qui es-tu et quels sont tes intérêts?"
relevant = ml_engine.get_relevant_memories(query, limit=5)

# Résultats classés par:
# 1. Recency (souvenirs récents en priorité)
# 2. Keywords matching
# 3. Facts correlation
for memory in relevant:
    print(f"- {memory['text'][:50]}...")
    print(f"  Labels: {memory['labels']}")
```

### 4. Récupérer par Label

```python
from nety.cortex_limbic.memory_manager import MemoryManager

mem_manager = MemoryManager()

# Récupérer tous les souvenirs avec le label "identity_info"
identity_memories = mem_manager.get_memories_by_label("identity_info")

# Récupérer tous les souvenirs avec le label "goal"
goal_memories = mem_manager.get_memories_by_label("goal")
```

---

## 📌 Types de Labels

### Labels Contextuels
| Label | Description | Exemple |
|-------|-------------|---------|
| `identity_info` | Information d'identité | "Je m'appelle Raptor_" |
| `preference` | Préférence utilisateur | "J'aime le café" |
| `goal` | Objectif utilisateur | "Je veux maîtriser le ML" |
| `health_update` | État de santé | "J'ai mal à la tête" |
| `detailed_context` | Message long (>50 mots) | Descriptions détaillées |
| `short_interaction` | Message court (<50 mots) | Messages brefs |

### Sentiment
- ✅ `positive` - Sentiments positifs
- ➖ `negative` - Sentiments négatifs
- ⚪ `neutral` - Pas d'émotion marquée

### Urgence
- 🔴 `high` - Urgent (bug, problème immédiat)
- 🟡 `medium` - Moyen terme (tâche à faire)
- 🟢 `low` - Pas urgent

---

## 🔍 Exemples Pratiques

### Exemple 1: Se Souvenir d'une Conversation

```python
from nety.core.brain import Brain

brain = Brain()

# Conversation utilisateur
messages = [
    "Bonjour, je m'appelle Alice",
    "Je suis développeuse Python",
    "J'aime résoudre des problèmes complexes",
    "Mon objectif est de devenir expert en IA"
]

for msg in messages:
    response = brain.think(msg)
    print(f"User: {msg}")
    print(f"NETY: {response}\n")

# Les souvenirs sont automatiquement enregistrés avec labels!
```

### Exemple 2: Requête sur la Mémoire

```python
ml_engine = MLEngine()

# Poser une question basée sur les souvenirs
query = "Dis-moi qui tu es"
memories = ml_engine.get_relevant_memories(query, limit=5)

print(f"Réponse basée sur les souvenirs:")
for mem in memories:
    print(f"- {mem['text']}")
    print(f"  (Labels: {mem.get('labels', [])})\n")
```

### Exemple 3: Analyser le Sentiment

```python
ml_engine = MLEngine()

# Enregistrer plusieurs messages
messages = [
    "Je suis très content du résultat!",
    "C'est horrible, ça ne marche pas",
    "Les choses avancent normalement"
]

for msg in messages:
    entry = ml_engine.assign_memory_labels(msg, user_id="user123")
    print(f"Sentiment de '{msg}':")
    print(f"  → {entry['meta']['sentiment']}\n")
```

### Exemple 4: Créer des Corrélations

```python
ml_engine = MLEngine()

# Enregistrer des informations liées
ml_engine.assign_memory_labels(
    "Je travaille chez OpenAI",
    user_id="raptor"
)

ml_engine.assign_memory_labels(
    "Mon email professionnel est raptor@openai.com",
    user_id="raptor"
)

# Les corrélations sont automatiquement créées!
# Le système sait maintenant que la personne travaille chez OpenAI
# et peut utiliser cette information pour de futures interactions
```

---

## 🗄️ Structure de Données

### Memory Entry
```json
{
    "id": "2026-02-01T19:31:36.623256-37",
    "timestamp": "2026-02-01T19:31:36.623256",
    "text": "Je m'appelle Raptor_ et je suis admin",
    "facts": {
        "name": ["Raptor_"],
        "traits": ["admin"]
    },
    "categories": ["identity"],
    "keywords": ["raptor", "admin"],
    "user_id": "00000001",
    "labels": ["identity_info", "short_interaction"],
    "meta": {
        "sentiment": "neutral",
        "urgency": "low"
    }
}
```

### Correlation Entry (key_info.jsonl)
```json
{
    "type": "correlation",
    "field": "name",
    "value": "Raptor_",
    "user_id": "00000001",
    "memory_id": "2026-02-01T19:31:36.623256-37",
    "timestamp": "2026-02-01T19:31:36.623256",
    "category": "identity",
    "labels": ["identity_info", "short_interaction"],
    "sentiment": "neutral"
}
```

---

## 🚀 Utilisation dans Brain

Le Brain utilise automatiquement ce système:

```python
# Dans brain.py process_message():

# 1. Enregistrer avec labels
memory_entry = self.ml_engine.assign_memory_labels(
    message, 
    user_id=user_id
)

# 2. Ajouter à la mémoire locale
self.memory.add_memory(
    message[:100],
    labels=memory_entry.get("labels"),
    metadata={
        "user_id": user_id,
        "categories": memory_entry.get("categories"),
        "sentiment": memory_entry.get("meta", {}).get("sentiment")
    }
)

# 3. Récupérer le contexte enrichi
context = self.retrieve_context(message, intent)
# context contient maintenant:
# - recent_labels: ['identity_info', 'goal']
# - memory_sentiment: 'positive'
```

---

## 💾 Fichiers de Stockage

### memory.jsonl
```bash
~/data/processed/ml_engine/memory.jsonl
```
Chaque ligne = une entrée mémoire avec labels et métadonnées

### key_info.jsonl
```bash
~/data/processed/ml_engine/key_info.jsonl
```
Les corrélations entre les informations clés

### stats.json
```bash
~/data/processed/ml_engine/stats.json
```
Statistiques globales (total d'entrées, catégories)

---

## 🧪 Tests

### Exécuter les Tests de Mémoire

```bash
# Test complet
python tests/test_memory_system.py

# Test allégé (rapide)
python tests/test_memory_lite.py
```

---

## ⚙️ Configuration

Pas de configuration nécessaire! Le système fonctionne automatiquement.

Mais vous pouvez ajuster les paramètres:

```python
ml_engine = MLEngine(
    data_dir="/custom/path/to/data",  # Chemin personnalisé
    model=custom_model  # Modèle personnalisé
)
```

---

## 🔐 Bonnes Pratiques

### ✅ À Faire

```python
# ✅ Toujours fournir un user_id
memory = ml_engine.assign_memory_labels(text, user_id="user123")

# ✅ Récupérer avec contexte
memories = ml_engine.get_memory_with_context(user_id="user123", limit=10)

# ✅ Filtrer par labels
identity = mem_manager.get_memories_by_label("identity_info")
```

### ❌ À Éviter

```python
# ❌ Ne pas ignorer les corrélations
# Le système les gère automatiquement!

# ❌ Ne pas stocker des données sensibles directement
# Penser à la sécurité

# ❌ Ne pas utiliser des querys sans limite
# Toujours spécifier limit=N
```

---

## 🆘 Dépannage

### Problème: Les souvenirs ne sont pas retrouvés

```python
# Solution: Vérifier user_id
memories = ml_engine.get_memory_with_context(user_id="exact_user_id")

# Les souvenirs sans user_id sont aussi retrouvés
memories = ml_engine.get_memory_with_context(limit=10)
```

### Problème: Labels non assignés

```python
# Vérifier que le message contient du contenu
# Les labels sont assignés automatiquement basé sur le contenu

# Le minimum: `short_interaction` est toujours assigné
entry = ml_engine.assign_memory_labels("test")
print(entry['labels'])  # Au moins ['short_interaction']
```

### Problème: Pas de corrélations

```python
# Les corrélations ne sont créées que si des faits sont extraits
# Vérifier entry['facts'] n'est pas vide

entry = ml_engine.assign_memory_labels("Je m'appelle Raptor_")
print(entry['facts'])  # Devrait contenir {'name': ['Raptor_']}
```

---

## 📞 Support

Pour plus d'informations:
- Voir `documentation/MEMORY_SYSTEM_UPGRADE_2026.md`
- Exécuter `tests/test_memory_lite.py` pour voir les fonctionnalités
- Consulter le code dans `nety/modules/machinelearning/ml_engine.py`

---

**Version:** 2.0  
**Date:** 1er Février 2026  
**Status:** ✅ Opérationnel
