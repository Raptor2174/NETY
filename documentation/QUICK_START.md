# 🚀 QUICK START - Cortex Textuel RNN

## TL;DR (Trop Long; Pas Lu)

```python
from nety.core.brain import Brain

brain = Brain()  # ✅ Cortex créé automatiquement!
response = brain.think("Bonjour")  # ✅ Utilise le cortex automatiquement!
print(response)
```

---

## 5 Minutes Pour Comprendre

### Qu'est-ce qui a changé?

**Ancien RNN:**
```
Simple LSTM (2 couches) → Output
```

**Nouveau Cortex Textuel:**
```
Embedding
  ↓
Attention multi-tête (4)
  ↓
Bi-LSTM (3 couches)
  ↓
État PERSISTANT entre interactions
  ↓
Modulation émotionnelle
  ↓
Output
```

### 3 Faits Clés

1. **État Persistant** 🧠
   - Le réseau "se souvient" entre les messages
   - Mémoire à court-terme via état LSTM
   - Mémoire contextuelle via fenêtre glissante

2. **Modulation Émotionnelle** ❤️
   - Émotions du limbic system modulent l'activation
   - Réponses plus empathiques
   - Adaptation au contexte

3. **Production Ready** ✅
   - ~2500 lignes de code new
   - Tests complets (5 suites)
   - Documentation complete

---

## Lancer les Tests (30 secondes)

```bash
python tests/test_textual_cortex.py
```

Résultat:
```
✅ TEST 1: Modèle RNN Brut
✅ TEST 2: Cortex Autonome
✅ TEST 3: Modulation Émotionnelle
✅ TEST 4: Statistiques
✅ TEST 5: Persistance d'État
✅ TOUS LES TESTS RÉUSSIS!
```

---

## Lancer la Démo (2 minutes)

```bash
# Mode automatisé (4 messages)
python scripts/demo_rnn_cortex.py --mode demo

# Mode interactif (conversation libre)
python scripts/demo_rnn_cortex.py --mode interactive
```

---

## Utilisation Simple

```python
from nety.core.brain import Brain

# Initialiser
brain = Brain()

# Message 1
response = brain.think("Bonjour!")
print(response)

# Message 2 - Le cortex se souvient!
response = brain.think("Comment tu ça marche?")
print(response)

# Accéder aux stats
stats = brain.textual_cortex.get_neural_statistics()
print(f"Activation: {stats['current_activation']:.3f}")
print(f"Profondeur: {stats['context_depth']}")
```

---

## Utilisation Avancée

```python
from nety.cortex_limbic.textual_cortex import TextualCortex
import torch

cortex = TextualCortex()

# Créer un embedding (768 dimensions)
embedding = torch.randn(1, 5, 768)

# Traiter avec modulation émotionnelle
emotional_context = {"emotions": {"joie": 0.8, "tristesse": 0.1}}

output, metadata = cortex.process_text_sequence(
    embedding,
    emotional_context=emotional_context,
    use_persistent_state=True  # Garder l'état
)

print(f"Activation: {metadata['activation_level']:.3f}")
print(f"Output shape: {output.shape}")  # (1, 512)
```

---

## Fichiers Clés

| Fichier | Usage |
|---------|-------|
| `nety/modules/text/modele_rnn.py` | Modèle RNN modernisé |
| `nety/cortex_limbic/textual_cortex.py` | Cortex wrapper |
| `nety/core/brain.py` | Brain avec cortex intégré |
| `documentation/CORTEX_TEXTUEL_RNN.md` | Doc technique complète |
| `CORTEX_TEXTUEL_INTEGRATION_GUIDE.md` | Guide d'intégration |
| `tests/test_textual_cortex.py` | Tests |
| `scripts/demo_rnn_cortex.py` | Démos |

---

## Vérifier l'Intégration

```python
from nety.core.brain import Brain

brain = Brain()

# Vérifier le cortex
assert hasattr(brain, 'textual_cortex'), "Cortex not found!"
print("✅ Cortex est présent")

# Vérifier les méthodes
assert hasattr(brain.textual_cortex, 'process_text_sequence')
assert hasattr(brain.textual_cortex, 'get_neural_statistics')
print("✅ Toutes les méthodes sont disponibles")

# Test simple
response = brain.think("Bonjour")
print(f"✅ Response: {response[:50]}...")

# Stats
stats = brain.textual_cortex.get_neural_statistics()
print(f"✅ Activation: {stats['current_activation']:.3f}")
```

---

## Questions Fréquentes

**Q: Le cortex remplace-t-il le LLM?**
A: Non, c'est complémentaire. Le cortex traite le texte, le LLM génère la réponse.

**Q: Est-ce que le code ancien continue à marcher?**
A: Oui! Rétro-compatibilité assurée.

**Q: Comment sauvegarder l'état?**
A: `state = brain.textual_cortex.get_persistent_state()`

**Q: Comment restaurer l'état?**
A: `brain.textual_cortex.load_persistent_state(state)`

**Q: Ça utilise GPU?**
A: Oui, automatiquement si disponible.

**Q: Combien de mémoire ça prend?**
A: ~120 MB (modèle + états)

---

## Documentation

Pour plus de détails:

- 📖 **`documentation/CORTEX_TEXTUEL_RNN.md`** - Documentation technique (400+ lignes)
- 📖 **`CORTEX_TEXTUEL_INTEGRATION_GUIDE.md`** - Guide d'intégration (400+ lignes)
- 📖 **`RNN_MODERNIZATION_README.md`** - Résumé rapide (150 lignes)
- 📊 **`documentation/RNN_MODERNIZATION_SUMMARY.md`** - Résumé avec visuals (350 lignes)

---

## Troubleshooting

**Erreur: "module 'nety.cortex_limbic' has no attribute 'TextualCortex'"**
→ Vérifier que `textual_cortex.py` existe
→ Vérifier que `__init__.py` l'exporte

**Erreur: "CUDA out of memory"**
→ Réduire `batch_size` ou `hidden_size`

**État ne se sauvegarde pas**
→ Vérifier `use_persistent_state=True`

**Performance lente**
→ Utiliser GPU (`device='cuda'`)
→ Réduire `num_layers` ou `seq_length`

---

## Architecture en 30 secondes

```
Input (Message)
    ↓
Embedding (768 dim)
    ↓
Attention (4 têtes) ← Focus sur parties clés
    ↓
LSTM Bi-directionnel (3 couches) ← Comprend des 2 côtés
    ↓
État Persistant ← Se souvient
    ↓
Modulation Émotionnelle ← Adapté émotionnellement
    ↓
Output (512 dim)
```

---

## Performance

| Métrique | Valeur |
|----------|--------|
| Paramètres | 3.5M |
| Forward pass | 10-30ms |
| Mémoire | 120 MB |
| GPU Support | ✅ Oui |

---

## Status

✅ **OPÉRATIONNEL**  
✅ **TESTÉ**  
✅ **DOCUMENTÉ**  
✅ **PRODUCTION-READY**  

---

## Commandes Rapides

```bash
# Tests
python tests/test_textual_cortex.py

# Démo automatisée
python scripts/demo_rnn_cortex.py --mode demo

# Démo interactive
python scripts/demo_rnn_cortex.py --mode interactive

# Utilisation simple
python -c "from nety.core.brain import Brain; b = Brain(); print(b.think('Bonjour'))"
```

---

**C'est tout!** 🎉

Pour plus de détails, voir les fichiers de documentation.

*Cortex Textuel RNN v2.0 - Production Ready*
