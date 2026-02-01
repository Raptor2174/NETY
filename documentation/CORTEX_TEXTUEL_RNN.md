# 🧠 Cortex Textuel RNN - Documentation Technique

## Vue d'ensemble

Le **Cortex Textuel Autonome** est le nouveau réseau neuronal du cortex cérébral (textuel) de NETY. C'est une version modernisée et améliorée du vieux RNN original, transformée en cœur neuronal autonome du système NETY.

### Caractéristiques Principales

✨ **Architecture Avancée:**
- **LSTM Bi-directionnel** (3 couches) : Traite le texte dans les deux sens pour une meilleure compréhension contextuelle
- **Attention Multi-tête** (4 têtes) : Focalise l'attention sur les parties pertinentes du texte
- **Batch Normalization** : Stabilise l'entraînement et l'inférence
- **Dropout** : Réduit l'overfitting (30%)
- **Activation GELU** : Activation moderne et progressive

🧠 **État Persistant Autonome:**
- **Mémoire Court-terme** : L'état caché (h, c) persiste entre les interactions
- **Historique d'État** : Les 50 derniers états cachés sont conservés
- **Contexte Glissant** : Fenêtre des 20 dernières interactions maintenue en mémoire

💭 **Intégration Limbique:**
- **Modulation Émotionnelle** : Les émotions du système limbique influencent l'activation neuronal
- **Adaptation Contextuelle** : L'activation neuronal s'adapte au contexte émotionnel
- **Apprentissage Continu** : Le réseau s'adapte à travers les interactions

---

## Architecture Détaillée

### 1. ModeleRNN (modele_rnn.py)

#### Classes

**MultiHeadAttention**
```python
class MultiHeadAttention(nn.Module):
    """Mécanisme d'attention multi-tête"""
    
    __init__(hidden_size, num_heads=4)
    forward(query, key, value) -> (output, attention_weights)
```

**ModeleRNN**
```python
class ModeleRNN(nn.Module):
    """LSTM bi-directionnel avec attention et état persistant"""
    
    __init__(input_size, hidden_size=256, output_size=512, 
             num_layers=3, num_heads=4, dropout=0.3, 
             bidirectional=True, use_attention=True, device=None)
    
    forward(x, use_persistent_state=False) -> Tensor
    reset_persistent_state() -> None
    get_state_history() -> List[Tuple]
    set_persistent_state(h, c) -> None
```

#### Flux de Traitement

```
Input (batch, seq_len, input_size)
    ↓
[Embedding optionnel si input_size < 512]
    ↓
Multi-Head Attention
    ↓
Bi-directional LSTM (3 couches)
    ↓
Batch Norm 1
    ↓
Dense Layer 1 (GELU + Dropout)
    ↓
Batch Norm 2
    ↓
Dense Layer 2 (GELU + Dropout)
    ↓
Batch Norm 3
    ↓
Output Dense Layer
    ↓
Layer Norm
    ↓
Output (batch, output_size)
```

### 2. TextualCortex (cortex_limbic/textual_cortex.py)

#### Classe Principale

```python
class TextualCortex:
    """Cortex Textuel Autonome - Cerveau Neuronal Textuel de NETY"""
    
    __init__(hidden_size=256, output_size=512, num_layers=3, 
             num_heads=4, dropout=0.3, device=None, 
             emotion_engine=None, memory_manager=None)
```

#### Méthodes Clés

**process_text_sequence()**
- Traite une séquence textuelle via le RNN
- Applique la modulation émotionnelle du limbic system
- Maintient l'état persistant
- Retourne: (neural_output, metadata)

**_apply_emotional_modulation()**
- Calcule le facteur émotionnel (positif - négatif)
- Modifie l'activation neuronal : output * (1 + factor * 0.3)
- Émotions positives augmentent l'activation
- Émotions négatives la réduisent

**_calculate_neural_activation()**
- Calcule le niveau d'activation (0-1)
- Activation = norm_moyenne(output)
- Met à jour les statistiques

**État Neural Persistant**
```python
neural_state = {
    "timestamp": datetime,
    "activation_level": float,  # [0, 1]
    "attention_focus": Optional,
    "emotional_modulation": Dict
}
```

---

## Intégration au Cerveau (Brain)

### Modifications du Brain.py

1. **Import du TextualCortex**
```python
from nety.cortex_limbic.textual_cortex import TextualCortex
```

2. **Initialisation dans __init__**
```python
self.textual_cortex = TextualCortex(
    hidden_size=256,
    output_size=512,
    num_layers=3,
    num_heads=4,
    dropout=0.3,
    emotion_engine=self.emotion_engine,
    memory_manager=self.memory
)
```

3. **Pipeline de Traitement**
```
Message d'entrée
    ↓
Analyse d'intention
    ↓
Récupération contextuelle
    ↓
Filtrage limbique
    ↓
[NOUVEAU] Traitement RNN Cortex Textuel ✨
    ├─ Conversion en embedding
    ├─ Traitement RNN avec attention
    ├─ Modulation émotionnelle
    └─ Mise à jour état persistant
    ↓
Génération de réponse
    ↓
Ingestion ML
    ↓
Sortie utilisateur
```

---

## Utilisation

### Installation

```bash
# Assurez-vous que PyTorch est installé
pip install torch

# Les dépendances sont déjà dans requirements.txt
```

### Utilisation Simple

```python
from nety.cortex_limbic.textual_cortex import TextualCortex
import torch

# Créer le cortex
cortex = TextualCortex(hidden_size=256, output_size=512)

# Créer un embedding (768 dimensions)
embedding = torch.randn(1, 5, 768)  # (batch, seq_len, features)

# Traiter
output, metadata = cortex.process_text_sequence(embedding)

# Afficher les résultats
print(f"Activation: {metadata['activation_level']:.3f}")
print(f"Output shape: {output.shape}")
```

### Utilisation avec Émotions

```python
from nety.cortex_limbic.emotion_engine import EmotionEngine

emotion_engine = EmotionEngine()
cortex = TextualCortex(emotion_engine=emotion_engine)

# Traiter avec modulation émotionnelle
emotional_context = {
    "emotions": emotion_engine.emotions
}

output, metadata = cortex.process_text_sequence(
    embedding,
    emotional_context=emotional_context,
    use_persistent_state=True
)
```

### Utilisation du Cerveau Complet

```python
from nety.core.brain import Brain

# Initialiser le cerveau (le cortex est créé automatiquement)
brain = Brain()

# Traiter un message (le cortex textuel est utilisé automatiquement)
response = brain.think("Bonjour, comment ça va?")

# Accéder aux statistiques neurales
stats = brain.textual_cortex.get_neural_statistics()
print(f"Activation: {stats['current_activation']:.3f}")
```

---

## Statistiques et Monitoring

### Statistiques Disponibles

```python
stats = cortex.get_neural_statistics()

# Contient:
# - total_activations: Nombre total d'activations
# - average_activation: Moyenne des 100 dernières activations
# - peak_activation: Pic d'activation observé
# - current_activation: Activation actuelle
# - context_depth: Profondeur de la mémoire d'état
# - last_update: Timestamp de la dernière mise à jour
```

### Contexte Summary

```python
context = cortex.get_context_summary()

# Contient:
# - window_size: Nombre d'interactions en fenêtre
# - neural_state: État neuronal actuel
# - activation_stats: Statistiques complètes
# - recent_interactions: 5 dernières interactions
```

---

## Performance et Optimisation

### Paramètres d'Optimisation

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| hidden_size | 256 | Taille de l'état caché LSTM |
| output_size | 512 | Taille de la représentation de sortie |
| num_layers | 3 | Nombre de couches LSTM |
| num_heads | 4 | Nombre de têtes d'attention |
| dropout | 0.3 | Taux de dropout |
| bidirectional | True | LSTM bi-directionnel |
| use_attention | True | Attention multi-tête |

### Mémoire et Complexité

**Nombre de Paramètres:** ~3.5M paramètres

**Utilisation Mémoire:**
- Modèle: ~15 MB
- État persistant (h, c): ~2 MB
- Historique d'état (50): ~100 MB
- Contexte glissant: Minimal (~1 MB)

**Complexité Computationnelle:**
- Forward pass: O(seq_len × hidden_size²)
- Avec attention: +O(seq_len² × hidden_size)

### Optimisations Implémentées

1. **Batch First LSTM** : Meilleure localité mémoire
2. **Gradient Checkpointing** : Possible avec torch.utils.checkpoint
3. **Mixed Precision** : Compatible avec torch.amp
4. **State Pruning** : Historique limité à 50 états

---

## Tests et Démonstration

### Tests Disponibles

**test_textual_cortex.py**
```bash
python tests/test_textual_cortex.py
```

Tests effectués:
1. Modèle RNN brut
2. Cortex Textuel autonome
3. Modulation émotionnelle
4. Statistiques neurales
5. Persistance d'état

### Démonstration Interactive

**Mode Automatisé:**
```bash
python scripts/demo_rnn_cortex.py --mode demo
```

**Mode Interactif:**
```bash
python scripts/demo_rnn_cortex.py --mode interactive
```

---

## Améliorations Futures

### Court Terme
- [ ] Sauvegarde/restauration d'état complet
- [ ] Quantization pour déploiement
- [ ] Export ONNX

### Moyen Terme
- [ ] Intégration avec transformers
- [ ] Fine-tuning supervisé
- [ ] Métriques d'attention

### Long Terme
- [ ] Apprentissage par renforcement
- [ ] Fusion multi-modale
- [ ] Compétition de modèles

---

## Troubleshooting

### Erreur: "CUDA out of memory"
→ Réduire `batch_size` ou `hidden_size`

### Activation neuronal = 0
→ Vérifier l'embedding d'entrée (ne doit pas être tous zéros)

### État persistant ne s'accumule pas
→ Vérifier `use_persistent_state=True`

### Performance lente
→ Réduire `num_layers` ou `seq_length`

---

## Références

- LSTM: [Hochreiter & Schmidhuber, 1997]
- Attention: [Vaswani et al., 2017]
- Bi-directional RNN: [Schuster & Paliwal, 1997]
- Batch Normalization: [Ioffe & Szegedy, 2015]

---

## Auteur & Historique

- **Création Initiale:** RNN simple (LSTM 2 couches)
- **Modernisation 2026:** Bi-directional + Attention + Modulation émotionnelle
- **Intégration NETY:** Cortex Textuel Autonome du Cerveau

**État:** ✅ Opérationnel et Intégré

---

*Documentation générée pour NETY - Cortex Textuel RNN v2.0*
