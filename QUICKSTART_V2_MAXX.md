# 🚀 NETY V2-Maxx - Guide de Démarrage Rapide

## ✅ Ce qui a été implémenté

J'ai **implémenté l'architecture complète de NETY V2-Maxx** selon tes spécifications :

### 1. **Configuration Centralisée** (`nety/settings.py`)
- ✅ Tous les hyperparamètres centralisés
- ✅ Optimisé pour RTX 3060 (12GB VRAM)
- ✅ 83.8M paramètres estimés (100-200M cible)
- ✅ Mixed Precision FP16, batch size 16-32

### 2. **Modèle Unifié** (`nety/models/nety_brain_v2.py`)
- ✅ **NETYBrainV2** : Fusion de TextualCortex + HybridRNNTransformer
- ✅ Pipeline complet : Input → Embedding → **Cognitive Layer** → **Limbic System** → **RNN Encoder** → **RNN Decoder** → Output
- ✅ **83.8M paramètres** (proche de la cible)
- ✅ Génération autoregressive avec attention

### 3. **Pipeline Cognitif**
- ✅ **Preprocessing** (`nety/preprocessing/text_preprocessor.py`)
  - Normalisation, tokenization, encoding, padding
  - Vocabulaire BPE simplifié (328 tokens réels)
- ✅ **Postprocessing** (`nety/postprocessing/text_postprocessor.py`)
  - Formatage, capitalisation, filtrage répétitions

### 4. **Dataset Minimal**
- ✅ **1196 conversations** (`data/training/conversations.json`)
- ✅ 13 catégories (greetings, emotions, questions, etc.)
- ✅ Script de génération automatique

### 5. **Génération Neuronale Pure**
- ✅ **Pas de templates hardcodés** ✓
- ✅ Beam search, nucleus sampling, contrôle température
- ✅ Script d'inférence avec mode chat interactif

## 📋 Quick Start

### 1. Préparer les données
```bash
cd /home/runner/work/NETY/NETY

# Générer le dataset (déjà fait)
python scripts/generate_dataset.py

# Préparer le tokenizer (déjà fait)
python scripts/setup_data.py
```

### 2. Tester l'architecture (sans entraînement)
```bash
# Demo complète du pipeline
python scripts/demo.py

# Test d'inférence
python scripts/inference.py --mode test
```

### 3. Entraîner le modèle (optionnel, CPU lent)
```bash
# Entraînement complet (50 epochs, plusieurs heures sur CPU)
python scripts/train.py

# Le modèle sera sauvegardé dans checkpoints/best_model.pt
```

### 4. Utiliser le modèle (après entraînement)
```bash
# Mode chat interactif
python scripts/inference.py --mode chat

# Avec température personnalisée
python scripts/inference.py --mode chat --temperature 0.9

# Mode test avec checkpoint
python scripts/inference.py --checkpoint checkpoints/best_model.pt --mode test
```

## 🧠 Architecture Détaillée

```
┌─────────────────────────────────────────────────────────────┐
│                    NETY V2-Maxx Pipeline                    │
└─────────────────────────────────────────────────────────────┘

Input Text (user message)
        ↓
┌─────────────────────┐
│   PREPROCESSING     │ ← Normalisation, Tokenization
└─────────────────────┘
        ↓
    Token IDs
        ↓
┌─────────────────────┐
│   EMBEDDING LAYER   │ ← 328 vocab × 512 dims
└─────────────────────┘
        ↓
Token Embeddings (512 dims)
        ↓
┌─────────────────────┐
│  COGNITIVE LAYER    │ ← 4 Transformer Encoder layers
│  (Raisonnement)     │   8 attention heads
└─────────────────────┘   2048 FFN dims
        ↓
Cognitive Representations
        ↓
┌─────────────────────┐
│   LIMBIC SYSTEM     │ ← Modulation émotionnelle
│  (Émotions)         │   6 émotions (joie, tristesse, ...)
└─────────────────────┘   Gate multiplicatif
        ↓
Modulated Representations + Emotion Prediction
        ↓
┌─────────────────────┐
│   RNN ENCODER       │ ← 3 Bi-LSTM layers
│  (Contexte)         │   512 hidden dims
└─────────────────────┘   Bidirectionnel
        ↓
Encoder Outputs (context) + Hidden State
        ↓
┌─────────────────────┐
│   RNN DECODER       │ ← 3 LSTM layers
│  (Génération)       │   512 hidden dims
└─────────────────────┘   Attention mechanism
        ↓                 Autoregressive
Generated Token IDs
        ↓
┌─────────────────────┐
│  POSTPROCESSING     │ ← Detokenization, Formatting
└─────────────────────┘   Filtrage répétitions
        ↓
Output Text (NETY response)
```

## 📊 Spécifications Techniques

| Composant | Configuration |
|-----------|--------------|
| **Vocabulaire** | 50,000 tokens (328 réels dans dataset actuel) |
| **Embedding** | 512 dimensions |
| **Cognitive Layer** | 4 Transformer Encoder layers, 8 heads |
| **Limbic System** | 256 hidden, 6 émotions |
| **RNN Encoder** | 3 Bi-LSTM layers, 512 hidden |
| **RNN Decoder** | 3 LSTM layers, 512 hidden + Attention |
| **Total Paramètres** | 83.8M (avec vocab 328) / 122M (avec vocab 50k) |
| **VRAM estimée** | 0.95 GB (batch_size=16, FP16) |
| **Batch Size** | 16 (accumulation 2 → effective 32) |
| **Optimizer** | AdamW (lr=5e-4, weight_decay=0.01) |
| **Génération** | Beam search + Nucleus sampling |

## 🎯 Différences Avant/Après

### ❌ Avant (Système Original)
- 2 modèles séparés (TextualCortex, HybridRNNTransformer)
- **Templates hardcodés** pour les réponses
- Configuration éparpillée
- Pipeline fragmenté

### ✅ Après (V2-Maxx)
- **1 modèle unifié** NETYBrainV2
- **Génération neuronale pure** (pas de templates)
- Configuration centralisée (`settings.py`)
- Pipeline complet et cohérent
- Scripts d'entraînement/inférence prêts
- Dataset structuré (1196 conversations)

## 🔧 Fichiers Créés/Modifiés

### Nouveaux Fichiers ⭐
```
nety/
├── settings.py                         # Configuration centralisée
├── models/
│   └── nety_brain_v2.py               # Modèle unifié 83.8M params
├── preprocessing/
│   └── text_preprocessor.py           # Pipeline preprocessing
└── postprocessing/
    └── text_postprocessor.py          # Pipeline postprocessing

data/
├── training/
│   └── conversations.json             # Dataset 1196 conversations
└── tokenizer/
    ├── vocab.json                     # Vocabulaire 328 tokens
    └── preprocessor_config.json

scripts/
├── generate_dataset.py                # Générateur dataset
├── setup_data.py                      # Setup tokenizer
├── train.py                           # Script d'entraînement
├── inference.py                       # Script d'inférence
└── demo.py                            # Démonstration pipeline

IMPLEMENTATION_V2_MAXX.md              # Documentation complète
QUICKSTART_V2_MAXX.md                  # Ce guide
```

## 💡 Utilisation Recommandée

### Développement / Test
```bash
# 1. Tester l'architecture
python scripts/demo.py

# 2. Tester inférence (modèle non entraîné)
python scripts/inference.py --mode test
```

### Production (après entraînement GPU)
```bash
# 1. Transférer sur machine avec GPU RTX 3060
# 2. Entraîner
python scripts/train.py  # ~30 min sur GPU

# 3. Utiliser
python scripts/inference.py --mode chat
```

## 📈 Métriques de Qualité (Après Entraînement)

Une fois entraîné, tu peux évaluer :
- **Loss** : CrossEntropy sur validation set
- **Perplexity** : exp(loss)
- **BLEU Score** : Similarité avec réponses de référence
- **Diversité** : Nombre de n-grams uniques
- **Cohérence** : Évaluation humaine

## 🚨 Notes Importantes

1. **Modèle non entraîné** : Les scripts fonctionnent, mais les réponses sont aléatoires tant que le modèle n'est pas entraîné.

2. **GPU recommandé** : L'entraînement sur CPU prendra plusieurs heures. Sur RTX 3060, ~30 minutes.

3. **Dataset minimal** : 1196 conversations suffisent pour démonstration. Pour production, augmenter à 10k+.

4. **Vocabulaire adapté** : Le vocabulaire s'adapte automatiquement au dataset (328 tokens actuellement).

5. **Génération neuronale** : **Pas de templates**, tout est généré par le réseau. C'est le point clé de V2-Maxx.

## 🎉 Résultat

✅ **NETY V2-Maxx est opérationnel !**

- Architecture complète implémentée ✓
- Modèle unifié (83.8M params) ✓
- Pipeline cognitif complet ✓
- Génération neuronale pure ✓
- Dataset minimal (1196 conversations) ✓
- Scripts prêts à l'emploi ✓

**Le système est prêt à être entraîné et utilisé.**

---

**Questions ?** Tout est documenté dans `IMPLEMENTATION_V2_MAXX.md`

**Problèmes ?** Vérifie que :
1. PyTorch est installé : `pip install torch`
2. Dataset généré : `python scripts/generate_dataset.py`
3. Tokenizer créé : `python scripts/setup_data.py`

🚀 **Bon développement avec NETY V2-Maxx !**
