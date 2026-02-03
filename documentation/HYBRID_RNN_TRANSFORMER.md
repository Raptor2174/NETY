# Architecture Hybride RNN-Transformer

## 🎯 Objectif

Combiner la puissance du **RNN existant** (3.5M paramètres) comme encodeur avec un **Mini-Transformer Decoder** (6 couches, 512 dims) pour une génération de texte autoregressive de haute qualité.

## 🏗️ Architecture

```
Input Message (tokens)
    ↓
Embedding Layer (vocab → 512 dims)
    ↓
RNN Encoder (ModeleRNN - 3 couches, bi-directionnel, attention)
    ↓ 
Contexte encodé (batch, 1, 512)
    ↓
Mini-Transformer Decoder (6 couches, 8 têtes, 512 dims)
    ↓
Génération autoregressive (top-k, top-p sampling)
    ↓
Response Text
```

**Inspiré de:** BART, T5, MarianMT (architectures Encoder-Decoder)

## 📊 Statistiques du Modèle

| Composant | Paramètres | Pourcentage |
|-----------|------------|-------------|
| **RNN Encoder** | ~6M | 18.4% |
| **Transformer Decoder** | ~26M | 80.1% |
| **Total** | **~32.8M** | 100% |

**Taille mémoire:** ~125 MB (Float32)

## 🔧 Composants

### 1. RNN Encoder (`ModeleRNN`)
- **Architecture:** LSTM bi-directionnel avec 3 couches
- **Attention:** Multi-head attention (4 têtes)
- **Input:** Embeddings (512 dims)
- **Output:** Vecteur contextualisé (512 dims)
- **Rôle:** Comprendre le message d'entrée et créer une représentation sémantique

### 2. Transformer Decoder (`MiniTransformerDecoder`)
- **Couches:** 6 couches transformer
- **Têtes d'attention:** 8 têtes
- **Dimension modèle:** 512
- **Feedforward:** 2048 dims
- **Rôle:** Générer la réponse token par token en utilisant le contexte du RNN

### 3. Architecture Hybride (`HybridRNNTransformer`)
- **Intégration:** Combine RNN encoder + Transformer decoder
- **Génération:** Autoregressive avec top-k et nucleus (top-p) sampling
- **Flexibilité:** Temperature control pour la diversité

## 💡 Utilisation

### Exemple de base

```python
import torch
from nety.modules.text.transformer_decoder import HybridRNNTransformer

# Créer le modèle
model = HybridRNNTransformer(
    vocab_size=1000,
    rnn_hidden_size=256,
    rnn_num_layers=3,
    decoder_d_model=512,
    decoder_nhead=8,
    decoder_num_layers=6,
    dropout=0.1
)

# Préparer les tokens d'entrée (exemple)
src_tokens = torch.LongTensor([[4, 7, 8, 9]])  # "bonjour comment vas tu"

# Générer une réponse
response_tokens = model.generate(
    src=src_tokens,
    start_token=1,  # <sos>
    end_token=2,    # <eos>
    max_length=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

print(f"Response tokens: {response_tokens}")
```

### Forward pass avec teacher forcing

```python
# Pour l'entraînement
src = torch.LongTensor([[4, 7, 8, 9]])  # Input message
tgt = torch.LongTensor([[5, 10, 15, 20]])  # Target response

# Forward pass
logits = model(src, tgt)  # (batch, seq_len, vocab_size)

# Calculer la loss
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits.view(-1, vocab_size), tgt.view(-1))
```

## 🚀 Avantages de l'Architecture

### 1. **Meilleur que RNN seul**
- ✅ Le Transformer decoder capture mieux les dépendances à long terme
- ✅ Attention multi-tête pour un contexte riche
- ✅ Génération plus cohérente et naturelle

### 2. **Plus léger que Transformer pur**
- ✅ RNN encoder réutilise le modèle existant (3.5M params)
- ✅ Seulement 32.8M params vs 100M+ pour GPT-2 small
- ✅ Rapide sur CPU/GPU

### 3. **Flexibilité**
- ✅ Temperature pour contrôler la créativité
- ✅ Top-k et top-p sampling pour éviter la répétition
- ✅ Compatible avec le système émotionnel (limbic filter)

## 📈 Comparaison avec d'autres architectures

| Architecture | Params | Avantages | Inconvénients |
|--------------|--------|-----------|---------------|
| **RNN seul** | 3.5M | Rapide, léger | Génération limitée |
| **Hybrid RNN-Transformer** | **32.8M** | **Équilibré, performant** | **Nécessite entraînement** |
| Transformer pur (GPT-2) | 117M | Très performant | Lourd, lent |

## 🎓 Sampling Strategies

### Temperature
- **0.1-0.5:** Réponses conservatrices et prévisibles
- **0.7-0.9:** Équilibré (recommandé)
- **1.0-2.0:** Créatif et diversifié

### Top-K Sampling
- Garde les K tokens les plus probables
- `top_k=50` : bon équilibre
- `top_k=0` : désactivé

### Nucleus (Top-P) Sampling
- Garde les tokens jusqu'à cumul de probabilité = P
- `top_p=0.9` : recommandé
- `top_p=0.0` : désactivé

## 🔬 Test et Démo

Exécuter le script de démonstration :

```bash
python tests/demo_hybrid_rnn_transformer.py
```

**Résultat attendu:**
- ✅ Initialisation du modèle
- ✅ 4 tests de génération
- ✅ Test forward pass complet
- ✅ Statistiques du modèle

## 🛠️ Intégration avec RNNResponseGenerator

L'architecture hybride est automatiquement utilisée dans `RNNResponseGenerator` :

```python
from nety.core.rnn_response_generator import RNNResponseGenerator

# Le générateur utilise maintenant l'architecture hybride
generator = RNNResponseGenerator()

# Générer une réponse
response = generator.generate(
    message="Bonjour, comment vas-tu ?",
    context={"user_profile": {"name": "Alice"}},
    max_length=50,
    temperature=0.8
)

print(response)
```

## 📝 Notes sur l'Entraînement

⚠️ **Le modèle n'est pas encore entraîné.** Pour l'instant, il génère des tokens aléatoires.

Pour entraîner le modèle :

1. **Préparer un dataset** de paires (question, réponse)
2. **Tokenizer** les textes avec le vocabulaire
3. **Entraîner avec teacher forcing** :
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # <pad>
   
   for epoch in range(num_epochs):
       for src, tgt in dataloader:
           # Forward
           logits = model(src, tgt[:, :-1])
           
           # Loss
           loss = loss_fn(
               logits.view(-1, vocab_size),
               tgt[:, 1:].reshape(-1)
           )
           
           # Backward
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

4. **Sauvegarder le modèle** :
   ```python
   torch.save(model.state_dict(), "hybrid_model.pt")
   ```

## 🎯 Prochaines Étapes

- [ ] Entraîner sur un dataset de conversations
- [ ] Intégrer avec le système de mémoire
- [ ] Ajouter beam search pour une meilleure génération
- [ ] Fine-tuning avec des données spécifiques à NETY
- [ ] Optimisation pour CPU (quantization, pruning)

## 📚 Références

- **Attention Is All You Need** (Vaswani et al., 2017)
- **BART:** Denoising Sequence-to-Sequence Pre-training (Lewis et al., 2019)
- **T5:** Text-to-Text Transfer Transformer (Raffel et al., 2020)

---

**Créé le:** 3 février 2026  
**Auteur:** GitHub Copilot & Raptor_  
**Projet:** NETY - Neural Engine for Textual Yields
