# Guide d'Installation et d'Utilisation - Base de Connaissances NETY

## 📦 Installation

### 1. Installer les Dépendances de Base

```bash
# Installer les dépendances requises
pip install -r requirements.txt
```

### 2. Installer les Dépendances Optionnelles

#### Pour la Recherche Sémantique (Recommandé)

```bash
# Chroma DB et Sentence Transformers
pip install chromadb>=0.4.0 sentence-transformers>=2.2.0
```

#### Pour le Cache Haute Performance (Optionnel)

```bash
# Redis
pip install redis>=5.0.0 hiredis>=2.0.0

# Installer et démarrer le serveur Redis
# Sur Ubuntu/Debian:
sudo apt-get install redis-server
sudo systemctl start redis

# Sur macOS:
brew install redis
brew services start redis

# Activer Redis dans les variables d'environnement
export REDIS_ENABLED=true
```

## 🚀 Démarrage Rapide

### Initialiser la Base de Données

```python
from nety.knowledge_base import init_databases

# Initialise SQLite, Chroma DB et Redis (si disponibles)
init_databases()
```

Cela crée automatiquement :
- `/data/databases/nety_knowledge.db` - Base SQLite
- `/data/databases/chroma/` - Données Chroma DB

### Exemple Basique

```python
from nety.knowledge_base import KnowledgeManager, SearchEngine

# Créer le gestionnaire
km = KnowledgeManager()

# Ajouter une connaissance
knowledge_id = km.add_knowledge(
    title="Les Réseaux de Neurones Récurrents",
    content="Les RNN sont des réseaux de neurones avec des connexions récurrentes...",
    category="deep_learning",
    tags=["rnn", "neural_networks"],
    metadata={"difficulty": "intermediate"}
)

# Rechercher
search = SearchEngine()
results = search.search("réseaux neurones")

# Obtenir du contexte pour RAG
context = search.get_context_for_query("Qu'est-ce qu'un RNN?")
```

## 📚 Utilisation Avancée

### 1. Gestion des Connaissances

```python
# Récupérer une connaissance
knowledge = km.get_knowledge(knowledge_id)

# Mettre à jour
km.update_knowledge(
    knowledge_id,
    content="Nouveau contenu...",
    tags=["rnn", "lstm", "gru"]
)

# Supprimer
km.delete_knowledge(knowledge_id)

# Récupérer toutes les connaissances
all_knowledge = km.get_all_knowledge(limit=50)
```

### 2. Recherche Sémantique

```python
# Recherche sémantique (nécessite Chroma DB)
results = search.search(
    query="apprentissage profond",
    use_semantic=True,
    limit=5
)

# Trouver des connaissances similaires
similar = search.find_similar(
    text="Je veux en savoir plus sur les CNN",
    limit=3
)

# Recherche par catégorie
results = search.search(
    category="deep_learning",
    limit=10
)
```

### 3. Gestion des Conversations

```python
# Sauvegarder une conversation
conversation_id = km.save_conversation(
    user_input="Explique-moi les RNN",
    nety_response="Les RNN sont...",
    session_id="user_123",
    context="[Contexte récupéré de la KB]",
    metadata={"language": "fr", "topic": "ml"}
)

# Récupérer l'historique
history = km.get_conversation_history(
    session_id="user_123",
    limit=20
)

# Accéder aux conversations
for conv in history:
    print(f"User: {conv['user_input']}")
    print(f"NETY: {conv['nety_response']}")
    print(f"Timestamp: {conv['timestamp']}")
```

### 4. RAG (Retrieval-Augmented Generation)

```python
def answer_with_context(user_question):
    """Répondre avec contexte de la base de connaissances"""
    
    # 1. Récupérer le contexte pertinent
    search = SearchEngine()
    context = search.get_context_for_query(user_question, max_results=3)
    
    # 2. Combiner avec la question
    prompt = f"""
    Contexte de la base de connaissances:
    {context}
    
    Question: {user_question}
    
    Réponse basée sur le contexte:
    """
    
    # 3. Envoyer au Brain de NETY
    # response = brain.generate(prompt)
    
    # 4. Sauvegarder la conversation
    # km.save_conversation(user_question, response, context=context)
    
    return prompt

# Utilisation
question = "Comment fonctionnent les réseaux de neurones récurrents?"
answer_with_context(question)
```

### 5. Statistiques et Monitoring

```python
# Obtenir les statistiques
stats = km.get_stats()

print(f"Connaissances: {stats['knowledge_count']}")
print(f"Conversations: {stats['conversations_count']}")
print(f"Catégories: {stats['categories']}")
print(f"Chroma DB: {stats['chroma_available']}")
print(f"Redis: {stats['redis_available']}")
```

## 🔧 Configuration

### Variables d'Environnement

```bash
# Redis (optionnel)
export REDIS_ENABLED=true
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=your_password  # si nécessaire
```

### Configuration des Chemins

Par défaut, les bases de données sont stockées dans :
- `data/databases/nety_knowledge.db`
- `data/databases/chroma/`

Pour personnaliser, modifier `nety/knowledge_base/database_config.py`.

## 🧪 Tests

```bash
# Exécuter les tests
pytest tests/test_knowledge_base.py -v

# Exécuter l'exemple de démonstration
python examples/knowledge_base_demo.py
```

## 📊 Performance

### Limites Recommandées

- **SQLite** : 
  - Jusqu'à 100,000 connaissances
  - >10,000 requêtes/seconde en lecture
  
- **Chroma DB** :
  - Millions d'embeddings
  - <100ms pour recherche top-10
  
- **Redis** :
  - >100,000 opérations/seconde

### Optimisation

1. **Index SQLite** : Déjà créés sur les colonnes fréquemment recherchées
2. **Cache Redis** : Active automatiquement si Redis est disponible
3. **Batch Operations** : Utiliser des transactions pour insertions multiples

```python
from nety.knowledge_base import DatabaseConnector

# Insertion en batch
knowledges = [
    ("Titre 1", "Contenu 1", "cat1"),
    ("Titre 2", "Contenu 2", "cat2"),
    # ...
]

with DatabaseConnector.sqlite_cursor() as cursor:
    cursor.executemany("""
        INSERT INTO knowledge (title, content, category)
        VALUES (?, ?, ?)
    """, knowledges)
```

## 🔒 Sécurité

### Bonnes Pratiques

1. **Pas de données sensibles** : Ne stockez pas d'informations personnelles sensibles
2. **Permissions fichiers** : Restreindre l'accès aux fichiers de base de données
3. **Backups réguliers** : Sauvegarder régulièrement les bases de données

### Backup

```bash
# Backup SQLite
cp data/databases/nety_knowledge.db data/backups/nety_knowledge_$(date +%Y%m%d).db

# Backup Chroma DB
tar -czf data/backups/chroma_$(date +%Y%m%d).tar.gz data/databases/chroma/
```

## 🆘 Dépannage

### Problème : "Module 'chromadb' not found"

```bash
pip install chromadb sentence-transformers
```

La base de connaissances fonctionnera en mode dégradé (sans recherche sémantique).

### Problème : "Redis connection refused"

Vérifier que Redis est démarré :
```bash
sudo systemctl status redis
# ou
redis-cli ping  # doit retourner "PONG"
```

Si non nécessaire, désactiver Redis :
```bash
export REDIS_ENABLED=false
```

### Problème : Database locked

SQLite peut bloquer si accédé par plusieurs processus. NETY utilise un modèle single-thread, mais si vous utilisez la KB dans un contexte multi-thread :

```python
# Utiliser des connexions séparées par thread
# Ou désactiver check_same_thread (déjà fait dans le code)
```

## 📖 Ressources

- [Documentation complète](../documentation/DATABASE_SOLUTIONS.md)
- [Exemple d'utilisation](../examples/knowledge_base_demo.py)
- [Tests](../tests/test_knowledge_base.py)

## 🎯 Prochaines Étapes

1. Intégrer avec le Brain de NETY pour RAG
2. Ajouter des connaissances spécifiques au domaine
3. Implémenter l'apprentissage continu
4. Optimiser les embeddings pour le français
5. Ajouter une interface d'administration

---

**Besoin d'aide?** Consultez les [issues GitHub](https://github.com/Raptor2174/NETY/issues) ou créez une nouvelle issue.
