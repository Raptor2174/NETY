#!/usr/bin/env python3
"""
Script d'exemple d'utilisation de la base de connaissances NETY

Ce script démontre comment :
1. Initialiser les bases de données
2. Ajouter des connaissances
3. Rechercher des informations
4. Sauvegarder des conversations
5. Obtenir des statistiques
"""

from nety.knowledge_base import (
    init_databases,
    KnowledgeManager,
    SearchEngine
)


def main():
    print("=" * 70)
    print(" 🧠 NETY Knowledge Base - Exemple d'utilisation")
    print("=" * 70)
    print()
    
    # ===============================
    # 1. INITIALISATION
    # ===============================
    print("📊 Étape 1 : Initialisation des bases de données")
    print("-" * 70)
    init_databases()
    print()
    
    # ===============================
    # 2. CRÉER LE GESTIONNAIRE
    # ===============================
    print("📊 Étape 2 : Création du gestionnaire de connaissances")
    print("-" * 70)
    km = KnowledgeManager()
    print("✅ KnowledgeManager créé")
    print()
    
    # ===============================
    # 3. AJOUTER DES CONNAISSANCES
    # ===============================
    print("📊 Étape 3 : Ajout de connaissances")
    print("-" * 70)
    
    # Connaissance sur les RNN
    rnn_id = km.add_knowledge(
        title="Réseaux de Neurones Récurrents (RNN)",
        content="""Les réseaux de neurones récurrents (RNN) sont un type d'architecture 
        de réseau de neurones conçu pour traiter des séquences de données. Contrairement 
        aux réseaux feedforward classiques, les RNN possèdent des connexions récurrentes 
        qui leur permettent de maintenir une forme de mémoire des entrées précédentes. 
        Cela les rend particulièrement adaptés pour le traitement du langage naturel, 
        la reconnaissance vocale et l'analyse de séries temporelles.""",
        category="deep_learning",
        source="Documentation NETY",
        tags=["rnn", "neural_networks", "nlp", "sequences"],
        metadata={"difficulty": "intermediate", "language": "fr"}
    )
    print(f"✅ Connaissance RNN ajoutée (ID: {rnn_id})")
    
    # Connaissance sur les CNN
    cnn_id = km.add_knowledge(
        title="Réseaux de Neurones Convolutionnels (CNN)",
        content="""Les réseaux de neurones convolutionnels (CNN) sont spécialisés dans 
        le traitement de données ayant une structure en grille, comme les images. 
        Ils utilisent des couches de convolution pour détecter automatiquement des 
        motifs et des caractéristiques visuelles. Les CNN sont largement utilisés 
        en vision par ordinateur pour des tâches comme la classification d'images, 
        la détection d'objets et la segmentation.""",
        category="deep_learning",
        source="Documentation NETY",
        tags=["cnn", "neural_networks", "computer_vision", "images"],
        metadata={"difficulty": "intermediate", "language": "fr"}
    )
    print(f"✅ Connaissance CNN ajoutée (ID: {cnn_id})")
    
    # Connaissance sur NETY
    nety_id = km.add_knowledge(
        title="NETY - IA Multimodale",
        content="""NETY est un projet d'intelligence artificielle multimodale capable 
        de traiter du texte (NLP), des images (CNN) et de l'audio (Speech-to-Text). 
        Le projet utilise PyTorch et TensorFlow comme frameworks principaux. 
        L'architecture est modulaire et extensible, permettant l'ajout facile 
        de nouvelles fonctionnalités.""",
        category="project_info",
        source="README NETY",
        tags=["nety", "ai", "multimodal", "nlp", "vision", "audio"],
        metadata={"project": "NETY", "language": "fr"}
    )
    print(f"✅ Connaissance NETY ajoutée (ID: {nety_id})")
    print()
    
    # ===============================
    # 4. RECHERCHE DE CONNAISSANCES
    # ===============================
    print("📊 Étape 4 : Recherche de connaissances")
    print("-" * 70)
    
    # Créer le moteur de recherche
    search = SearchEngine()
    
    # Recherche 1: Recherche textuelle simple
    print("\n🔍 Recherche 1 : 'réseaux de neurones'")
    results = search.search("réseaux de neurones", use_semantic=False)
    print(f"   Nombre de résultats : {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['title']} (Catégorie: {result['category']})")
    
    # Recherche 2: Par catégorie
    print("\n🔍 Recherche 2 : Catégorie 'deep_learning'")
    results = search.search(None, category="deep_learning")
    print(f"   Nombre de résultats : {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['title']}")
    
    # Recherche 3: Contexte pour RAG
    print("\n🔍 Recherche 3 : Contexte pour 'qu'est-ce qu'un CNN?'")
    context = search.get_context_for_query("qu'est-ce qu'un CNN?", max_results=2)
    print(f"   Contexte récupéré ({len(context)} caractères)")
    print(f"   Aperçu: {context[:200]}...")
    print()
    
    # ===============================
    # 5. MISE À JOUR D'UNE CONNAISSANCE
    # ===============================
    print("📊 Étape 5 : Mise à jour d'une connaissance")
    print("-" * 70)
    
    success = km.update_knowledge(
        nety_id,
        tags=["nety", "ai", "multimodal", "nlp", "vision", "audio", "python"],
        metadata={"project": "NETY", "language": "fr", "updated": True}
    )
    print(f"✅ Mise à jour {'réussie' if success else 'échouée'}")
    print()
    
    # ===============================
    # 6. CONVERSATIONS
    # ===============================
    print("📊 Étape 6 : Sauvegarde de conversations")
    print("-" * 70)
    
    # Conversation 1
    conv1_id = km.save_conversation(
        user_input="Qu'est-ce qu'un RNN?",
        nety_response="Un RNN (Réseau de Neurones Récurrent) est un type de réseau "
                     "de neurones conçu pour traiter des séquences de données. "
                     "Il possède une mémoire des entrées précédentes.",
        session_id="demo_session_001",
        metadata={"language": "fr", "topic": "deep_learning"}
    )
    print(f"✅ Conversation 1 sauvegardée (ID: {conv1_id})")
    
    # Conversation 2
    conv2_id = km.save_conversation(
        user_input="Et un CNN?",
        nety_response="Un CNN (Réseau de Neurones Convolutionnel) est spécialisé "
                     "dans le traitement d'images. Il utilise des couches de "
                     "convolution pour détecter des motifs visuels.",
        session_id="demo_session_001",
        metadata={"language": "fr", "topic": "deep_learning"}
    )
    print(f"✅ Conversation 2 sauvegardée (ID: {conv2_id})")
    
    # Récupérer l'historique
    print("\n📜 Historique de la session 'demo_session_001':")
    history = km.get_conversation_history(session_id="demo_session_001")
    for i, conv in enumerate(history, 1):
        print(f"   {i}. User: {conv['user_input'][:50]}...")
        print(f"      NETY: {conv['nety_response'][:50]}...")
    print()
    
    # ===============================
    # 7. STATISTIQUES
    # ===============================
    print("📊 Étape 7 : Statistiques de la base de connaissances")
    print("-" * 70)
    
    stats = km.get_stats()
    print(f"📈 Statistiques:")
    print(f"   - Nombre de connaissances : {stats['knowledge_count']}")
    print(f"   - Nombre de conversations : {stats['conversations_count']}")
    print(f"   - Catégories :")
    for category, count in stats['categories'].items():
        print(f"      • {category}: {count} connaissance(s)")
    print(f"   - Chroma DB disponible : {'✅' if stats['chroma_available'] else '❌'}")
    print(f"   - Redis disponible : {'✅' if stats['redis_available'] else '❌'}")
    print()
    
    # ===============================
    # 8. EXEMPLE RAG (Retrieval-Augmented Generation)
    # ===============================
    print("📊 Étape 8 : Exemple RAG - Génération augmentée par récupération")
    print("-" * 70)
    
    user_query = "Parle-moi des différents types de réseaux de neurones"
    print(f"❓ Question utilisateur : '{user_query}'")
    print()
    
    # Récupérer le contexte pertinent
    context = search.get_context_for_query(user_query, max_results=3)
    
    print("📚 Contexte récupéré de la base de connaissances:")
    print(f"   {len(context)} caractères de contexte pertinent")
    print()
    
    # Dans une vraie application, on utiliserait ce contexte avec un LLM
    print("💡 Utilisation typique:")
    print("   1. Récupérer le contexte pertinent (fait ✅)")
    print("   2. Combiner contexte + question utilisateur")
    print("   3. Envoyer au modèle de langage (Brain)")
    print("   4. Générer une réponse informée")
    print()
    
    # ===============================
    # CONCLUSION
    # ===============================
    print("=" * 70)
    print("✅ Démonstration terminée avec succès!")
    print()
    print("💡 La base de connaissances NETY est prête à être utilisée.")
    print("   - Les données sont stockées dans: data/databases/")
    print("   - SQLite pour les données structurées")
    print("   - Chroma DB pour la recherche sémantique (si disponible)")
    print("   - Redis pour le cache (si disponible et activé)")
    print("=" * 70)


if __name__ == "__main__":
    main()
