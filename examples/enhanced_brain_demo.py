"""
Exemple d'intégration de la base de connaissances avec le Brain de NETY
Ce fichier montre comment enrichir les réponses de NETY avec des connaissances stockées
"""

from nety.knowledge_base import KnowledgeManager, SearchEngine, init_databases


class EnhancedBrain:
    """
    Version enrichie du Brain NETY avec accès à la base de connaissances
    """
    
    def __init__(self):
        # Initialiser la base de connaissances
        init_databases()
        
        self.km = KnowledgeManager()
        self.search = SearchEngine()
        
        # Initialiser avec quelques connaissances de base sur NETY
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Initialise la base avec des connaissances sur NETY"""
        
        # Vérifier si déjà initialisé
        stats = self.km.get_stats()
        if stats['knowledge_count'] > 0:
            print("📚 Base de connaissances déjà initialisée")
            return
        
        print("📚 Initialisation de la base de connaissances...")
        
        # Ajouter des connaissances sur NETY
        self.km.add_knowledge(
            title="Qu'est-ce que NETY?",
            content="""NETY est un projet d'intelligence artificielle multimodale 
            conçu pour le traitement du langage naturel (NLP), le traitement d'images (CNN) 
            et le traitement audio (Speech-to-Text). Le projet utilise PyTorch et TensorFlow 
            comme frameworks principaux et possède une architecture modulaire et extensible.""",
            category="about_nety",
            tags=["nety", "ia", "multimodal"],
            metadata={"priority": "high", "language": "fr"}
        )
        
        self.km.add_knowledge(
            title="RNN - Réseaux de Neurones Récurrents",
            content="""Les réseaux de neurones récurrents (RNN) sont utilisés dans NETY 
            pour le traitement du langage naturel. Ils possèdent des connexions récurrentes 
            qui leur permettent de maintenir une mémoire des entrées précédentes, 
            ce qui est essentiel pour comprendre le contexte dans les conversations.""",
            category="deep_learning",
            tags=["rnn", "nlp", "neural_networks"],
            metadata={"difficulty": "intermediate"}
        )
        
        self.km.add_knowledge(
            title="CNN - Réseaux de Neurones Convolutionnels",
            content="""Les réseaux de neurones convolutionnels (CNN) sont utilisés dans NETY 
            pour le traitement d'images. Ils utilisent des couches de convolution pour détecter 
            automatiquement des motifs et des caractéristiques visuelles dans les images.""",
            category="deep_learning",
            tags=["cnn", "computer_vision", "neural_networks"],
            metadata={"difficulty": "intermediate"}
        )
        
        print("✅ Base de connaissances initialisée avec succès")
    
    def think(self, user_input: str, session_id: str = None) -> str:
        """
        Processus de réflexion enrichi par la base de connaissances
        
        Args:
            user_input: Message de l'utilisateur
            session_id: ID de session pour le suivi
            
        Returns:
            Réponse de NETY enrichie par les connaissances
        """
        print(f"\n💭 NETY réfléchit à: '{user_input}'")
        
        # 1. Récupérer le contexte pertinent de la base de connaissances
        context = self.search.get_context_for_query(user_input, max_results=2)
        
        if context:
            print(f"📚 Contexte récupéré: {len(context)} caractères")
        else:
            print("📚 Aucun contexte spécifique trouvé")
        
        # 2. Générer la réponse (simulation)
        # Dans la vraie implémentation, ceci serait envoyé au modèle RNN/LLM
        response = self._generate_response(user_input, context)
        
        # 3. Sauvegarder la conversation
        self.km.save_conversation(
            user_input=user_input,
            nety_response=response,
            context=context[:500] if context else None,  # Limiter la taille
            session_id=session_id,
            metadata={"has_context": bool(context)}
        )
        
        return response
    
    def _generate_response(self, user_input: str, context: str) -> str:
        """
        Simule la génération de réponse
        Dans la vraie implémentation, ceci utiliserait le modèle RNN/LLM
        """
        if not context:
            return "Je n'ai pas assez d'informations dans ma base de connaissances pour répondre précisément à cette question."
        
        # Simulation simplifiée
        return f"Basé sur mes connaissances: {context[:200]}..."
    
    def add_knowledge_from_conversation(
        self,
        title: str,
        content: str,
        category: str = "learned"
    ):
        """
        Permet à NETY d'apprendre de nouvelles connaissances
        """
        knowledge_id = self.km.add_knowledge(
            title=title,
            content=content,
            category=category,
            source="conversation",
            tags=["learned", "user_input"]
        )
        print(f"📖 Nouvelle connaissance apprise (ID: {knowledge_id})")
        return knowledge_id
    
    def get_knowledge_stats(self):
        """Retourne les statistiques de la base de connaissances"""
        return self.km.get_stats()


def demo_enhanced_brain():
    """Démonstration du Brain enrichi"""
    
    print("=" * 70)
    print(" 🧠 NETY Enhanced Brain - Démonstration")
    print("=" * 70)
    print()
    
    # Créer le Brain enrichi
    brain = EnhancedBrain()
    
    # Session de conversation
    session_id = "demo_session_001"
    
    # Conversation 1
    print("\n" + "-" * 70)
    user_input_1 = "Qu'est-ce que NETY?"
    response_1 = brain.think(user_input_1, session_id)
    print(f"\n👤 Utilisateur: {user_input_1}")
    print(f"🤖 NETY: {response_1}")
    
    # Conversation 2
    print("\n" + "-" * 70)
    user_input_2 = "Comment fonctionnent les RNN?"
    response_2 = brain.think(user_input_2, session_id)
    print(f"\n👤 Utilisateur: {user_input_2}")
    print(f"🤖 NETY: {response_2}")
    
    # Conversation 3 - Apprendre quelque chose de nouveau
    print("\n" + "-" * 70)
    print("\n📖 NETY apprend une nouvelle connaissance...")
    brain.add_knowledge_from_conversation(
        title="LSTM - Long Short-Term Memory",
        content="""LSTM est une variante améliorée des RNN qui résout le problème 
        du gradient qui disparaît. Les LSTM utilisent des portes (gates) pour contrôler 
        le flux d'information et peuvent maintenir des dépendances à long terme.""",
        category="deep_learning"
    )
    
    # Conversation 4 - Utiliser la nouvelle connaissance
    print("\n" + "-" * 70)
    user_input_3 = "Parle-moi des LSTM"
    response_3 = brain.think(user_input_3, session_id)
    print(f"\n👤 Utilisateur: {user_input_3}")
    print(f"🤖 NETY: {response_3}")
    
    # Afficher les statistiques
    print("\n" + "=" * 70)
    print(" 📊 Statistiques de la Base de Connaissances")
    print("=" * 70)
    stats = brain.get_knowledge_stats()
    print(f"Connaissances stockées: {stats['knowledge_count']}")
    print(f"Conversations sauvegardées: {stats['conversations_count']}")
    print(f"Catégories:")
    for cat, count in stats['categories'].items():
        print(f"  - {cat}: {count}")
    
    print("\n" + "=" * 70)
    print("✅ Démonstration terminée")
    print("=" * 70)


if __name__ == "__main__":
    demo_enhanced_brain()
