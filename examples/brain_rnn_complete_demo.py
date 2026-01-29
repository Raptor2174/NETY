#!/usr/bin/env python3
"""
Démonstration complète : Brain + RNN + Knowledge Base

Ce script montre comment NETY peut :
1. Recevoir une question
2. Récupérer le contexte de la base de connaissances (RAG)
3. Traiter avec le RNN
4. Générer une réponse
"""

import torch
from nety.knowledge_base import init_databases, KnowledgeManager, SearchEngine
from nety.modules.text.modele_rnn import ModeleRNN
from nety.modules.text.tokenizer import SimpleTokenizer


class NETYBrainWithRNN:
    """Brain NETY intégrant RNN et Knowledge Base"""
    
    def __init__(self):
        print("=" * 70)
        print(" 🧠 Initialisation NETY Brain + RNN + Knowledge Base")
        print("=" * 70)
        
        # 1. Initialize Knowledge Base
        print("\n📊 Initialisation de la base de connaissances...")
        init_databases()
        self.km = KnowledgeManager()
        self.search = SearchEngine()
        
        # 2. Initialize RNN
        print("🤖 Initialisation du modèle RNN...")
        self.rnn_model = ModeleRNN(
            input_size=1,
            hidden_size=64,
            output_size=10,  # 10 classes de réponses possibles
            num_layers=2
        )
        
        # 3. Initialize Tokenizer
        print("📝 Initialisation du tokenizer...")
        self.tokenizer = SimpleTokenizer(vocab_size=2000)
        
        # 4. Load initial knowledge
        self._load_initial_knowledge()
        
        print("\n✅ Initialisation terminée!\n")
    
    def _load_initial_knowledge(self):
        """Charge des connaissances de base"""
        print("\n📚 Chargement des connaissances de base...")
        
        knowledge_base = [
            {
                "title": "Intelligence Artificielle",
                "content": "L'intelligence artificielle (IA) est la simulation de processus d'intelligence humaine par des machines",
                "category": "ia"
            },
            {
                "title": "RNN",
                "content": "Les réseaux de neurones récurrents (RNN) sont adaptés pour traiter des séquences de données",
                "category": "deep_learning"
            },
            {
                "title": "Python",
                "content": "Python est un langage de programmation interprété, orienté objet et de haut niveau",
                "category": "programmation"
            },
            {
                "title": "NETY",
                "content": "NETY est une IA multimodale capable de traiter du texte, des images et de l'audio",
                "category": "projet"
            }
        ]
        
        # Check if already loaded
        stats = self.km.get_stats()
        if stats['knowledge_count'] >= len(knowledge_base):
            print(f"   ✅ {stats['knowledge_count']} connaissances déjà chargées")
            return
        
        # Add knowledge
        for kb in knowledge_base:
            self.km.add_knowledge(**kb)
        
        print(f"   ✅ {len(knowledge_base)} connaissances chargées")
        
        # Train tokenizer on all knowledge
        all_texts = [kb["content"] for kb in knowledge_base]
        self.tokenizer.fit(all_texts)
    
    def process_question(self, question: str) -> dict:
        """
        Traite une question avec RAG + RNN
        
        Returns:
            dict avec context, rnn_output, et answer
        """
        print(f"\n{'='*70}")
        print(f"❓ Question: {question}")
        print(f"{'='*70}")
        
        # 1. Retrieve context from KB
        print("\n📚 Étape 1: Récupération du contexte...")
        context = self.search.get_context_for_query(question, max_results=2)
        
        if context:
            print(f"   ✅ Contexte trouvé ({len(context)} caractères)")
            print(f"   Aperçu: {context[:100]}...")
        else:
            print("   ⚠️  Aucun contexte trouvé")
            context = ""
        
        # 2. Prepare input for RNN
        print("\n🔧 Étape 2: Préparation pour le RNN...")
        combined_input = f"{question} {context}"
        
        # Tokenize
        encoded = self.tokenizer.encode(combined_input[:200], max_length=30)
        print(f"   ✅ Texte tokenisé: {encoded.shape}")
        
        # Reshape for RNN: (batch=1, seq_len=30, input_size=1)
        x = encoded.unsqueeze(0).unsqueeze(-1).float()
        
        # 3. Process with RNN
        print("\n🤖 Étape 3: Traitement par le RNN...")
        with torch.no_grad():
            rnn_output = self.rnn_model(x)
        
        # Get predicted class
        predicted_class = torch.argmax(rnn_output, dim=1).item()
        confidence = torch.softmax(rnn_output, dim=1).max().item()
        
        print(f"   ✅ Classe prédite: {predicted_class}")
        print(f"   ✅ Confiance: {confidence:.2%}")
        
        # 4. Generate answer (simulation)
        print("\n💬 Étape 4: Génération de la réponse...")
        
        # Dans une vraie implémentation, on utiliserait un décodeur
        # Ici on simule avec le contexte
        if context:
            answer = f"D'après mes connaissances : {context[:150]}..."
        else:
            answer = "Je n'ai pas assez d'informations pour répondre."
        
        print(f"   ✅ Réponse générée")
        
        return {
            "question": question,
            "context": context,
            "rnn_output": rnn_output,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "answer": answer
        }
    
    def interactive_mode(self):
        """Mode interactif"""
        print("\n" + "="*70)
        print(" 🎮 Mode Interactif NETY")
        print("="*70)
        print("\nTapez 'exit' pour quitter\n")
        
        while True:
            try:
                question = input("💭 Vous: ")
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Au revoir!")
                    break
                
                if not question.strip():
                    continue
                
                result = self.process_question(question)
                
                print(f"\n🤖 NETY: {result['answer']}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir!")
                break


def main():
    """Demo principale"""
    
    # Create NETY Brain
    nety = NETYBrainWithRNN()
    
    # Demo questions
    demo_questions = [
        "Qu'est-ce que l'intelligence artificielle?",
        "Parle-moi des RNN",
        "C'est quoi Python?",
        "Qui est NETY?"
    ]
    
    print("\n" + "="*70)
    print(" 🎯 Démonstration avec Questions Prédéfinies")
    print("="*70)
    
    for question in demo_questions:
        result = nety.process_question(question)
        
        print(f"\n{'─'*70}")
        print(f"🤖 NETY répond:")
        print(f"{'─'*70}")
        print(result['answer'])
        print()
        
        input("Appuyez sur Entrée pour continuer...")
    
    # Interactive mode
    print("\n" + "="*70)
    print(" Voulez-vous passer en mode interactif? (y/n)")
    print("="*70)
    
    choice = input("Votre choix: ")
    if choice.lower() in ['y', 'yes', 'o', 'oui']:
        nety.interactive_mode()


if __name__ == "__main__":
    main()
