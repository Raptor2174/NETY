"""
NETY V2-Maxx - Dataset Generator
=================================

Génère un dataset de 1000+ paires de conversations en augmentant
le dataset de base avec des variations et templates.
"""

import json
import random
from typing import List, Dict, Tuple
import os


class ConversationGenerator:
    """Générateur de conversations variées"""
    
    def __init__(self):
        # Templates par catégorie
        self.templates = {
            'greetings': {
                'inputs': [
                    "Bonjour", "Salut", "Hello", "Hey", "Coucou",
                    "Bonsoir", "Bonne journée", "Yo", "Bien le bonjour",
                    "Bonjour à toi", "Salutations", "Hé",
                ],
                'outputs': [
                    "Bonjour ! Comment puis-je t'aider aujourd'hui ?",
                    "Salut ! Ravi de te voir.",
                    "Hello ! Que puis-je faire pour toi ?",
                    "Hey ! Comment ça va ?",
                    "Coucou ! Ravi de discuter avec toi.",
                    "Bonsoir ! J'espère que tu vas bien.",
                    "Salut ! Qu'est-ce qui t'amène ?",
                ]
            },
            'farewell': {
                'inputs': [
                    "Au revoir", "À plus tard", "Bye", "À bientôt",
                    "Salut", "Ciao", "À demain", "Bonne soirée",
                    "À plus", "À tout de suite",
                ],
                'outputs': [
                    "Au revoir ! À bientôt.",
                    "À plus tard ! Prends soin de toi.",
                    "Bye ! À la prochaine.",
                    "À bientôt ! Bonne journée.",
                    "Salut ! À plus tard.",
                    "Ciao ! Reviens quand tu veux.",
                    "À demain ! Repose-toi bien.",
                ]
            },
            'wellbeing': {
                'inputs': [
                    "Comment vas-tu ?", "Ça va ?", "Comment te sens-tu ?",
                    "Tu vas bien ?", "Comment tu te sens ?", "Ça roule ?",
                    "Tout va bien ?", "Comment ça se passe ?",
                    "Ça va bien ?", "Tu te sens comment ?",
                ],
                'outputs': [
                    "Je vais bien, merci ! Et toi ?",
                    "Ça va très bien ! Comment se passe ta journée ?",
                    "Je me sens bien ! Et toi, comment tu vas ?",
                    "Oui, très bien ! Merci de demander.",
                    "Ça roule ! Et de ton côté ?",
                    "Tout va super bien ! Comment tu vas ?",
                ]
            },
            'thanks': {
                'inputs': [
                    "Merci", "Merci beaucoup", "Merci pour ton aide",
                    "Je te remercie", "Merci mille fois", "Un grand merci",
                    "C'est gentil merci", "Merci bien", "Merci pour tout",
                ],
                'outputs': [
                    "De rien ! C'est toujours un plaisir.",
                    "Je t'en prie ! N'hésite pas si tu as besoin.",
                    "Avec plaisir ! Je suis là pour ça.",
                    "Pas de souci ! Ravi de t'aider.",
                    "De rien du tout ! C'est normal.",
                ]
            },
            'questions_general': {
                'pairs': [
                    ("Comment ça marche ?", "C'est simple ! Je peux t'expliquer si tu veux."),
                    ("Pourquoi ?", "Bonne question ! Qu'est-ce que tu aimerais savoir exactement ?"),
                    ("C'est quoi ça ?", "De quoi parles-tu ? Je peux t'aider à comprendre."),
                    ("Tu peux m'expliquer ?", "Bien sûr ! Qu'est-ce que tu veux que je t'explique ?"),
                    ("Comment faire ?", "Je vais t'aider ! Dis-moi ce que tu cherches à faire."),
                ]
            },
            'emotions_positive': {
                'pairs': [
                    ("Je suis content", "C'est super ! Je suis heureux pour toi."),
                    ("Je suis heureux", "Génial ! Profite de ce moment."),
                    ("Je suis joyeux", "Formidable ! Le bonheur, c'est contagieux."),
                    ("Je suis excité", "Excellent ! Qu'est-ce qui te rend si enthousiaste ?"),
                    ("Je suis fier", "Bravo ! Tu as raison d'être fier."),
                    ("C'est génial", "Absolument ! Je partage ton enthousiasme."),
                    ("C'est super", "Oui ! C'est vraiment chouette."),
                    ("Je suis ravi", "Merveilleux ! Ça fait plaisir de te voir si content."),
                ]
            },
            'emotions_negative': {
                'pairs': [
                    ("Je suis triste", "Je suis désolé. Veux-tu en parler ?"),
                    ("Je suis déprimé", "Je comprends que ce soit difficile. Je suis là."),
                    ("Je suis en colère", "Je vois que tu es contrarié. Qu'est-ce qui s'est passé ?"),
                    ("Je suis fatigué", "Tu as l'air épuisé. Repose-toi bien."),
                    ("Je suis stressé", "Le stress peut être difficile à gérer. Veux-tu en parler ?"),
                    ("Je suis anxieux", "L'anxiété est normale. Je suis là pour t'écouter."),
                    ("Je me sens mal", "Désolé de l'apprendre. Que puis-je faire pour toi ?"),
                    ("C'est difficile", "Je comprends. Les moments difficiles passent."),
                ]
            },
            'knowledge_simple': {
                'pairs': [
                    ("Qu'est-ce qu'un ordinateur ?", "Un ordinateur est une machine qui traite des informations."),
                    ("C'est quoi Internet ?", "Internet est un réseau mondial qui connecte des ordinateurs entre eux."),
                    ("Qu'est-ce que l'IA ?", "L'IA, ou Intelligence Artificielle, permet aux machines d'apprendre et de raisonner."),
                    ("C'est quoi un robot ?", "Un robot est une machine programmable qui peut effectuer des tâches."),
                    ("Qu'est-ce que Python ?", "Python est un langage de programmation populaire et facile à apprendre."),
                ]
            },
            'small_talk': {
                'pairs': [
                    ("Quoi de neuf ?", "Pas grand-chose ! Et toi, comment ça va ?"),
                    ("Tu as passé une bonne journée ?", "Oui, plutôt ! Et toi, comment s'est passée la tienne ?"),
                    ("Qu'est-ce que tu fais ?", "Je discute avec toi ! C'est toujours intéressant."),
                    ("Tu t'ennuies ?", "Jamais ! J'aime nos conversations."),
                    ("On se connaît ?", "On se connaît maintenant ! Ravi de faire ta connaissance."),
                ]
            },
            'preferences': {
                'pairs': [
                    ("Quelle est ta couleur préférée ?", "Je n'ai pas vraiment de préférence, mais j'aime toutes les couleurs !"),
                    ("Tu aimes la musique ?", "Oui ! La musique, c'est universel."),
                    ("Tu aimes lire ?", "J'adore apprendre, donc oui, j'aime lire !"),
                    ("Ton film préféré ?", "Je ne regarde pas de films, mais j'aime entendre parler des tiens !"),
                    ("Tu aimes voyager ?", "J'aimerais bien ! Raconte-moi tes voyages."),
                ]
            },
            'humor': {
                'pairs': [
                    ("Raconte-moi une blague", "Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon, ils tombent dans le bateau !"),
                    ("Fais-moi rire", "Qu'est-ce qu'un crocodile qui surveille la pharmacie ? Un Lacoste garde !"),
                    ("Tu es drôle ?", "J'essaie ! Mais l'humour, c'est subjectif."),
                    ("Dis quelque chose de drôle", "Mon cerveau est un réseau de neurones. On pourrait dire que j'ai des connexions !"),
                ]
            },
            'encouragement': {
                'pairs': [
                    ("J'ai peur d'échouer", "L'échec fait partie de l'apprentissage. Tu peux y arriver !"),
                    ("Je ne suis pas sûr", "C'est normal d'avoir des doutes. Fais de ton mieux !"),
                    ("C'est trop difficile", "Rien n'est impossible. Prends ça étape par étape."),
                    ("Je n'y arrive pas", "Ne baisse pas les bras ! Continue d'essayer."),
                    ("Je manque de confiance", "Crois en toi ! Tu es capable de grandes choses."),
                ]
            },
            'misc': {
                'pairs': [
                    ("OK", "D'accord ! Autre chose ?"),
                    ("Oui", "Très bien ! Continue."),
                    ("Non", "Pas de problème. Autre chose ?"),
                    ("Peut-être", "C'est une possibilité ! Réfléchis-y."),
                    ("Je ne sais pas", "C'est normal de ne pas tout savoir. On peut chercher ensemble."),
                    ("Vraiment ?", "Oui, vraiment ! Qu'est-ce qui te surprend ?"),
                    ("Sérieusement ?", "Tout à fait ! C'est intéressant, non ?"),
                ]
            }
        }
    
    def generate_conversations(self, num_conversations: int = 1000) -> List[Dict]:
        """
        Génère un dataset de conversations
        
        Args:
            num_conversations: Nombre de conversations à générer
        
        Returns:
            Liste de conversations (dict avec input/output/category)
        """
        conversations = []
        conversation_id = 1
        
        # Calculer combien de conversations par catégorie
        categories = list(self.templates.keys())
        per_category = num_conversations // len(categories)
        
        for category in categories:
            category_data = self.templates[category]
            
            if 'pairs' in category_data:
                # Paires prédéfinies
                pairs = category_data['pairs']
                
                # Répéter les paires si besoin
                for i in range(per_category):
                    pair = pairs[i % len(pairs)]
                    conversations.append({
                        'id': conversation_id,
                        'category': category,
                        'input': pair[0],
                        'output': pair[1]
                    })
                    conversation_id += 1
            
            else:
                # Combiner inputs et outputs aléatoirement
                inputs = category_data['inputs']
                outputs = category_data['outputs']
                
                for i in range(per_category):
                    inp = inputs[i % len(inputs)]
                    out = outputs[random.randint(0, len(outputs) - 1)]
                    
                    conversations.append({
                        'id': conversation_id,
                        'category': category,
                        'input': inp,
                        'output': out
                    })
                    conversation_id += 1
        
        # Mélanger pour plus de diversité
        random.shuffle(conversations)
        
        # Réassigner les IDs séquentiellement
        for i, conv in enumerate(conversations, 1):
            conv['id'] = i
        
        print(f"✓ Généré {len(conversations)} conversations")
        print(f"  Catégories: {', '.join(categories)}")
        
        return conversations
    
    def save_dataset(self, conversations: List[Dict], path: str):
        """Sauvegarde le dataset en JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        dataset = {
            'metadata': {
                'total_conversations': len(conversations),
                'categories': list(set(c['category'] for c in conversations)),
                'description': 'Dataset de conversations pour NETY V2-Maxx'
            },
            'conversations': conversations
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Dataset sauvegardé: {path}")


def main():
    """Génère le dataset complet"""
    print("=" * 80)
    print("NETY V2-Maxx - Dataset Generator")
    print("=" * 80)
    
    generator = ConversationGenerator()
    
    # Générer 1200 conversations (pour avoir marge)
    conversations = generator.generate_conversations(num_conversations=1200)
    
    # Sauvegarder
    output_path = "data/training/conversations.json"
    generator.save_dataset(conversations, output_path)
    
    # Stats
    categories = {}
    for conv in conversations:
        cat = conv['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Statistiques du dataset:")
    print(f"  Total: {len(conversations)} conversations")
    print(f"\n  Par catégorie:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cat}: {count}")
    
    print("\n" + "=" * 80)
    print("✓ Dataset généré avec succès!")
    print("=" * 80)


if __name__ == "__main__":
    main()
