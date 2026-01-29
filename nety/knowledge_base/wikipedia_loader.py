# nety/knowledge_base/wikipedia_loader.py
import wikipediaapi
from nety.knowledge_base.knowledge_manager import KnowledgeManager

class WikipediaLoader:
    def __init__(self, km):
        self.wiki = wikipediaapi.Wikipedia('fr')
        self.km = km
    
    def load_concept(self, concept):
        page = self.wiki.page(concept)
        if page.exists():
            self.km.add_knowledge(
                title=concept,
                content=page.summary[:1000],  # Limiter taille
                category="wikipedia",
                source="wikipedia_fr",
                tags=["encyclopédie", concept.lower()]
            )
            return True
        return False
    
    # Script d'initialisation
base_concepts = [
    "Chat", "Chien", "Arbre", "Voiture", 
    "Ordinateur", "Maison", "Eau", "Feu",
    "Intelligence artificielle", "Python (langage)",
    # ... 100-200 concepts essentiels
]

km = KnowledgeManager()
loader = WikipediaLoader(km)
for concept in base_concepts:
    loader.load_concept(concept)