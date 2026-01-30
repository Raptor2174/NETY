"""
Chargeur Wikipedia pour NETY Knowledge Base
"""
try:
    import wikipediaapi
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

from .knowledge_manager import KnowledgeManager


class WikipediaLoader:
    """Charge des concepts depuis Wikipedia"""
    
    def __init__(self, km: KnowledgeManager):
        if not WIKIPEDIA_AVAILABLE:
            raise ImportError("wikipediaapi n'est pas installé. Installez-le avec: pip install wikipedia-api")
        
        self.wiki = wikipediaapi.Wikipedia(
            language='fr',
            user_agent='NETY/1.0 (https://github.com/Raptor2174/NETY)'
        )
        self.km = km
    
    def load_concept(self, concept: str) -> bool:
        """
        Charge un concept depuis Wikipedia
        
        Args:
            concept: Nom du concept à charger
            
        Returns:
            True si succès, False sinon
        """
        page = self.wiki.page(concept)
        if page.exists():
            self.km.add_knowledge(
                title=concept,
                content=page.summary[:1000],  # Limiter la taille
                category="wikipedia",
                source="wikipedia_fr",
                tags=["encyclopédie", concept.lower()]
            )
            return True
        return False
    
    def load_multiple_concepts(self, concepts: list) -> dict:
        """
        Charge plusieurs concepts
        
        Args:
            concepts: Liste de concepts à charger
            
        Returns:
            Dict avec statistiques de chargement
        """
        success = 0
        failed = 0
        
        for concept in concepts:
            if self.load_concept(concept):
                success += 1
                print(f"✅ Chargé: {concept}")
            else:
                failed += 1
                print(f"❌ Échec: {concept}")
        
        return {
            "success": success,
            "failed": failed,
            "total": len(concepts)
        }


# Script d'initialisation (à exécuter manuellement)
def initialize_base_concepts():
    """Initialise la base avec des concepts essentiels"""
    base_concepts = [
        "Chat", "Chien", "Arbre", "Voiture", 
        "Ordinateur", "Maison", "Eau", "Feu",
        "Intelligence artificielle", "Python (langage)",
        "France", "Paris", "Terre", "Soleil",
        "Musique", "Art", "Science", "Mathématiques"
    ]
    
    km = KnowledgeManager()
    loader = WikipediaLoader(km)
    results = loader.load_multiple_concepts(base_concepts)
    
    print("\n" + "=" * 60)
    print(f"✅ Chargement terminé: {results['success']}/{results['total']} concepts")
    print("=" * 60)


if __name__ == "__main__":
    initialize_base_concepts()