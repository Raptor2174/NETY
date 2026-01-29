"""
Moteur de recherche sémantique pour la base de connaissances NETY
"""
from typing import List, Dict, Optional

from .knowledge_manager import KnowledgeManager


class SearchEngine:
    """
    Moteur de recherche intelligent combinant recherche textuelle et sémantique
    """
    
    def __init__(self):
        self.knowledge_manager = KnowledgeManager()
    
    def search(
        self,
        query: str,
        category: str = None,
        limit: int = 10,
        use_semantic: bool = True
    ) -> List[Dict]:
        """
        Recherche des connaissances pertinentes
        
        Args:
            query: Texte de recherche
            category: Filtrer par catégorie (optionnel)
            limit: Nombre max de résultats
            use_semantic: Utiliser la recherche sémantique si disponible
            
        Returns:
            Liste de résultats de recherche
        """
        return self.knowledge_manager.search_knowledge(
            query=query,
            category=category,
            limit=limit,
            semantic=use_semantic
        )
    
    def find_similar(
        self,
        text: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Trouve des connaissances similaires à un texte donné
        
        Args:
            text: Texte de référence
            limit: Nombre de résultats similaires
            
        Returns:
            Liste de connaissances similaires
        """
        return self.knowledge_manager.search_knowledge(
            query=text,
            limit=limit,
            semantic=True
        )
    
    def get_context_for_query(
        self,
        query: str,
        max_results: int = 3
    ) -> str:
        """
        Récupère un contexte pertinent pour une requête
        Utilisé pour RAG (Retrieval-Augmented Generation)
        
        Args:
            query: Question ou requête de l'utilisateur
            max_results: Nombre max de connaissances à inclure
            
        Returns:
            Contexte textuel combiné
        """
        results = self.search(query, limit=max_results, use_semantic=True)
        
        if not results:
            return ""
        
        # Combiner les contenus pertinents
        context_parts = []
        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            context_parts.append(f"[{title}]\n{content}")
        
        return "\n\n".join(context_parts)
