# nety/knowledge_base/knowledge_manager.py


class KnowledgeManager:
    """
    Gère la base de connaissances de NETY
    Recherche et récupère les informations pertinentes
    """
    
    def __init__(self):
        # Base de connaissances simple (à étendre avec une vraie base de données)
        self.knowledge_base = {
            "identité": {
                "nom": "NETY",
                "type": "IA de traitement du langage naturel",
                "créateur": "Raptor_",
                "objectif": "Apprendre et aider les utilisateurs"
            },
            "capacités": {
                "traitement_texte": "Analyse et génération de texte en français",
                "traitement_image": "Reconnaissance d'images avec CNN",
                "traitement_audio": "Conversion Speech-to-Text"
            }
        }
    
    def search(self, query: str, intent: dict) -> dict:
        """
        Recherche dans la base de connaissances
        
        Args:
            query: La requête de recherche
            intent: L'intention détectée par l'IntentAnalyzer
        
        Returns:
            Contexte de connaissances pertinent
        """
        if not query or not isinstance(query, str):
            return {"results": [], "relevance": 0.0}
        
        query_lower = query.lower()
        
        # Recherche simple par mots-clés
        relevant_knowledge = {}
        relevance_score = 0.0
        
        # Chercher dans chaque catégorie de la base de connaissances
        for category, content in self.knowledge_base.items():
            if category in query_lower or self._matches_keywords(query_lower, category):
                relevant_knowledge[category] = content
                relevance_score += 0.3
        
        # Si aucune correspondance, retourner des infos de base
        if not relevant_knowledge:
            relevant_knowledge = {
                "identité": self.knowledge_base.get("identité", {})
            }
            relevance_score = 0.1
        
        return {
            "results": relevant_knowledge,
            "relevance": min(relevance_score, 1.0),
            "query": query,
            "intent_type": intent.get("type", "unknown")
        }
    
    def _matches_keywords(self, query: str, category: str) -> bool:
        """
        Vérifie si la requête correspond à des mots-clés de la catégorie
        
        Args:
            query: La requête en minuscules
            category: La catégorie à vérifier
        
        Returns:
            True si correspondance trouvée
        """
        # Mots-clés associés aux catégories
        category_keywords = {
            "identité": ["qui", "nom", "es-tu", "êtes-vous", "toi"],
            "capacités": ["peux-tu", "capable", "faire", "fonction", "capacité"]
        }
        
        keywords = category_keywords.get(category, [])
        return any(keyword in query for keyword in keywords)
    
    def add_knowledge(self, category: str, key: str, value: str):
        """
        Ajoute une nouvelle connaissance à la base
        
        Args:
            category: Catégorie de la connaissance
            key: Clé de la connaissance
            value: Valeur de la connaissance
        """
        if category not in self.knowledge_base:
            self.knowledge_base[category] = {}
        
        if isinstance(self.knowledge_base[category], dict):
            self.knowledge_base[category][key] = value
