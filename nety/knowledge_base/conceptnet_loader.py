"""
Chargeur ConceptNet pour NETY Knowledge Base
"""
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ConceptNetLoader:
    """Charge des relations sémantiques depuis ConceptNet"""
    
    def __init__(self):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests n'est pas installé. Installez-le avec: pip install requests")
        
        self.base_url = "http://api.conceptnet.io"
    
    def get_relations(self, concept: str, language: str = "fr") -> list:
        """
        Récupère les relations d'un concept
        
        Args:
            concept: Concept à rechercher
            language: Langue (défaut: français)
            
        Returns:
            Liste de relations
        """
        url = f"{self.base_url}/c/{language}/{concept}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"❌ Erreur lors de la requête ConceptNet: {e}")
            return []
        
        relations = []
        for edge in data.get('edges', [])[:10]:  # Top 10
            relations.append({
                'type': edge['rel']['label'],
                'related': edge['end']['label'],
                'weight': edge.get('weight', 1)
            })
        
        return relations