# nety/knowledge_base/conceptnet_loader.py
import requests

class ConceptNetLoader:
    def get_relations(self, concept):
        url = f"http://api.conceptnet.io/c/fr/{concept}"
        response = requests.get(url).json()
        
        relations = []
        for edge in response.get('edges', [])[:10]:  # Top 10
            relations.append({
                'type': edge['rel']['label'],
                'related': edge['end']['label'],
                'weight': edge.get('weight', 1)
            })
        
        return relations
    
    