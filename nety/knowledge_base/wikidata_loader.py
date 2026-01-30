"""
Chargeur Wikidata pour NETY Knowledge Base
"""
try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    WIKIDATA_AVAILABLE = True
except ImportError:
    WIKIDATA_AVAILABLE = False


class WikidataLoader:
    """Charge des informations depuis Wikidata via SPARQL"""
    
    def __init__(self):
        if not WIKIDATA_AVAILABLE:
            raise ImportError("SPARQLWrapper n'est pas installé. Installez-le avec: pip install sparqlwrapper")
        
        self.sparql = SPARQLWrapper(
            "https://query.wikidata.org/sparql",
            agent='NETY/1.0 (https://github.com/Raptor2174/NETY)'
        )
    
    def get_entity_info(self, entity_name: str) -> dict:
        """
        Récupère des informations sur une entité
        
        Args:
            entity_name: Nom de l'entité en français
            
        Returns:
            Résultats de la requête SPARQL
        """
        query = f"""
        SELECT ?item ?itemLabel ?description WHERE {{
          ?item rdfs:label "{entity_name}"@fr .
          ?item schema:description ?description .
          FILTER(LANG(?description) = "fr")
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "fr". 
          }}
        }}
        LIMIT 5
        """
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        
        try:
            return self.sparql.query().convert()
        except Exception as e:
            print(f"❌ Erreur lors de la requête Wikidata: {e}")
            return {"results": {"bindings": []}}