"""
Chargeur Wikidata pour NETY Knowledge Base
"""
try:
    from SPARQLWrapper import SPARQLWrapper
    WIKIDATA_AVAILABLE = True
except ImportError:
    WIKIDATA_AVAILABLE = False
    SPARQLWrapper = None


class WikidataLoader:
    """Charge des informations depuis Wikidata via SPARQL"""
    
    def __init__(self):
        if not WIKIDATA_AVAILABLE:
            raise ImportError(
                "SPARQLWrapper n'est pas installé.\n"
                "Installez-le avec: pip install sparqlwrapper"
            )
        
        if SPARQLWrapper is None:
            raise ImportError(
                "SPARQLWrapper n'est pas disponible."
            )
        
        self.sparql = SPARQLWrapper(
            "https://query.wikidata.org/sparql"
        )
        # Ajouter un User-Agent pour respecter les bonnes pratiques
        self.sparql.addCustomHttpHeader("User-Agent", "NETY/1.0 (https://github.com/Raptor2174/NETY)")
    
    def get_entity_info(self, entity_name: str) -> dict:
        """
        Récupère des informations sur une entité
        
        Args:
            entity_name: Nom de l'entité en français
            
        Returns:
            Résultats de la requête SPARQL (dict avec 'results' et 'bindings')
        """
        if not entity_name or not isinstance(entity_name, str):
            return {"results": {"bindings": []}}
        
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
        
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat("json")
            result = self.sparql.query().convert()
            # Le résultat est déjà un dict après .convert() avec le format "json"
            if isinstance(result, dict):
                return {"results": result.get("results", {"bindings": []})}
            return {"results": {"bindings": []}}
        
        except Exception as e:
            print(f"❌ Erreur lors de la requête Wikidata pour '{entity_name}': {e}")
            return {"results": {"bindings": []}}