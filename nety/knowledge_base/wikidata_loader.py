# nety/knowledge_base/wikidata_loader.py
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataLoader:
    def __init__(self):
        self.sparql = SPARQLWrapper(
            "https://query.wikidata.org/sparql"
        )
    
    def get_entity_info(self, entity_name):
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
        return self.sparql.query().convert()