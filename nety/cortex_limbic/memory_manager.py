

# nety/cortex_limbic/memory_manager.py

class MemoryManager:
    """Gère la mémoire locale de NETY"""
    
    def __init__(self):
        self.consolidated_memories = []
        self.learned_rules = [
            "toujours être poli",
            "ne jamais partager d'infos personnelles"
        ]
    
    def get_memories(self) -> list:
        """Récupère les souvenirs consolidés"""
        return self.consolidated_memories
    
    def add_memory(self, summary: str):
        """Ajoute un nouveau souvenir"""
        self.consolidated_memories.append(summary)
        
        # Compression si trop de mémoires
        if len(self.consolidated_memories) > 100:
            self.compress_memories()
    
    def compress_memories(self):
        """Résume/compresse les anciennes mémoires"""
        # Logique de compression
        pass
    
    def get_rules(self) -> list:
        """Retourne les règles apprises"""
        return self.learned_rules