

# nety/cortex_limbic/memory_manager.py

from typing import Dict, List, Optional
from datetime import datetime
import json
import os


class MemoryManager:
    """Gère la mémoire locale de NETY avec support pour les labels et corrélations"""
    
    def __init__(self):
        self.consolidated_memories = []
        self.memory_labels = {}  # Mappe memory_id -> labels
        self.correlations = {}   # Mappe memory_id -> [related_ids]
        self.learned_rules = [
            "toujours être poli",
            "ne jamais partager d'infos personnelles"
        ]
    
    def get_memories(self) -> list:
        """Récupère les souvenirs consolidés"""
        return self.consolidated_memories
    
    def add_memory(self, summary: str, labels: Optional[List[str]] = None, 
                   metadata: Optional[Dict] = None) -> str:
        """
        Ajoute un nouveau souvenir avec labels et métadonnées.
        Retourne l'ID du souvenir créé.
        """
        memory_id = f"{datetime.utcnow().isoformat()}-{len(summary)}"
        
        memory_entry = {
            "id": memory_id,
            "timestamp": datetime.utcnow().isoformat(),
            "content": summary,
            "labels": labels or ["untagged"],
            "metadata": metadata or {},
        }
        
        self.consolidated_memories.append(memory_entry)
        if labels:
            self.memory_labels[memory_id] = labels
        
        # Compression si trop de mémoires
        if len(self.consolidated_memories) > 100:
            self.compress_memories()
        
        return memory_id
    
    def add_correlation(self, source_id: str, target_id: str, 
                       correlation_type: str = "related") -> None:
        """
        Enregistre une corrélation entre deux souvenirs.
        """
        if source_id not in self.correlations:
            self.correlations[source_id] = []
        
        self.correlations[source_id].append({
            "target_id": target_id,
            "type": correlation_type,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_memory_with_labels(self, memory_id: str) -> Optional[Dict]:
        """Récupère un souvenir avec ses labels associés"""
        for memory in self.consolidated_memories:
            if memory["id"] == memory_id:
                return {
                    **memory,
                    "labels": self.memory_labels.get(memory_id, ["untagged"]),
                    "correlations": self.correlations.get(memory_id, [])
                }
        return None
    
    def get_memories_by_label(self, label: str) -> List[Dict]:
        """Récupère tous les souvenirs avec un label spécifique"""
        matching = []
        for memory_id, labels in self.memory_labels.items():
            if label in labels:
                mem = self.get_memory_with_labels(memory_id)
                if mem:
                    matching.append(mem)
        return matching
    
    def get_recent_memories(self, count: int = 5) -> List[Dict]:
        """Récupère les N souvenirs les plus récents avec leurs métadonnées"""
        recent = sorted(
            self.consolidated_memories,
            key=lambda m: m["timestamp"],
            reverse=True
        )[:count]
        
        return [
            self.get_memory_with_labels(m["id"]) or m
            for m in recent
        ]
    
    def compress_memories(self) -> None:
        """
        Résume/compresse les anciennes mémoires.
        Garde les récentes avec labels importants.
        """
        if len(self.consolidated_memories) <= 100:
            return
        
        # Garder les mémoires avec labels "identity_info" ou "goal"
        important_labels = {"identity_info", "goal", "health_update"}
        
        important = []
        unimportant = []
        
        for mem in self.consolidated_memories:
            mem_labels = set(self.memory_labels.get(mem["id"], []))
            if mem_labels & important_labels:
                important.append(mem)
            else:
                unimportant.append(mem)
        
        # Garder tous les importants + les 50 les plus récents des non-importants
        recent_unimportant = sorted(
            unimportant,
            key=lambda m: m["timestamp"],
            reverse=True
        )[:50]
        
        self.consolidated_memories = important + recent_unimportant
    
    def get_rules(self) -> list:
        """Retourne les règles apprises"""
        return self.learned_rules
    
    def add_learned_rule(self, rule: str) -> None:
        """Ajoute une nouvelle règle apprise"""
        if rule not in self.learned_rules:
            self.learned_rules.append(rule)