from typing import List, Dict
from datetime import datetime, timedelta
import numpy as np

class EmotionalMemory:
    """Mémoire émotionnelle pour NETY"""
    
    def __init__(self):
        self.episodic_memories = []  # Souvenirs spécifiques
        self.semantic_memories = {}   # Connaissances factuelles
        self.emotional_associations = {}  # Liens émotionnels
        
        # Paramètres de consolidation
        self.importance_threshold = 0.6
        self.max_short_term = 50
        self.max_long_term = 500
    
    def store_interaction(self, interaction: dict):
        """Stocke une interaction avec son contexte émotionnel"""
        
        # Calculer l'importance émotionnelle
        emotional_intensity = interaction.get("emotional_state", {}).get("intensity", 0.5)
        user_sentiment = interaction.get("user_sentiment", 0.0)
        novelty = self._calculate_novelty(interaction)
        
        importance = (emotional_intensity * 0.4 + 
                     abs(user_sentiment) * 0.4 + 
                     novelty * 0.2)
        
        memory_entry = {
            "timestamp": datetime.now(),
            "content": interaction.get("message"),
            "response": interaction.get("response"),
            "emotional_state": interaction.get("emotional_state"),
            "importance": importance,
            "access_count": 0,
            "last_accessed": None
        }
        
        # Stockage initial en mémoire épisodique
        self.episodic_memories.append(memory_entry)
        
        # Consolidation si important
        if importance > self.importance_threshold:
            self._consolidate_memory(memory_entry)
        
        # Gestion de la capacité
        if len(self.episodic_memories) > self.max_short_term:
            self._compress_memories()
    
    def retrieve_relevant_memories(self, context: dict, top_k: int = 5) -> List[dict]:
        """Récupère les souvenirs pertinents pour le contexte"""
        
        current_emotion = context.get("emotional_state", {}).get("dominant_emotion")
        query = context.get("message", "")
        
        scored_memories = []
        
        for memory in self.episodic_memories:
            # Score de similarité émotionnelle
            memory_emotion = memory.get("emotional_state", {}).get("dominant_emotion")
            emotional_similarity = 1.0 if memory_emotion == current_emotion else 0.3
            
            # Score de recency (plus récent = plus pertinent)
            time_diff = (datetime.now() - memory["timestamp"]).total_seconds()
            recency_score = np.exp(-time_diff / 86400)  # Décroissance exponentielle sur 24h
            
            # Score de pertinence sémantique (simpliste)
            semantic_score = self._calculate_semantic_similarity(query, memory.get("content", ""))
            
            # Score d'importance
            importance = memory["importance"]
            
            # Score final pondéré
            final_score = (emotional_similarity * 0.25 +
                          recency_score * 0.25 +
                          semantic_score * 0.3 +
                          importance * 0.2)
            
            scored_memories.append((memory, final_score))
        
        # Trier et retourner les top_k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Incrémenter le compteur d'accès
        for memory, _ in scored_memories[:top_k]:
            memory["access_count"] += 1
            memory["last_accessed"] = datetime.now()
        
        return [mem for mem, _ in scored_memories[:top_k]]
    
    def _consolidate_memory(self, memory: dict):
        """Consolide une mémoire importante en mémoire à long terme"""
        
        # Extraire les concepts clés (simpliste, à améliorer avec NLP)
        content = memory.get("content", "")
        keywords = self._extract_keywords(content)
        
        for keyword in keywords:
            if keyword not in self.semantic_memories:
                self.semantic_memories[keyword] = []
            
            self.semantic_memories[keyword].append({
                "reference": memory["timestamp"],
                "emotional_context": memory["emotional_state"],
                "importance": memory["importance"]
            })
    
    def _compress_memories(self):
        """Compresse les mémoires anciennes peu importantes"""
        
        # Trier par importance et recency
        sorted_memories = sorted(
            self.episodic_memories,
            key=lambda m: m["importance"] * (1 + m["access_count"] * 0.1),
            reverse=True
        )
        
        # Garder les plus importantes
        self.episodic_memories = sorted_memories[:self.max_short_term]
    
    def _calculate_novelty(self, interaction: dict) -> float:
        """Calcule la nouveauté d'une interaction"""
        # Comparer avec les interactions récentes
        recent = self.episodic_memories[-10:] if len(self.episodic_memories) > 10 else self.episodic_memories
        
        if not recent:
            return 1.0
        
        content = interaction.get("message", "")
        similarities = [self._calculate_semantic_similarity(content, mem.get("content", "")) 
                       for mem in recent]
        
        avg_similarity = np.mean(similarities) if similarities else 0
        return float(1.0 - avg_similarity)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcul simple de similarité (à remplacer par embedding)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extraction basique de mots-clés (à améliorer avec TF-IDF)"""
        stopwords = {"le", "la", "les", "un", "une", "de", "du", "des", "et", "ou", "à"}
        words = text.lower().split()
        return [w for w in words if len(w) > 3 and w not in stopwords][:5]