"""
Gestionnaire de la base de connaissances NETY
Gère le stockage, la récupération et la recherche de connaissances
"""
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
# nety/knowledge_base/knowledge_manager.py

from .database_connector import DatabaseConnector
from .database_config import DatabaseConfig
from typing import Any as ChromaClient
try:
    from redis import Redis
except ImportError:
    Redis = Any

from typing import Dict, Any


class KnowledgeManager:
    """
    Gestionnaire principal de la base de connaissances
    Combine SQLite pour les données structurées et Chroma pour la recherche sémantique
    """
    
    def __init__(self):
        self.sqlite_conn = DatabaseConnector.get_sqlite_connection()
        self.chroma_client: Any = DatabaseConnector.get_chroma_client()
        self.redis_client: Optional[Any] = DatabaseConnector.get_redis_client()
        
        # Lazy loading de l'embedding model
        self._embedding_model = None
        
        # Récupérer les collections Chroma
        if self.chroma_client:
            try:
                self.knowledge_collection = self.chroma_client.get_collection(
                    DatabaseConfig.CHROMA_COLLECTION_KNOWLEDGE
                )
                self.conversations_collection = self.chroma_client.get_collection(
                    DatabaseConfig.CHROMA_COLLECTION_CONVERSATIONS
                )
            except Exception as e:
                print(f"⚠️  Erreur lors de la récupération des collections Chroma: {e}")
                self.knowledge_collection = None
                self.conversations_collection = None
        else:
            self.knowledge_collection = None
            self.conversations_collection = None
    
    # =====================
    # GESTION DES CONNAISSANCES
    # =====================
    
    def add_knowledge(
        self,
        title: str,
        content: str,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Ajoute une nouvelle connaissance à la base
        
        Args:
            title: Titre de la connaissance
            content: Contenu textuel
            category: Catégorie (optionnel)
            source: Source de l'information (optionnel)
            tags: Liste de tags (optionnel)
            metadata: Métadonnées additionnelles (optionnel)
            
        Returns:
            ID de la connaissance créée
        """
        # Préparer les tags et metadata
        tags_str = json.dumps(tags) if tags else None
        metadata_str = json.dumps(metadata) if metadata else None
        
        # Insérer dans SQLite
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("""
                INSERT INTO knowledge (title, content, category, source, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, content, category, source, tags_str, metadata_str))
            
            knowledge_id = cursor.lastrowid or 0
        
        # Ajouter l'embedding dans Chroma si disponible
        if self.knowledge_collection:
            try:
                self.knowledge_collection.add(
                    documents=[content],
                    metadatas=[{
                        "title": title,
                        "category": category or "",
                        "source": source or "",
                        "knowledge_id": knowledge_id
                    }],
                    ids=[f"knowledge_{knowledge_id}"]
                )
            except Exception as e:
                print(f"⚠️  Erreur lors de l'ajout de l'embedding: {e}")
        
        # Invalider le cache Redis si disponible
        if self.redis_client:
            self.redis_client.delete("knowledge:all")
        
        return knowledge_id
    
    def get_knowledge(self, knowledge_id: int) -> Optional[Dict]:
        """
        Récupère une connaissance par son ID
        
        Args:
            knowledge_id: ID de la connaissance
            
        Returns:
            Dict avec les données de la connaissance ou None
        """
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM knowledge WHERE id = ?
            """, (knowledge_id,))
            
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            
        return None
    
    def update_knowledge(
        self,
        knowledge_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Met à jour une connaissance existante
        
        Args:
            knowledge_id: ID de la connaissance à mettre à jour
            title, content, category, source, tags, metadata: Nouveaux valeurs (optionnel)
            
        Returns:
            True si mise à jour réussie, False sinon
        """
        # Récupérer les valeurs actuelles
        current = self.get_knowledge(knowledge_id)
        if not current:
            return False
        
        # Préparer les nouvelles valeurs
        new_title = title if title is not None else current["title"]
        new_content = content if content is not None else current["content"]
        new_category = category if category is not None else current["category"]
        new_source = source if source is not None else current["source"]
        
        # Pour tags et metadata, nous devons les convertir en JSON
        # current["tags"] et current["metadata"] sont déjà des objets Python (parsés par _row_to_dict)
        if tags is not None:
            new_tags = json.dumps(tags)
        else:
            # Reconvertir les tags actuels en JSON
            new_tags = json.dumps(current["tags"]) if current.get("tags") else None
        
        if metadata is not None:
            new_metadata = json.dumps(metadata)
        else:
            # Reconvertir les metadata actuels en JSON
            new_metadata = json.dumps(current["metadata"]) if current.get("metadata") else None
        
        # Mettre à jour dans SQLite
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("""
                UPDATE knowledge
                SET title = ?, content = ?, category = ?, source = ?, 
                    tags = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_title, new_content, new_category, new_source, 
                  new_tags, new_metadata, knowledge_id))
        
        # Mettre à jour l'embedding dans Chroma si le contenu a changé
        if content is not None and self.knowledge_collection:
            try:
                self.knowledge_collection.update(
                    documents=[new_content],
                    metadatas=[{
                        "title": new_title,
                        "category": new_category or "",
                        "source": new_source or "",
                        "knowledge_id": knowledge_id
                    }],
                    ids=[f"knowledge_{knowledge_id}"]
                )
            except Exception as e:
                print(f"⚠️  Erreur lors de la mise à jour de l'embedding: {e}")
        
        # Invalider le cache
        if self.redis_client:
            self.redis_client.delete(f"knowledge:{knowledge_id}")
            self.redis_client.delete("knowledge:all")
        
        return True
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """
        Supprime une connaissance
        
        Args:
            knowledge_id: ID de la connaissance à supprimer
            
        Returns:
            True si suppression réussie, False sinon
        """
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
            deleted = cursor.rowcount > 0
        
        # Supprimer de Chroma
        if deleted and self.knowledge_collection:
            try:
                self.knowledge_collection.delete(ids=[f"knowledge_{knowledge_id}"])
            except Exception as e:
                print(f"⚠️  Erreur lors de la suppression de l'embedding: {e}")
        
        # Invalider le cache
        if self.redis_client:
            self.redis_client.delete(f"knowledge:{knowledge_id}")
            self.redis_client.delete("knowledge:all")
        
        return deleted
    
    def search_knowledge(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
        semantic: bool = True
    ) -> List[Dict]:
        """
        Recherche des connaissances
        
        Args:
            query: Texte de recherche (optionnel)
            category: Filtrer par catégorie (optionnel)
            limit: Nombre max de résultats
            semantic: Utiliser la recherche sémantique si disponible
            
        Returns:
            Liste de connaissances correspondantes
        """
        # Recherche sémantique avec Chroma si disponible et demandée
        if semantic and query and self.knowledge_collection:
            try:
                results = self.knowledge_collection.query(
                    query_texts=[query],
                    n_results=limit,
                    where={"category": category} if category else None
                )
                
                # Récupérer les détails complets depuis SQLite
                knowledge_ids = [
                    meta["knowledge_id"] 
                    for meta in results["metadatas"][0]
                ]
                
                detailed_results = []
                for kid in knowledge_ids:
                    k = self.get_knowledge(kid)
                    if k:
                        detailed_results.append(k)
                
                return detailed_results
                
            except Exception as e:
                print(f"⚠️  Erreur lors de la recherche sémantique: {e}")
                # Fallback sur recherche SQL
        
        # Recherche SQL classique
        sql = "SELECT * FROM knowledge WHERE 1=1"
        params = []
        
        if query:
            sql += " AND (title LIKE ? OR content LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
        return [d for d in [self._row_to_dict(row) for row in rows] if d is not None]
    
    def get_all_knowledge(self, limit: int = 100) -> List[Dict]:
        """
        Récupère toutes les connaissances
        
        Args:
            limit: Nombre max de résultats
            
        Returns:
            Liste de toutes les connaissances
        """
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM knowledge 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
        return [d for d in [self._row_to_dict(row) for row in rows] if d is not None]
    
    # =====================
    # GESTION DES CONVERSATIONS
    # =====================
    
    def save_conversation(
        self,
        user_input: str,
        nety_response: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Sauvegarde une conversation
        
        Args:
            user_input: Message de l'utilisateur
            nety_response: Réponse de NETY
            context: Contexte utilisé (optionnel)
            session_id: ID de session (optionnel)
            metadata: Métadonnées (optionnel)
            
        Returns:
            ID de la conversation
        """
        metadata_str = json.dumps(metadata) if metadata else None
        
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations 
                (user_input, nety_response, context, session_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (user_input, nety_response, context, session_id, metadata_str))
            
            conversation_id = cursor.lastrowid or 0
        
        # Ajouter l'embedding de la conversation dans Chroma
        if self.conversations_collection:
            try:
                combined_text = f"{user_input} {nety_response}"
                self.conversations_collection.add(
                    documents=[combined_text],
                    metadatas=[{
                        "user_input": user_input[:100],
                        "session_id": session_id or "",
                        "conversation_id": conversation_id
                    }],
                    ids=[f"conversation_{conversation_id}"]
                )
            except Exception as e:
                print(f"⚠️  Erreur lors de l'ajout de l'embedding conversation: {e}")
        
        return conversation_id
    
    def get_conversation_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Récupère l'historique des conversations
        
        Args:
            session_id: Filtrer par session (optionnel)
            limit: Nombre max de résultats
            
        Returns:
            Liste des conversations
        """
        sql = "SELECT * FROM conversations"
        params = []
        
        if session_id:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with DatabaseConnector.sqlite_cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
        
        return [d for d in [self._row_to_dict(row) for row in rows] if d is not None]
    
    # =====================
    # UTILITAIRES
    # =====================
    
    def _row_to_dict(self, row) -> Optional[Dict]:
        """Convertit une Row SQLite en dictionnaire"""
        if row is None:
            return None
        
        data = dict(row)
        
        # Parser les champs JSON
        if "tags" in data and data["tags"]:
            try:
                data["tags"] = json.loads(data["tags"])
            except json.JSONDecodeError:
                pass
        
        if "metadata" in data and data["metadata"]:
            try:
                data["metadata"] = json.loads(data["metadata"])
            except json.JSONDecodeError:
                pass
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la base de connaissances
        
        Returns:
            Dict avec diverses statistiques
        """
        with DatabaseConnector.sqlite_cursor() as cursor:
            # Nombre de connaissances
            cursor.execute("SELECT COUNT(*) as count FROM knowledge")
            knowledge_count = cursor.fetchone()["count"]
            
            # Nombre de conversations
            cursor.execute("SELECT COUNT(*) as count FROM conversations")
            conversations_count = cursor.fetchone()["count"]
            
            # Catégories
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM knowledge 
                WHERE category IS NOT NULL
                GROUP BY category
            """)
            categories = {row["category"]: row["count"] for row in cursor.fetchall()}
        
        stats = {
            "knowledge_count": knowledge_count,
            "conversations_count": conversations_count,
            "categories": categories,
            "chroma_available": self.knowledge_collection is not None,
            "redis_available": self.redis_client is not None
        }
        
        return stats

    def search(self, message: str, intent: dict) -> dict:
        """Search knowledge base for relevant information"""
        # Implement search logic here
        return {}
    Gère la base de connaissances de NETY
    Recherche et récupère les informations pertinentes
    """
    
    def __init__(self) -> None:
        # Base de connaissances simple (à étendre avec une vraie base de données)
        self.knowledge_base: Dict[str, Dict[str, str]] = {
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
    
    def add_knowledge(self, category: str, key: str, value: str) -> None:
        """
        Ajoute une nouvelle connaissance à la base
        
        Args:
            category: Catégorie de la connaissance
            key: Clé de la connaissance
            value: Valeur de la connaissance
        """
        if category not in self.knowledge_base:
            self.knowledge_base[category] = {}
        
        self.knowledge_base[category][key] = value
