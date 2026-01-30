"""
Tests pour la base de connaissances NETY
"""
import pytest
import os
import tempfile
import time
import gc
from pathlib import Path

from nety.knowledge_base import (
    KnowledgeManager,
    SearchEngine,
    DatabaseInitializer,
    DatabaseConfig,
    DatabaseConnector
)


@pytest.fixture
def temp_db_dir():
    """Crée un répertoire temporaire pour les tests"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Override database paths for testing
        original_db_dir = DatabaseConfig.DB_DIR
        original_sqlite_path = DatabaseConfig.SQLITE_DB_PATH
        original_chroma_dir = DatabaseConfig.CHROMA_PERSIST_DIR
        
        DatabaseConfig.DB_DIR = Path(tmpdir)
        DatabaseConfig.SQLITE_DB_PATH = Path(tmpdir) / "test_nety.db"
        DatabaseConfig.CHROMA_PERSIST_DIR = Path(tmpdir) / "chroma"
        
        yield tmpdir
        
        # Restore original paths
        DatabaseConfig.DB_DIR = original_db_dir
        DatabaseConfig.SQLITE_DB_PATH = original_sqlite_path
        DatabaseConfig.CHROMA_PERSIST_DIR = original_chroma_dir
        
        # Close all connections FIRST
        DatabaseConnector.close_all()
        
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(0.2)  # Small delay to ensure files are released on Windows


@pytest.fixture
def knowledge_manager(temp_db_dir):
    """Crée un KnowledgeManager pour les tests"""
    # Reset singletons before initializing
    DatabaseConnector._sqlite_connection = None
    DatabaseConnector._chroma_client = None
    DatabaseConnector._redis_client = None
    
    DatabaseInitializer.initialize_sqlite()
    # Chroma may not be available in test environment, that's ok
    try:
        DatabaseInitializer.initialize_chroma()
    except Exception:
        pass
    
    manager = KnowledgeManager()
    
    yield manager
    
    # Clean up after test
    DatabaseConnector.close_all()
    gc.collect()
    time.sleep(0.1)


class TestDatabaseConfig:
    """Tests de configuration"""
    
    def test_config_paths(self):
        """Test que les chemins sont bien définis"""
        assert DatabaseConfig.SQLITE_DB_PATH is not None
        assert DatabaseConfig.CHROMA_PERSIST_DIR is not None
        assert DatabaseConfig.EMBEDDING_MODEL is not None
    
    def test_ensure_directories(self, temp_db_dir):
        """Test la création des répertoires"""
        DatabaseConfig.ensure_directories()
        assert DatabaseConfig.DB_DIR.exists()


class TestDatabaseInitialization:
    """Tests d'initialisation"""
    
    def test_sqlite_initialization(self, temp_db_dir):
        """Test l'initialisation de SQLite"""
        DatabaseInitializer.initialize_sqlite()
        
        # Vérifier que le fichier DB existe
        assert DatabaseConfig.SQLITE_DB_PATH.exists()
        
        # Vérifier que les tables existent
        conn = DatabaseConnector.get_sqlite_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "knowledge" in tables
        assert "conversations" in tables
        assert "config" in tables
        assert "logs" in tables


class TestKnowledgeManager:
    """Tests du gestionnaire de connaissances"""
    
    def test_add_knowledge(self, knowledge_manager):
        """Test l'ajout d'une connaissance"""
        knowledge_id = knowledge_manager.add_knowledge(
            title="Test Knowledge",
            content="This is a test content",
            category="test",
            source="unittest",
            tags=["test", "demo"],
            metadata={"author": "test_user"}
        )
        
        assert knowledge_id > 0
    
    def test_get_knowledge(self, knowledge_manager):
        """Test la récupération d'une connaissance"""
        # Ajouter une connaissance
        knowledge_id = knowledge_manager.add_knowledge(
            title="Test Knowledge",
            content="This is a test content"
        )
        
        # Récupérer la connaissance
        knowledge = knowledge_manager.get_knowledge(knowledge_id)
        
        assert knowledge is not None
        assert knowledge["title"] == "Test Knowledge"
        assert knowledge["content"] == "This is a test content"
        assert knowledge["id"] == knowledge_id
    
    def test_update_knowledge(self, knowledge_manager):
        """Test la mise à jour d'une connaissance"""
        # Ajouter une connaissance
        knowledge_id = knowledge_manager.add_knowledge(
            title="Original Title",
            content="Original Content"
        )
        
        # Mettre à jour
        success = knowledge_manager.update_knowledge(
            knowledge_id,
            title="Updated Title",
            content="Updated Content"
        )
        
        assert success is True
        
        # Vérifier la mise à jour
        knowledge = knowledge_manager.get_knowledge(knowledge_id)
        assert knowledge["title"] == "Updated Title"
        assert knowledge["content"] == "Updated Content"
    
    def test_delete_knowledge(self, knowledge_manager):
        """Test la suppression d'une connaissance"""
        # Ajouter une connaissance
        knowledge_id = knowledge_manager.add_knowledge(
            title="To Delete",
            content="Will be deleted"
        )
        
        # Supprimer
        success = knowledge_manager.delete_knowledge(knowledge_id)
        assert success is True
        
        # Vérifier que la connaissance n'existe plus
        knowledge = knowledge_manager.get_knowledge(knowledge_id)
        assert knowledge is None
    
    def test_search_knowledge_simple(self, knowledge_manager):
        """Test la recherche simple de connaissances"""
        # Ajouter plusieurs connaissances
        knowledge_manager.add_knowledge(
            title="Python Programming",
            content="Python is a high-level programming language",
            category="programming"
        )
        knowledge_manager.add_knowledge(
            title="Machine Learning",
            content="Machine learning is a subset of AI",
            category="ai"
        )
        
        # Recherche par query
        results = knowledge_manager.search_knowledge(
            query="Python",
            semantic=False  # Désactiver la recherche sémantique pour test
        )
        
        assert len(results) >= 1
        assert any("Python" in r["title"] for r in results)
        
        # Recherche par catégorie
        results = knowledge_manager.search_knowledge(
            category="programming",
            semantic=False
        )
        
        assert len(results) >= 1
        assert all(r["category"] == "programming" for r in results)
    
    def test_get_all_knowledge(self, knowledge_manager):
        """Test la récupération de toutes les connaissances"""
        # Ajouter plusieurs connaissances
        for i in range(5):
            knowledge_manager.add_knowledge(
                title=f"Knowledge {i}",
                content=f"Content {i}"
            )
        
        # Récupérer toutes
        all_knowledge = knowledge_manager.get_all_knowledge()
        
        assert len(all_knowledge) >= 5
    
    def test_save_conversation(self, knowledge_manager):
        """Test la sauvegarde de conversations"""
        conversation_id = knowledge_manager.save_conversation(
            user_input="Bonjour NETY",
            nety_response="Bonjour! Comment puis-je vous aider?",
            session_id="test_session",
            metadata={"language": "fr"}
        )
        
        assert conversation_id > 0
    
    def test_get_conversation_history(self, knowledge_manager):
        """Test la récupération de l'historique"""
        # Ajouter des conversations
        knowledge_manager.save_conversation(
            user_input="Message 1",
            nety_response="Response 1",
            session_id="session_1"
        )
        knowledge_manager.save_conversation(
            user_input="Message 2",
            nety_response="Response 2",
            session_id="session_1"
        )
        
        # Récupérer l'historique
        history = knowledge_manager.get_conversation_history(
            session_id="session_1"
        )
        
        assert len(history) >= 2
        assert all(h["session_id"] == "session_1" for h in history)
    
    def test_get_stats(self, knowledge_manager):
        """Test les statistiques"""
        # Ajouter des données
        knowledge_manager.add_knowledge(
            title="Test",
            content="Test content",
            category="test"
        )
        knowledge_manager.save_conversation(
            user_input="Test",
            nety_response="Test"
        )
        
        # Récupérer les stats
        stats = knowledge_manager.get_stats()
        
        assert "knowledge_count" in stats
        assert "conversations_count" in stats
        assert "categories" in stats
        assert stats["knowledge_count"] >= 1
        assert stats["conversations_count"] >= 1


class TestSearchEngine:
    """Tests du moteur de recherche"""
    
    def test_search_basic(self, knowledge_manager):
        """Test la recherche basique"""
        # Ajouter des connaissances
        km = knowledge_manager
        km.add_knowledge(
            title="Deep Learning",
            content="Deep learning uses neural networks with multiple layers"
        )
        
        # Créer le moteur de recherche
        search_engine = SearchEngine()
        
        # Rechercher
        results = search_engine.search("neural networks", use_semantic=False)
        
        assert len(results) >= 1
    
    def test_get_context_for_query(self, knowledge_manager):
        """Test la récupération de contexte pour RAG"""
        # Ajouter des connaissances
        km = knowledge_manager
        km.add_knowledge(
            title="RNN",
            content="Recurrent Neural Networks are good for sequences"
        )
        km.add_knowledge(
            title="CNN",
            content="Convolutional Neural Networks are good for images"
        )
        
        # Créer le moteur de recherche
        search_engine = SearchEngine()
        
        # Récupérer le contexte
        context = search_engine.get_context_for_query(
            "neural networks",
            max_results=2
        )
        
        assert isinstance(context, str)
        # Context should contain some information
        assert len(context) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
