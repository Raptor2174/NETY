"""
Configuration des bases de données pour NETY Knowledge Base
"""
import os
from pathlib import Path


class DatabaseConfig:
    """Configuration centralisée pour toutes les bases de données"""
    
    # Répertoire de base pour les données
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    DB_DIR = DATA_DIR / "databases"
    
    # SQLite Configuration
    SQLITE_DB_PATH = DB_DIR / "nety_knowledge.db"
    
    # Chroma DB Configuration
    CHROMA_PERSIST_DIR = DB_DIR / "chroma"
    CHROMA_COLLECTION_KNOWLEDGE = "nety_knowledge"
    CHROMA_COLLECTION_CONVERSATIONS = "conversations"
    
    # Redis Configuration (optionnel)
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    
    # Cache TTL (Time To Live) en secondes
    CACHE_TTL_SHORT = 300      # 5 minutes
    CACHE_TTL_MEDIUM = 1800    # 30 minutes
    CACHE_TTL_LONG = 3600      # 1 heure
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION = 384
    
    @classmethod
    def ensure_directories(cls):
        """Crée les répertoires nécessaires s'ils n'existent pas"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.DB_DIR.mkdir(exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_sqlite_path(cls) -> str:
        """Retourne le chemin complet de la base SQLite"""
        cls.ensure_directories()
        return str(cls.SQLITE_DB_PATH)
    
    @classmethod
    def get_chroma_path(cls) -> str:
        """Retourne le chemin du répertoire de persistance Chroma"""
        cls.ensure_directories()
        return str(cls.CHROMA_PERSIST_DIR)
