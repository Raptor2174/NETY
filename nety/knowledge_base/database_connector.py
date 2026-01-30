"""
Gestionnaire de connexions aux bases de données
"""
import sqlite3
from typing import Optional
from contextlib import contextmanager

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import redis
    from redis import ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisConnectionError = None

from .database_config import DatabaseConfig


class DatabaseConnector:
    """Gestionnaire centralisé des connexions aux bases de données"""
    
    _sqlite_connection = None
    _chroma_client = None
    _redis_client = None
    
    @classmethod
    def get_sqlite_connection(cls) -> sqlite3.Connection:
        """
        Retourne une connexion SQLite (singleton)
        
        NOTE: check_same_thread=False est utilisé car NETY utilise un modèle
        single-threaded pour le Brain. Pour une utilisation multi-thread,
        utilisez un pool de connexions à la place.
        """
        if cls._sqlite_connection is None:
            db_path = DatabaseConfig.get_sqlite_path()
            cls._sqlite_connection = sqlite3.connect(
                db_path,
                check_same_thread=False  # Safe pour le modèle single-thread de NETY
            )
            cls._sqlite_connection.row_factory = sqlite3.Row
            
            # Activer les foreign keys
            cls._sqlite_connection.execute("PRAGMA foreign_keys = ON")
            
        return cls._sqlite_connection
    
    @classmethod
    @contextmanager
    def sqlite_cursor(cls):
        """
        Context manager pour obtenir un curseur SQLite
        Usage:
            with DatabaseConnector.sqlite_cursor() as cursor:
                cursor.execute(...)
        """
        conn = cls.get_sqlite_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
    
    @classmethod
    def get_chroma_client(cls) -> Optional[object]:
        """
        Retourne un client Chroma DB (singleton)
        Retourne None si Chroma n'est pas disponible
        """
        if not CHROMA_AVAILABLE:
            return None
            
        if cls._chroma_client is None:
            persist_dir = DatabaseConfig.get_chroma_path()
            cls._chroma_client = chromadb.PersistentClient(
                path=persist_dir
            )
            
        return cls._chroma_client
    
    @classmethod
    def get_redis_client(cls) -> Optional[object]:
        """
        Retourne un client Redis (singleton)
        Retourne None si Redis n'est pas disponible ou désactivé
        """
        if not REDIS_AVAILABLE or not DatabaseConfig.REDIS_ENABLED:
            return None
            
        if cls._redis_client is None:
            cls._redis_client = redis.Redis(
                host=DatabaseConfig.REDIS_HOST,
                port=DatabaseConfig.REDIS_PORT,
                db=DatabaseConfig.REDIS_DB,
                password=DatabaseConfig.REDIS_PASSWORD,
                decode_responses=True
            )
            
            # Test de connexion
            try:
                cls._redis_client.ping()
            except Exception:
                print("⚠️  Redis non disponible, fonctionnement sans cache")
                cls._redis_client = None
                
        return cls._redis_client
    
    @classmethod
    def close_all(cls):
        """Ferme toutes les connexions"""
        if cls._sqlite_connection:
            cls._sqlite_connection.close()
            cls._sqlite_connection = None
            
        if cls._redis_client:
            try:
                cls._redis_client.close()
            except Exception:
                pass
            cls._redis_client = None
            
        # Chroma DB se ferme via garbage collection
        cls._chroma_client = None
