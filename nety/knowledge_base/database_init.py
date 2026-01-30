"""
Schémas et initialisation des bases de données
"""
from typing import TYPE_CHECKING, Optional
from .database_connector import DatabaseConnector


if TYPE_CHECKING:
    import chromadb


class DatabaseSchema:
    """Définition des schémas SQL pour SQLite"""
    
    # Table des connaissances
    KNOWLEDGE_TABLE = """
    CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        category TEXT,
        source TEXT,
        tags TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT
    )
    """
    
    # Index pour les recherches fréquentes
    KNOWLEDGE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_created_at ON knowledge(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_title ON knowledge(title)"
    ]
    
    # Table des conversations
    CONVERSATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_input TEXT NOT NULL,
        nety_response TEXT,
        context TEXT,
        metadata TEXT,
        session_id TEXT
    )
    """
    
    # Index pour les conversations
    CONVERSATIONS_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)"
    ]
    
    # Table de configuration
    CONFIG_TABLE = """
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        description TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # Table des logs (optionnel)
    LOGS_TABLE = """
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        level TEXT NOT NULL,
        module TEXT,
        message TEXT NOT NULL,
        metadata TEXT
    )
    """
    
    LOGS_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)"
    ]


class DatabaseInitializer:
    """Initialise toutes les bases de données"""
    
    @staticmethod
    def initialize_sqlite():
        """Initialise la base SQLite avec les schémas"""
        print("📊 Initialisation de la base SQLite...")
        
        with DatabaseConnector.sqlite_cursor() as cursor:
            # Créer les tables
            cursor.execute(DatabaseSchema.KNOWLEDGE_TABLE)
            cursor.execute(DatabaseSchema.CONVERSATIONS_TABLE)
            cursor.execute(DatabaseSchema.CONFIG_TABLE)
            cursor.execute(DatabaseSchema.LOGS_TABLE)
            
            # Créer les indexes
            for index_sql in DatabaseSchema.KNOWLEDGE_INDEXES:
                cursor.execute(index_sql)
            
            for index_sql in DatabaseSchema.CONVERSATIONS_INDEXES:
                cursor.execute(index_sql)
                
            for index_sql in DatabaseSchema.LOGS_INDEXES:
                cursor.execute(index_sql)
        
        print("✅ Base SQLite initialisée")
    
    @staticmethod
    def initialize_chroma():
        """Initialise Chroma DB avec les collections"""
        client = DatabaseConnector.get_chroma_client()
        
        if client is None:
            print("⚠️  Chroma DB non disponible, fonctionnement en mode dégradé")
            return
        
        print("📊 Initialisation de Chroma DB...")
        
        from .database_config import DatabaseConfig
        
        # Créer ou récupérer les collections
        try:
            # Collection pour les connaissances
            knowledge_collection = client.get_or_create_collection(
                name=DatabaseConfig.CHROMA_COLLECTION_KNOWLEDGE,
                metadata={"description": "NETY knowledge base embeddings"}
            )
            
            # Collection pour les conversations
            conversations_collection = client.get_or_create_collection(
                name=DatabaseConfig.CHROMA_COLLECTION_CONVERSATIONS,
                metadata={"description": "Conversation history embeddings"}
            )
            
            print("✅ Chroma DB initialisée")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de Chroma: {e}")
    
    @staticmethod
    def initialize_redis():
        """Initialise Redis (vérification de connexion)"""
        from typing import Any
        try:
            import redis
            RedisType = redis.Redis
        except ImportError:
            RedisType = Any  # Fallback if redis is not installed

        client = DatabaseConnector.get_redis_client()
        
        if client is None:
            print("⚠️  Redis non disponible, fonctionnement sans cache")
            return
        
        print("📊 Initialisation de Redis...")
        
        try:
            # Test de connexion
            if hasattr(client, 'ping'):
                response = client.ping()
                if response:
                    print("📌 Redis connexion vérifiée")
            
            # Initialiser quelques clés de configuration
            version_exists = client.exists("nety:version")
            if not version_exists:
                client.set("nety:version", "1.0.0")
                print("📌 Version Redis configurée: 1.0.0")
            
            print("✅ Redis initialisé")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de Redis: {e}")
    
    @classmethod
    def initialize_all(cls):
        """Initialise toutes les bases de données"""
        print("=" * 60)
        print("🚀 Initialisation des bases de données NETY")
        print("=" * 60)
        
        # Toujours initialiser SQLite (obligatoire)
        cls.initialize_sqlite()
        
        # Initialiser Chroma (optionnel mais recommandé)
        cls.initialize_chroma()
        
        # Initialiser Redis (optionnel)
        cls.initialize_redis()
        
        print("=" * 60)
        print("✅ Initialisation terminée")
        print("=" * 60)


def init_databases():
    """Point d'entrée pour initialiser les databases"""
    DatabaseInitializer.initialize_all()


if __name__ == "__main__":
    # Permet d'exécuter directement ce script pour initialiser les DB
    init_databases()
