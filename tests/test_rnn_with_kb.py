"""
Tests d'intégration RNN + Knowledge Base
"""
import pytest
import torch
import tempfile
from pathlib import Path

from nety.modules.text.modele_rnn import ModeleRNN
from nety.modules.text.tokenizer import SimpleTokenizer
from nety.knowledge_base import KnowledgeManager, SearchEngine, DatabaseInitializer, DatabaseConfig, DatabaseConnector


@pytest.fixture
def temp_kb():
    """Crée une base de connaissances temporaire pour les tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override paths
        original_db_dir = DatabaseConfig.DB_DIR
        original_sqlite = DatabaseConfig.SQLITE_DB_PATH
        original_chroma = DatabaseConfig.CHROMA_PERSIST_DIR
        
        DatabaseConfig.DB_DIR = Path(tmpdir)
        DatabaseConfig.SQLITE_DB_PATH = Path(tmpdir) / "test.db"
        DatabaseConfig.CHROMA_PERSIST_DIR = Path(tmpdir) / "chroma"
        
        # Initialize
        DatabaseInitializer.initialize_sqlite()
        
        km = KnowledgeManager()
        
        yield km
        
        # Restore
        DatabaseConfig.DB_DIR = original_db_dir
        DatabaseConfig.SQLITE_DB_PATH = original_sqlite
        DatabaseConfig.CHROMA_PERSIST_DIR = original_chroma
        DatabaseConnector.close_all()


class TestRNNWithKnowledgeBase:
    """Tests RNN + Knowledge Base"""
    
    def test_retrieve_context_and_process(self, temp_kb):
        """Test récupération de contexte et traitement par RNN"""
        
        # 1. Ajouter des connaissances
        temp_kb.add_knowledge(
            title="Chat",
            content="Un chat est un animal domestique avec quatre pattes",
            category="animaux"
        )
        
        temp_kb.add_knowledge(
            title="Chien", 
            content="Un chien est un animal fidèle et joueur",
            category="animaux"
        )
        
        # 2. Créer le moteur de recherche
        search = SearchEngine()
        
        # 3. Récupérer contexte (using category search)
        results = search.search(query="animal", category="animaux", limit=2, use_semantic=False)
        context = "\n\n".join([f"[{r['title']}]\n{r['content']}" for r in results]) if results else ""
        
        assert len(context) > 0
        assert "chat" in context.lower() or "chien" in context.lower()
        
        # 4. Préparer pour RNN
        tokenizer = SimpleTokenizer(vocab_size=1000)
        tokenizer.fit([context])
        
        encoded = tokenizer.encode(context[:100], max_length=20)
        
        # 5. Créer RNN adapté au vocabulaire
        model = ModeleRNN(input_size=1, hidden_size=32, output_size=10)
        
        # Reshape pour RNN (batch_size=1, seq_length=20, input_size=1)
        x = encoded.unsqueeze(0).unsqueeze(-1).float()
        
        # 6. Forward pass
        output = model(x)
        
        assert output.shape == (1, 10)
    
    def test_full_rag_pipeline(self, temp_kb):
        """Test pipeline RAG complet : Question -> Retrieve -> Process -> Answer"""
        
        # Setup KB
        temp_kb.add_knowledge(
            title="Python",
            content="Python est un langage de programmation interprété",
            category="programmation"
        )
        
        # Question utilisateur
        user_question = "qu'est-ce que python"
        
        # 1. Retrieve (using non-semantic search for testing)
        search = SearchEngine()
        results = search.search("python", limit=1, use_semantic=False)
        context = "\n\n".join([f"[{r['title']}]\n{r['content']}" for r in results]) if results else ""
        
        assert "python" in context.lower()
        
        # 2. Tokenize
        tokenizer = SimpleTokenizer(vocab_size=500)
        combined_text = f"{user_question} {context}"
        tokenizer.fit([combined_text])
        
        encoded = tokenizer.encode(combined_text[:50], max_length=15)
        
        # 3. Process with RNN
        model = ModeleRNN(input_size=1, hidden_size=16, output_size=5)
        x = encoded.unsqueeze(0).unsqueeze(-1).float()
        
        output = model(x)
        
        # 4. Verify output
        assert output.shape == (1, 5)
        
        # Simulation: prendre l'indice avec la plus haute valeur
        predicted_class = torch.argmax(output, dim=1)
        assert predicted_class.item() >= 0
