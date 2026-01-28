"""
Tests unitaires pour le module Brain de NETY
"""

import pytest
import sys
sys.path.insert(0, '/home/runner/work/NETY/NETY')

from nety.core.brain import Brain


class TestBrain:
    """Tests pour la classe Brain"""
    
    def test_brain_initialization(self):
        """Test de l'initialisation du Brain"""
        brain = Brain()
        assert brain is not None
        assert brain.limbic_filter is not None
        assert brain.memory is not None
        assert brain.knowledge is not None
        assert brain.intent_analyzer is not None
        assert brain.response_generator is not None
    
    def test_brain_think_method(self):
        """Test de la méthode think()"""
        brain = Brain()
        result = brain.think("Test message")
        assert result is not None
        assert isinstance(result, str)
    
    def test_brain_get_modules_status(self):
        """Test de get_modules_status()"""
        brain = Brain()
        status = brain.get_modules_status()
        assert isinstance(status, dict)
        assert len(status) > 0
        assert "cortex_limbic" in status
        assert "memory" in status
        assert "knowledge_base" in status
        assert "intent_analyzer" in status
    
    def test_brain_get_context(self):
        """Test de get_context()"""
        brain = Brain()
        
        # Le contexte doit être vide au départ
        context = brain.get_context()
        assert isinstance(context, list)
        assert len(context) == 0
        
        # Après un think, le contexte doit contenir une interaction
        brain.think("Hello")
        context = brain.get_context()
        assert len(context) == 1
        assert "input" in context[0]
        assert "output" in context[0]
    
    def test_brain_context_history_limit(self):
        """Test de la limite de l'historique du contexte"""
        brain = Brain()
        
        # Ajouter plus de 100 interactions
        for i in range(105):
            brain.think(f"Message {i}")
        
        context = brain.get_context()
        # L'historique doit être limité à 100
        assert len(context) == 100
    
    def test_brain_process_message_pipeline(self):
        """Test du pipeline complet de traitement"""
        brain = Brain()
        
        test_messages = [
            "Bonjour NETY",
            "Comment vas-tu ?",
            "Quelle est ta mission ?"
        ]
        
        for message in test_messages:
            result = brain.think(message)
            assert result is not None
            assert isinstance(result, str)
        
        # Vérifier que toutes les interactions sont enregistrées
        context = brain.get_context()
        assert len(context) == 3


class TestIntentAnalyzer:
    """Tests pour la classe IntentAnalyzer"""
    
    def test_intent_analyzer_initialization(self):
        """Test de l'initialisation de IntentAnalyzer"""
        from nety.core.intent_analyzer import IntentAnalyzer
        analyzer = IntentAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'intent_categories')
    
    def test_intent_analyzer_analyze_question(self):
        """Test de détection d'une question"""
        from nety.core.intent_analyzer import IntentAnalyzer
        analyzer = IntentAnalyzer()
        
        result = analyzer.analyze("Comment ça va ?")
        assert result['type'] == 'question'
        assert result['confidence'] > 0.5
    
    def test_intent_analyzer_analyze_greeting(self):
        """Test de détection d'une salutation"""
        from nety.core.intent_analyzer import IntentAnalyzer
        analyzer = IntentAnalyzer()
        
        result = analyzer.analyze("Bonjour NETY")
        assert result['type'] == 'greeting'
        assert result['confidence'] > 0.5


class TestKnowledgeManager:
    """Tests pour la classe KnowledgeManager"""
    
    def test_knowledge_manager_initialization(self):
        """Test de l'initialisation de KnowledgeManager"""
        from nety.knowledge_base.knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        assert km is not None
        assert hasattr(km, 'knowledge_base')
        assert isinstance(km.knowledge_base, dict)
    
    def test_knowledge_manager_search(self):
        """Test de la recherche dans la base de connaissances"""
        from nety.knowledge_base.knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        
        intent = {"type": "question", "confidence": 0.8}
        result = km.search("Qui es-tu ?", intent)
        
        assert isinstance(result, dict)
        assert "results" in result
        assert "relevance" in result
    
    def test_knowledge_manager_add_knowledge(self):
        """Test de l'ajout de connaissances"""
        from nety.knowledge_base.knowledge_manager import KnowledgeManager
        km = KnowledgeManager()
        
        km.add_knowledge("test_category", "test_key", "test_value")
        assert "test_category" in km.knowledge_base
        assert km.knowledge_base["test_category"]["test_key"] == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
