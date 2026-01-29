"""
Tests pour le modèle RNN standalone
"""
import pytest
import torch
from nety.modules.text.modele_rnn import ModeleRNN

class TestModeleRNN:
    """Tests du modèle RNN"""
    
    def test_model_creation(self):
        """Test que le modèle se crée sans erreur"""
        model = ModeleRNN(input_size=10, hidden_size=64, output_size=1)
        assert model is not None
        assert model.hidden_size == 64
        assert model.num_layers == 2
    
    def test_forward_pass(self):
        """Test qu'une propagation avant fonctionne"""
        model = ModeleRNN(input_size=10, hidden_size=64, output_size=1)
        
        # Créer des données de test (batch_size=2, seq_length=5, input_size=10)
        x = torch.randn(2, 5, 10)
        
        # Forward pass
        output = model(x)
        
        # Vérifier la forme de sortie
        assert output.shape == (2, 1)  # (batch_size, output_size)
    
    def test_different_batch_sizes(self):
        """Test avec différentes tailles de batch"""
        model = ModeleRNN(input_size=10, hidden_size=64, output_size=1)
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 5, 10)
            output = model(x)
            assert output.shape == (batch_size, 1)
    
    def test_different_sequence_lengths(self):
        """Test avec différentes longueurs de séquence"""
        model = ModeleRNN(input_size=10, hidden_size=64, output_size=1)
        
        for seq_length in [5, 10, 20]:
            x = torch.randn(2, seq_length, 10)
            output = model(x)
            assert output.shape == (2, 1)
    
    def test_gradient_flow(self):
        """Test que les gradients se propagent correctement"""
        model = ModeleRNN(input_size=10, hidden_size=64, output_size=1)
        
        x = torch.randn(2, 5, 10, requires_grad=True)
        output = model(x)
        
        # Backprop
        loss = output.sum()
        loss.backward()
        
        # Vérifier que les gradients existent
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None


class TestTokenizer:
    """Tests du tokenizer"""
    
    def test_tokenizer_creation(self):
        """Test la création du tokenizer"""
        from nety.modules.text.tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer(vocab_size=100)
        assert tokenizer is not None
        assert tokenizer.vocab_size == 100
    
    def test_fit_and_encode(self):
        """Test fit et encode"""
        from nety.modules.text.tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer(vocab_size=100)
        
        # Entraîner sur quelques phrases
        texts = [
            "bonjour NETY",
            "comment vas tu",
            "je vais bien merci"
        ]
        tokenizer.fit(texts)
        
        # Encoder une phrase
        encoded = tokenizer.encode("bonjour NETY", max_length=10)
        
        assert encoded.shape == (10,)
        assert encoded.dtype == torch.long
    
    def test_encode_decode(self):
        """Test que encode puis decode retourne quelque chose de similaire"""
        from nety.modules.text.tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer(vocab_size=100)
        
        texts = ["bonjour", "hello world"]
        tokenizer.fit(texts)
        
        original = "bonjour"
        encoded = tokenizer.encode(original, max_length=5)
        decoded = tokenizer.decode(encoded)
        
        assert "bonjour" in decoded
