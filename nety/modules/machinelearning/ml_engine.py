import torch
import torch.nn as nn

class MLEngine:
    def __init__(self, model=None):
        """
        Initialise le moteur ML
        
        Args:
            model: Modèle PyTorch (nn.Module) ou None pour un modèle par défaut
        """
        if model is None:
            # Créer un modèle simple par défaut pour la V1
            self.model = self._create_default_model()
        else:
            self.model = model
        
        print("✓ ML Engine initialisé")

    def _create_default_model(self):
        """Crée un modèle simple pour la V1 (sera remplacé plus tard)"""
        # Pour l'instant, un modèle factice pour ne pas crasher
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.fc(x)
        
        return DummyModel()

    # ==========================================
    # 🎯 MÉTHODES APPELÉES PAR BRAIN
    # ==========================================
    def transform_text(self, text: str) -> str:
        """
        Transforme/réécrit un texte
        Pour V1 : implémentation simple, sera améliorée avec ML
        """
        print(f"🔄 ML Engine transforme : {text}")
        
        # Pour l'instant : transformation simple (V1)
        # Tu pourras ajouter le vrai ML plus tard
        transformed = text.upper()  # Exemple simple
        return f"[Transformé] {transformed}"

    def generate_response(self, text: str) -> str:
        """
        Génère une réponse conversationnelle
        Pour V1 : réponses basiques, sera améliorée avec ML
        """
        print(f"💬 ML Engine génère une réponse pour : {text}")
        
        # Pour l'instant : réponses prédéfinies (V1)
        # Tu pourras ajouter un vrai modèle de langage plus tard
        responses = {
            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "salut": "Salut ! Que puis-je faire pour toi ?",
            "comment ça va": "Je vais bien, merci ! Et toi ?"
        }
        
        text_lower = text.lower()
        for keyword, response in responses.items():
            if keyword in text_lower:
                return response
        
        return f"Je comprends que tu dis : '{text}'. Comment puis-je t'aider ?"

    # ==========================================
    # 🧠 MÉTHODES ML ORIGINALES
    # ==========================================
    def train(self, data, labels, epochs=10, learning_rate=0.01):
        """Entraîne le modèle"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def evaluate(self, data, labels):
        """Évalue le modèle"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy
    
    def predict(self, data):
        """Fait une prédiction brute"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted