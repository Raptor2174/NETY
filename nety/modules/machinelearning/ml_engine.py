import torch
import torch.nn as nn

class MLEngine:
    def __init__(self, model):
        self.model = model

    def train(self, data, labels, epochs=10, learning_rate=0.01):
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
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        return accuracy
    
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted
 