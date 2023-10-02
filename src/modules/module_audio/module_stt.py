import torch
import torch.nn as nn

class ModeleSTT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModeleSTT, self).__init__()
        
        # Couches récurrentes (LSTM) pour la séquence audio
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        
        # Couche de sortie pour la prédiction des caractères
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Propagation avant à travers les couches LSTM
        lstm_out, _ = self.lstm(x)
        
        # Utilisation de la dernière sortie de la séquence pour la prédiction
        predictions = self.output_layer(lstm_out[:, -1, :])
        return predictions

# Créez une instance de votre modèle STT avec des paramètres spécifiques
modele_stt = ModeleSTT(input_size=..., output_size=...)

print(modele_stt)

# Vous pouvez également charger un modèle pré-entraîné ici si nécessaire