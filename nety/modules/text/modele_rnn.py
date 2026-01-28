
"""
Module de réseau de neurones récurrent (RNN) pour le traitement de texte.

Ce module implémente un LSTM pour le traitement séquentiel de données textuelles.

Fichier reutiliser depuis l'ancienne version de NETY. ! attention aux compatibilités !
"""

import torch
import torch.nn as nn



class ModeleRNN(nn.Module):
    """
    Réseau de neurones récurrent (LSTM) pour le traitement de séquences.
    
    Architecture:
        - Couches LSTM empilées
        - Couches entièrement connectées avec activation ReLU
        - Couche de sortie pour la prédiction
    
    Args:
        input_size: Dimension des features d'entrée
        hidden_size: Dimension de l'état caché du LSTM
        output_size: Dimension de la sortie
        num_layers: Nombre de couches LSTM (défaut: 2)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2) -> None:
        super(ModeleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couche RNN (LSTM)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Couches entièrement connectées
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        # Fonction d'activation
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant à travers le réseau LSTM.
        
        Args:
            x: Tensor d'entrée de forme (batch_size, sequence_length, input_size)
            
        Returns:
            Tensor de sortie de forme (batch_size, output_size)
        """
        # Initialiser les états cachés
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Propagation avant à travers LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Prendre la sortie de la dernière étape temporelle
        out = out[:, -1, :]
        
        # Couches entièrement connectées
        out = self.activation(self.fc1(out))
        out = self.fc2(out)

        return out


# Créer une instance du modèle
modele_rnn = ModeleRNN(input_size=10, hidden_size=64, output_size=1)

# Afficher la structure du modèle
print(modele_rnn)