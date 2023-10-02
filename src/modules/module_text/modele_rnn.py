
import torch
import torch.nn as nn

# Définir une classe pour le modèle
class MonReseauDeNeurones(nn.Module):
    def __init__(self):
        super(MonReseauDeNeurones, self).__init__()
        
        # Couche d'entrée
        self.input_layer = nn.Linear(in_features=10, out_features=64)  # Exemple : 10 entrées, 64 sorties
        
        # Couches cachées
        self.hidden_layer1 = nn.Linear(in_features=64, out_features=128)
        self.hidden_layer2 = nn.Linear(in_features=128, out_features=64)
        self.hidden_layer3 = nn.Linear(in_features=64, out_features=32)
        
        # Couche de sortie
        self.output_layer = nn.Linear(in_features=32, out_features=1)  # Exemple : Régression vers une valeur
        
        # Fonction d'activation (par exemple, ReLU)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Propagation avant
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer1(x))
        x = self.activation(self.hidden_layer2(x))
        x = self.activation(self.hidden_layer3(x))
        x = self.output_layer(x)
        return x

# Créer une instance du modèle
modele = MonReseauDeNeurones()

# Afficher la structure du modèle
print(modele)
