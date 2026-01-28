"""
Module de réseau de neurones convolutionnel (CNN) pour le traitement d'images.

Ce module implémente un CNN à 5 couches de convolution pour la classification d'images.

Fichier reutiliser depuis l'ancienne version de NETY. ! attention aux compatibilités !
"""

import torch
import torch.nn as nn


class ModeleCNN(nn.Module):
    """
    Réseau de neurones convolutionnel pour la classification d'images.

    Architecture:
        - 5 couches de convolution avec activation ReLU et max pooling
        - Nombre croissant de filtres : 3 -> 16 -> 32 -> 64 -> 128 -> 256
        - Global Average Pooling pour réduire à une dimension fixe
        - Couche entièrement connectée finale pour la classification

    Args:
        num_classes: Nombre de classes pour la classification (défaut: 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super(ModeleCNN, self).__init__()

        # Couche de convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche de convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche de convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche de convolution 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche de convolution 5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        # Global Average Pooling pour avoir une taille fixe indépendante de l'entrée
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Couche de sortie
        self.fc = nn.Linear(in_features=256, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant à travers le réseau.

        Args:
            x: Tensor d'entrée de forme (batch_size, 3, height, width)

        Returns:
            Tensor de sortie de forme (batch_size, num_classes)
        """
        # Propagation avant avec pooling
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.relu5(self.conv5(x))

        # Global Average Pooling pour réduire à (batch_size, 256, 1, 1)
        x = self.global_avg_pool(x)

        # Aplatir les données pour la couche de sortie
        x = x.view(x.size(0), -1)

        # Couche de sortie
        x = self.fc(x)
        return x


# Exemple d'utilisation :
# Assurez-vous d'ajuster le nombre de classes (num_classes) en fonction de votre tâche.
num_classes = 10  # Par exemple, pour la classification d'images en 10 classes.
mon_cnn = ModeleCNN(num_classes=num_classes)
