import torch
import torch.nn as nn

class ModeleCNN(nn.Module):
    def __init__(self):
        super(ModeleCNN, self).__init__()
        
        # Couche de convolution 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        
        # Couche de convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # Couche de convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        
        # Couche de convolution 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        
        # Couche de convolution 5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        
        # Couche de sortie
        self.fc = nn.Linear(in_features=256, out_features=num_classes)
        
    def forward(self, x):
        # Propagation avant
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        
        # Aplatir les données pour la couche de sortie
        x = x.view(x.size(0), -1)
        
        # Couche de sortie
        x = self.fc(x)
        return x

# Exemple d'utilisation :
# Assurez-vous d'ajuster le nombre de classes (num_classes) en fonction de votre tâche.
num_classes = 10  # Par exemple, pour la classification d'images en 10 classes.
mon_cnn = ModeleCNN()
