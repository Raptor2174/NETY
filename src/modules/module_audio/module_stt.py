import os
import torch
import torchaudio
import torch.nn as nn
from torchsummary import summary

# Définition du modèle STT
class ModeleSTT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModeleSTT, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.output_layer(lstm_out[:, -1, :])
        return predictions

# Fonction pour charger des données audio
def charger_audio(data\audio\audio_data.wav):
    waveform, sample_rate =torchaudio.load(charger_audio)
    return waveform, sample_rate

# Fonction pour prétraiter les données audio
def pretraiter_audio(waveform):
    # Appliquer un prétraitement audio si nécessaire (normalisation, etc.)
    return waveform

# Fonction pour effectuer la reconnaissance vocale
def reconnaissance_vocale(modele, waveform):
    with torch.no_grad():
        output = modele(waveform)
    return output

if __name__ == "__main__":
    # Exemple d'utilisation du module STT
    chemin_fichier_audio = "data/audio/audio_data.wav"
    waveform, sample_rate = charger_audio(data\audio\audio_data.wav)
    waveform = pretraiter_audio(waveform)
    
    # Initialisation et affichage de la structure du modèle STT
    modele_stt = ModeleSTT(input_dim=waveform.shape[1], hidden_dim=256, output_dim=26)
    summary(modele_stt, input_size=(waveform.shape[1], waveform.shape[2]))
    
    # Reconnaissance vocale
    predictions = reconnaissance_vocale(modele_stt, waveform)
    
    print("Résultats de la reconnaissance vocale :")
    print(predictions)
