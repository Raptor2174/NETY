import torchaudio
import torch

# Classe pour le prétraitement audio
class AudioPreprocessor:
    def __init__(self):
        # Spécifiez les transformations audio que vous souhaitez appliquer (par exemple, échantillonnage, normalisation, etc.)
        self.transforms = torchaudio.transforms.MelSpectrogram()

    def preprocess_input(self, audio_path):
        # Chargez l'audio depuis le chemin du fichier
        waveform, sample_rate = torchaudio.load(audio_path)
        # Appliquez les transformations définies
        preprocessed_audio = self.transforms(waveform)
        return preprocessed_audio

# Exemple d'utilisation
if __name__ == '__main__':
    # Créez une instance du préprocesseur
    preprocessor = AudioPreprocessor()

    # Supposons que vous ayez un chemin audio en entrée
    audio_path = 'audio.wav'

    # Prétraitez l'audio
    preprocessed_audio = preprocessor.preprocess_input(audio_path)

    # Affichez l'audio prétraité (pour le débogage)
    print(preprocessed_audio)
