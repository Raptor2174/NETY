# module_sst.py

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_audio(file_path, duration=10, sr=22050):
    audio, sr = librosa.load(file_path, duration=duration, sr=sr)
    return audio, sr

def preprocessing(audio):
    # Importez ici le module de prétraitement
    from src.modules.preprocessing.image_preprocessing import preprocess_audio
    preprocessed_audio = preprocess_audio(audio)
    return preprocessed_audio

def postprocessing(predictions):
    # Importez ici le module de post-traitement
    from src.modules.postprocessing.image_postprocessing import postprocess_predictions
    postprocessed_predictions = postprocess_predictions(predictions)
    return postprocessed_predictions

def create_audio_processing_model(input_shape, num_classes):
    model = Sequential()
    
    # Couche de convolution 2D
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2))
    
    # Couche de convolution 2D
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))
    
    # Couche d'aplatissement
    model.add(Flatten())
    
    # Couche entièrement connectée
    model.add(Dense(128, activation='relu'))
    
    # Couche de sortie
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# Exemple d'utilisation
if __name__ == "__main__":
    file_path = "data/audio/audio_data.wav"  # Chemin vers le fichier audio
    audio, sr = load_audio(file_path)
    preprocessed_audio = preprocessing(audio)

    input_shape = (audio_height, audio_width, audio_channels)  # Spécifiez les dimensions d'entrée
    num_classes = number_of_classes  # Spécifiez le nombre de classes
    model = create_audio_processing_model(input_shape, num_classes)
    
    # Exécutez ici les prédictions avec le modèle sur preprocessed_audio
    predictions = model.predict(preprocessed_audio)

    postprocessed_predictions = postprocessing(predictions)
    
    print(postprocessed_predictions)
