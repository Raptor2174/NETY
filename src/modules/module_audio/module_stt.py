"""
Module de traitement audio et Speech-to-Text (STT).

Ce module fournit des fonctionnalités pour charger, prétraiter et traiter des fichiers audio
en utilisant des modèles de deep learning pour la reconnaissance vocale.
"""

# module_stt.py

import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from typing import Tuple, Any


def load_audio(file_path: str, duration: int = 10, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Charge un fichier audio.
    
    Args:
        file_path: Chemin vers le fichier audio
        duration: Durée maximale en secondes (défaut: 10)
        sr: Taux d'échantillonnage (défaut: 22050 Hz)
        
    Returns:
        Tuple contenant:
            - audio: Tableau numpy des échantillons audio
            - sr: Taux d'échantillonnage utilisé
    """
    audio, sr = librosa.load(file_path, duration=duration, sr=sr)
    return audio, sr


def preprocessing(audio: np.ndarray) -> np.ndarray:
    """
    Prétraite les données audio.
    
    Args:
        audio: Tableau numpy des échantillons audio bruts
        
    Returns:
        Données audio prétraitées
    """
    # Importez ici le module de prétraitement
    from src.modules.preprocessing.image_preprocessing import preprocess_audio
    preprocessed_audio = preprocess_audio(audio)
    return preprocessed_audio


def postprocessing(predictions: np.ndarray) -> Any:
    """
    Post-traite les prédictions du modèle.
    
    Args:
        predictions: Prédictions brutes du modèle
        
    Returns:
        Prédictions post-traitées
    """
    # Importez ici le module de post-traitement
    from src.modules.postprocessing.image_postprocessing import postprocess_predictions
    postprocessed_predictions = postprocess_predictions(predictions)
    return postprocessed_predictions


def create_audio_processing_model(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """
    Crée un modèle CNN pour le traitement audio.
    
    Architecture:
        - 2 blocs de convolution 2D avec max pooling
        - Couche de flatten
        - Couche dense cachée avec 128 neurones
        - Couche de sortie avec activation softmax
    
    Args:
        input_shape: Forme des données d'entrée (height, width, channels)
        num_classes: Nombre de classes pour la classification
        
    Returns:
        Modèle Keras Sequential compilé
    """
    model = Sequential()
    
    # Couche de convolution 2D
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Couche de convolution 2D
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Couche d'aplatissement
    model.add(Flatten())
    
    # Couche entièrement connectée
    model.add(Dense(128, activation='relu'))
    
    # Couche de sortie
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de configuration
    file_path = "data/audio/audio_data.wav"  # Chemin vers le fichier audio
    
    # Paramètres du modèle (à adapter selon vos besoins)
    audio_height = 128  # Hauteur du spectrogramme
    audio_width = 128   # Largeur du spectrogramme
    audio_channels = 1  # Nombre de canaux (1 pour mono)
    num_classes = 10    # Nombre de classes à classifier
    
    # Charger et prétraiter l'audio
    audio, sr = load_audio(file_path)
    preprocessed_audio = preprocessing(audio)

    # Créer le modèle
    input_shape = (audio_height, audio_width, audio_channels)
    model = create_audio_processing_model(input_shape, num_classes)
    
    # Exécutez ici les prédictions avec le modèle sur preprocessed_audio
    predictions = model.predict(preprocessed_audio)

    # Post-traiter les prédictions
    postprocessed_predictions = postprocessing(predictions)
    
    print(postprocessed_predictions)
