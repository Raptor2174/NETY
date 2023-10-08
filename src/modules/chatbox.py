from src.modules.module_text.modele_rnn import ModeleRNN
from src.modules.module_image.modele_cnn import ModeleCNN
from src.modules.module_audio.module_stt import ModeleSTT

from PIL import Image

from postprocessing.text_postprocessing import postprocess_output

from preprocessing.text_preprocessing import preprocess_input
from preprocessing.image_preprocessing import preprocess_image
from preprocessing.image_preprocessing import preprocess_input
from preprocessing.audio_preprocessing import preprocess_audio
import torchaudio

def generer_reponse(entree_utilisateur, modele_texte, modele_image, modele_audio):
    # Prétraitez l'entrée utilisateur si nécessaire
    donnees_pretraitees = preprocess_input(entree_utilisateur)

    # Utilisez le modèle de traitement de texte pour les prédictions
    predictions_texte = modele_texte(donnees_pretraitees)

    # Utilisez le modèle de vision par ordinateur pour les prédictions d'images (si applicable)
    image = Image.open("data/images/image1.jpg")
    image = preprocess_image(image)
    predictions_images = modele_image(image)

    # Utilisez le modèle de reconnaissance vocale pour la conversion audio en texte (si applicable)
    waveform, sample_rate = torchaudio.load("data/audio/audio_data.wav") 
    waveform = preprocess_audio(waveform)
    predictions_audio = modele_audio(waveform)

    # Post-traitez les prédictions et générez une réponse
    reponse_generale = postprocess_output(predictions_texte, predictions_images, predictions_audio)

    return reponse_generale
