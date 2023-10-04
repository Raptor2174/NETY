from src.modules.module_text.modele_rnn import ModeleRNN
from src.modules.module_image.modele_cnn import ModeleCNN
from src.modules.module_audio.module_stt import ModeleSTT
from src.modules.chatbox import generer_reponse

def main():
    # Créez des instances des modèles de texte, d'image et d'audio
    modele_texte = ModeleRNN(input_size=..., hidden_size=..., output_size=...)
    modele_image = ModeleCNN(...)
    modele_audio = ModeleSTT(...)

    # Boucle principale
    while True:
        # Lisez l'entrée de l'utilisateur
        entree_utilisateur = input("Vous: ")

        # Utilisez le chatbox pour générer une réponse
        reponse = generer_reponse(entree_utilisateur, modele_texte, modele_image, modele_audio)

        # Affichez la réponse
        print("NETY: ", reponse)

if __name__ == "__main__":
    main()
    