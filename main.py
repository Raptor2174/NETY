import tkinter as tk
from src.modules.module_text.modele_rnn import ModeleRNN
from src.modules.module_image.modele_cnn import ModeleCNN
from src.modules.module_audio.module_stt import ModeleSTT
from src.modules.chatbox import generer_reponse
import sys
sys.path.append(r'C:\Users\trexm\OneDrive\Bureau\project ai\NETY')


def poser_question():
    entree_utilisateur = entree.get()
    reponse = generer_reponse(entree_utilisateur, modele_texte, modele_image, modele_audio)
    reponse_label.config(text="NETY: " + reponse)

# Créez des instances des modèles de texte, d'image et d'audio
modele_texte = ModeleRNN(input_size=..., hidden_size=..., output_size=...)
modele_image = ModeleCNN(...)
modele_audio = ModeleSTT(...)

# Créez la fenêtre Tkinter
fenetre = tk.Tk()
fenetre.title("NETY IA")

# Ajoutez un champ d'entrée et un bouton
entree = tk.Entry(fenetre)
entree.pack()
bouton = tk.Button(fenetre, text="Poser une question", command=poser_question)
bouton.pack()

# Ajoutez une zone de texte pour afficher la réponse
reponse_label = tk.Label(fenetre, text="")
reponse_label.pack()

# Lancez la boucle principale de Tkinter
fenetre.mainloop()
