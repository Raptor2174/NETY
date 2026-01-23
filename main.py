#!/usr/bin/env python3
"""
NETY - IA de Traitement de Langage Naturel
Point d'entrée principal de l'application

Ce module fournit une interface en ligne de commande pour interagir avec
les différents modules de NETY : traitement de texte (RNN), traitement d'images (CNN)
et traitement audio (STT).
"""

import sys
import os
import torch
import numpy as np


def clear_screen():
    """Efface l'écran du terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """Affiche la bannière de l'application."""
    print("=" * 70)
    print(" " * 15 + "NETY - Intelligence Artificielle")
    print(" " * 10 + "Traitement Multimodal : Texte, Image, Audio")
    print("=" * 70)
    print()


def print_menu():
    """Affiche le menu principal."""
    print("\n" + "─" * 70)
    print("MENU PRINCIPAL")
    print("─" * 70)
    print("1. Module de traitement de texte (RNN/LSTM)")
    print("2. Module de traitement d'images (CNN)")
    print("3. Module de traitement audio (STT)")
    print("4. À propos de NETY")
    print("0. Quitter l'application")
    print("─" * 70)


def module_text():
    """Démonstration du module de traitement de texte avec RNN."""
    clear_screen()
    print("=" * 70)
    print("MODULE DE TRAITEMENT DE TEXTE (RNN/LSTM)")
    print("=" * 70)
    print()
    
    try:
        from src.modules.module_text.modele_rnn import ModeleRNN
        
        print("Initialisation du modèle RNN...")
        # Paramètres du modèle
        input_size = 10
        hidden_size = 64
        output_size = 1
        num_layers = 2
        
        # Créer le modèle
        model = ModeleRNN(input_size, hidden_size, output_size, num_layers)
        print("✓ Modèle RNN créé avec succès!")
        print(f"  - Taille d'entrée: {input_size}")
        print(f"  - Taille cachée: {hidden_size}")
        print(f"  - Nombre de couches LSTM: {num_layers}")
        print(f"  - Taille de sortie: {output_size}")
        print()
        
        # Créer un exemple de données
        print("Test du modèle avec des données aléatoires...")
        batch_size = 1
        sequence_length = 5
        test_input = torch.randn(batch_size, sequence_length, input_size)
        
        # Faire une prédiction
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✓ Prédiction effectuée avec succès!")
        print(f"  - Forme de sortie: {output.shape}")
        print(f"  - Valeur de sortie: {output.item():.4f}")
        print()
        print("Note: Ce modèle doit être entraîné avec des données réelles pour")
        print("      être utilisé dans des applications concrètes.")
        
    except ImportError as e:
        print(f"✗ Erreur d'importation: {e}")
        print("  Vérifiez que tous les modules sont présents.")
    except Exception as e:
        print(f"✗ Erreur: {e}")
    
    print()
    input("Appuyez sur Entrée pour continuer...")


def module_image():
    """Démonstration du module de traitement d'images avec CNN."""
    clear_screen()
    print("=" * 70)
    print("MODULE DE TRAITEMENT D'IMAGES (CNN)")
    print("=" * 70)
    print()
    
    try:
        from src.modules.module_image.modele_cnn import ModeleCNN
        
        print("Initialisation du modèle CNN...")
        # Paramètres du modèle
        num_classes = 10
        
        # Créer le modèle
        model = ModeleCNN(num_classes=num_classes)
        print("✓ Modèle CNN créé avec succès!")
        print(f"  - Nombre de classes: {num_classes}")
        print(f"  - Architecture: 5 couches de convolution")
        print(f"  - Pooling: Global Average Pooling")
        print()
        
        # Créer un exemple d'image
        print("Test du modèle avec une image aléatoire...")
        batch_size = 1
        channels = 3  # RGB
        height = 224
        width = 224
        test_image = torch.randn(batch_size, channels, height, width)
        
        # Faire une prédiction
        model.eval()
        with torch.no_grad():
            output = model(test_image)
        
        print(f"✓ Classification effectuée avec succès!")
        print(f"  - Forme de sortie: {output.shape}")
        print(f"  - Classe prédite: {torch.argmax(output, dim=1).item()}")
        print()
        print("Note: Ce modèle doit être entraîné avec des images réelles pour")
        print("      effectuer des classifications précises.")
        
    except ImportError as e:
        print(f"✗ Erreur d'importation: {e}")
        print("  Vérifiez que tous les modules sont présents.")
    except Exception as e:
        print(f"✗ Erreur: {e}")
    
    print()
    input("Appuyez sur Entrée pour continuer...")


def module_audio():
    """Démonstration du module de traitement audio avec STT."""
    clear_screen()
    print("=" * 70)
    print("MODULE DE TRAITEMENT AUDIO (STT)")
    print("=" * 70)
    print()
    
    try:
        from src.modules.module_audio.module_stt import create_audio_processing_model
        
        print("Initialisation du modèle de traitement audio...")
        # Paramètres du modèle
        audio_height = 128
        audio_width = 128
        audio_channels = 1
        num_classes = 10
        input_shape = (audio_height, audio_width, audio_channels)
        
        # Créer le modèle
        model = create_audio_processing_model(input_shape, num_classes)
        print("✓ Modèle STT créé avec succès!")
        print(f"  - Forme d'entrée: {input_shape}")
        print(f"  - Nombre de classes: {num_classes}")
        print(f"  - Architecture: CNN pour spectrogrammes")
        print()
        
        # Afficher le résumé du modèle
        print("Résumé du modèle:")
        model.summary()
        print()
        print("Note: Ce modèle utilise des CNN pour traiter les spectrogrammes audio")
        print("      et doit être entraîné avec des données audio réelles.")
        
    except ImportError as e:
        print(f"✗ Erreur d'importation: {e}")
        print("  Vérifiez que TensorFlow/Keras et les modules sont installés.")
    except Exception as e:
        print(f"✗ Erreur: {e}")
    
    print()
    input("Appuyez sur Entrée pour continuer...")


def about():
    """Affiche les informations sur NETY."""
    clear_screen()
    print("=" * 70)
    print("À PROPOS DE NETY")
    print("=" * 70)
    print()
    print("NETY est un projet d'intelligence artificielle multimodale axé sur :")
    print()
    print("  • Traitement du Langage Naturel (NLP) avec RNN/LSTM")
    print("  • Traitement d'Images avec CNN")
    print("  • Traitement Audio (Speech-to-Text) avec CNN")
    print()
    print("Objectifs du projet :")
    print()
    print("  ✓ Créer une IA multimodale capable de traiter texte, images et audio")
    print("  ✓ Apprendre les concepts fondamentaux du machine learning")
    print("  ✓ Développer une architecture modulaire et extensible")
    print("  ✓ Participer à l'innovation dans le domaine de l'IA")
    print()
    print("Technologies utilisées :")
    print()
    print("  • PyTorch (RNN/LSTM et CNN)")
    print("  • TensorFlow/Keras (STT)")
    print("  • spaCy (prétraitement NLP)")
    print("  • librosa (traitement audio)")
    print()
    print("Chef de projet : Raptor_")
    print("GitHub : https://github.com/Raptor2174/NETY")
    print()
    print("─" * 70)
    print('"On n\'a jamais fini d\'apprendre" 🚀')
    print("─" * 70)
    print()
    input("Appuyez sur Entrée pour continuer...")


def main():
    """Fonction principale de l'application."""
    try:
        while True:
            clear_screen()
            print_banner()
            print_menu()
            
            try:
                choice = input("\nVotre choix : ").strip()
                
                if choice == "1":
                    module_text()
                elif choice == "2":
                    module_image()
                elif choice == "3":
                    module_audio()
                elif choice == "4":
                    about()
                elif choice == "0":
                    clear_screen()
                    print("\n" + "=" * 70)
                    print(" " * 20 + "Merci d'avoir utilisé NETY!")
                    print(" " * 15 + "Au revoir et à bientôt! 👋")
                    print("=" * 70 + "\n")
                    sys.exit(0)
                else:
                    print("\n✗ Choix invalide. Veuillez sélectionner une option valide.")
                    input("Appuyez sur Entrée pour continuer...")
                    
            except KeyboardInterrupt:
                print("\n\nInterruption détectée...")
                clear_screen()
                print("\n" + "=" * 70)
                print(" " * 20 + "Merci d'avoir utilisé NETY!")
                print(" " * 15 + "Au revoir et à bientôt! 👋")
                print("=" * 70 + "\n")
                sys.exit(0)
                
    except Exception as e:
        print(f"\n✗ Erreur inattendue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
