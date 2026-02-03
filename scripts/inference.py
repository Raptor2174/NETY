"""
NETY V2-Maxx - Inference Script
================================

Script d'inférence pour tester la génération neuronale.
Démonstration de génération pure (sans templates).
"""

import torch
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nety.settings import NETYSettings
from nety.models.nety_brain_v2 import NETYBrainV2, NETYBrainConfig
from nety.preprocessing.text_preprocessor import Preprocessor
from nety.postprocessing.text_postprocessor import Postprocessor


class NETYInference:
    """Générateur de réponses NETY V2"""
    
    def __init__(
        self,
        model: NETYBrainV2,
        preprocessor: Preprocessor,
        postprocessor: Postprocessor,
        settings: NETYSettings,
        device: str = 'cpu'
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.settings = settings
        self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
    
    def generate_response(
        self,
        user_input: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        Génère une réponse à partir d'un input utilisateur
        
        Args:
            user_input: Texte de l'utilisateur
            max_length: Longueur max de génération
            temperature: Température de sampling
            top_k: Top-K sampling
            top_p: Nucleus sampling
        
        Returns:
            Réponse générée et post-processée
        """
        # Paramètres par défaut
        max_length = max_length or self.settings.generation.max_length
        temperature = temperature or self.settings.generation.temperature
        top_k = top_k or self.settings.generation.top_k
        top_p = top_p or self.settings.generation.top_p
        
        # 1. Préprocessing
        encoded = self.preprocessor(
            user_input,
            padding='max_length',
            truncation=True
        )
        
        input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
        
        # 2. Génération neuronale
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=self.settings.generation.repetition_penalty,
                eos_token_id=self.settings.model.eos_token_id,
                pad_token_id=self.settings.model.pad_token_id
            )
        
        # 3. Détokenization
        generated_text = self.preprocessor.decode(
            generated_ids[0].cpu().tolist(),
            skip_special_tokens=True
        )
        
        # 4. Postprocessing
        cleaned_text = self.postprocessor(generated_text)
        
        return cleaned_text if cleaned_text else "Je n'ai pas pu générer de réponse appropriée."
    
    def interactive_chat(self):
        """Mode chat interactif"""
        print("=" * 80)
        print("NETY V2-Maxx - Mode Chat Interactif")
        print("=" * 80)
        print("Tapez 'quit' ou 'exit' pour quitter")
        print("Tapez 'help' pour voir les commandes")
        print("=" * 80 + "\n")
        
        while True:
            try:
                user_input = input("Vous: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nAu revoir ! 👋")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommandes disponibles:")
                    print("  quit/exit - Quitter le chat")
                    print("  help - Afficher cette aide")
                    print("  temp <value> - Changer la température (0.1-2.0)")
                    print()
                    continue
                
                if user_input.lower().startswith('temp '):
                    try:
                        temp = float(user_input.split()[1])
                        if 0.1 <= temp <= 2.0:
                            self.settings.generation.temperature = temp
                            print(f"✓ Température changée: {temp}")
                        else:
                            print("⚠️  Température doit être entre 0.1 et 2.0")
                    except:
                        print("⚠️  Format invalide. Usage: temp 0.8")
                    continue
                
                # Générer réponse
                response = self.generate_response(user_input)
                print(f"NETY: {response}\n")
            
            except KeyboardInterrupt:
                print("\n\nInterruption détectée. Au revoir ! 👋")
                break
            except (ValueError, IndexError, RuntimeError) as e:
                print(f"⚠️  Erreur: {e}\n")


def load_model(checkpoint_path: str, settings: NETYSettings) -> NETYBrainV2:
    """Charge le modèle depuis un checkpoint"""
    # Charger preprocessor pour obtenir vocab_size
    preprocessor = Preprocessor.load("data/tokenizer")
    
    # Créer config
    config = NETYBrainConfig(
        vocab_size=len(preprocessor.tokenizer.token_to_id),
        embedding_dim=settings.model.embedding_dim,
        max_seq_length=settings.model.max_seq_length
    )
    
    # Créer modèle
    model = NETYBrainV2(config)
    
    # Charger poids
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Modèle chargé: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model, preprocessor


def test_generation(nety: NETYInference):
    """Teste la génération avec quelques exemples"""
    print("=" * 80)
    print("NETY V2-Maxx - Tests de Génération")
    print("=" * 80)
    
    test_inputs = [
        "Bonjour",
        "Comment vas-tu ?",
        "Qu'est-ce que tu peux faire ?",
        "Je suis content",
        "Merci pour ton aide",
        "Raconte-moi une blague",
        "Au revoir"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 Utilisateur: {user_input}")
        response = nety.generate_response(user_input)
        print(f"🤖 NETY: {response}")
    
    print("\n" + "=" * 80)


def main():
    """Script principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NETY V2-Maxx Inference")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Chemin vers le checkpoint'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'chat'],
        default='test',
        help='Mode: test (exemples) ou chat (interactif)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Température de génération'
    )
    
    args = parser.parse_args()
    
    # Charger settings
    settings = NETYSettings()
    
    if args.temperature is not None:
        settings.generation.temperature = args.temperature
    
    # Vérifier si modèle entraîné existe
    if not os.path.exists(args.checkpoint):
        print(f"⚠️  Checkpoint non trouvé: {args.checkpoint}")
        print("ℹ️  Le modèle n'est pas encore entraîné.")
        print("   Utilisation d'un modèle non entraîné pour démonstration.\n")
        
        # Créer modèle non entraîné
        preprocessor = Preprocessor.load("data/tokenizer") if os.path.exists("data/tokenizer/vocab.json") else None
        
        if preprocessor is None:
            print("⚠️  Tokenizer non trouvé. Exécutez d'abord train.py")
            return
        
        config = NETYBrainConfig(
            vocab_size=len(preprocessor.tokenizer.token_to_id),
            embedding_dim=settings.model.embedding_dim
        )
        model = NETYBrainV2(config)
        
        print("⚠️  ATTENTION: Modèle non entraîné, réponses aléatoires!")
    else:
        # Charger modèle entraîné
        model, preprocessor = load_model(args.checkpoint, settings)
    
    # Créer postprocessor
    postprocessor = Postprocessor()
    
    # Créer NETY
    nety = NETYInference(
        model=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        settings=settings,
        device=settings.hardware.device
    )
    
    # Mode
    if args.mode == 'test':
        test_generation(nety)
    else:
        nety.interactive_chat()


if __name__ == "__main__":
    main()
