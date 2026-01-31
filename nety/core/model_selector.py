"""
Sélecteur de modèle interactif pour NETY - Modèles locaux uniquement
"""
import os
from typing import Literal

ModelChoice = Literal["mistral", "bloomz"]

class ModelSelector:
    """Gestionnaire de sélection de modèle"""
    
    def __init__(self):
        from nety.core.llm_config import LLMConfig
        self.config = LLMConfig()
        
        self.available_models = {
            "1": {
                "name": "mistral",
                "display": "Mistral-7B (Local GPU - Puissant mais gourmand)",
                "backend": "local",
                "requires_gpu": True,
                "ram_gb": 8,
                "cost": "Gratuit (utilise ton matériel)",
            },
            "2": {
                "name": "bloomz",
                "display": "BLOOMZ-560M (Local CPU - Léger)",
                "backend": "local",
                "requires_gpu": False,
                "ram_gb": 2,
                "cost": "Gratuit (utilise ton matériel)",
            },
            # Future models can be added here

        }
    
    def display_menu(self) -> None:
        """Affiche le menu avec infos de coût"""
        print("\n" + "=" * 70)
        print("🤖 SÉLECTION DU MODÈLE D'IA POUR NETY")
        print("=" * 70)
        print()
        
        for key, model in self.available_models.items():
            print(f"{key}. {model['display']}")
            print(f"   Backend: {model['backend']}")
            print(f"   RAM requise: {model['ram_gb']} GB")
            print(f"   💰 Coût: {model['cost']}")
            if model['requires_gpu']:
                print("   ⚠️ Nécessite un GPU avec 4+ GB VRAM")
            print()
    
    def get_user_choice(self) -> ModelChoice:
        """Demande à l'utilisateur de choisir"""
        self.display_menu()
        
        while True:
            choice = input("👉 Choisis ton modèle (1 ou 2): ").strip()
            
            if choice in self.available_models:
                selected = self.available_models[choice]
                
                print(f"✅ Modèle sélectionné: {selected['display']}")
                print()
                return selected["name"]
            else:
                print("❌ Choix invalide. Entre 1 ou 2.")
    
    def auto_select(self, prefer_local: bool = True) -> ModelChoice:
        """Sélection automatique"""
        import torch
        
        if torch.cuda.is_available():
            print("🤖 Auto-sélection: Mistral-7B (GPU détecté)")
            return "mistral"
        
        print("🤖 Auto-sélection: BLOOMZ (CPU uniquement)")
        return "bloomz"


def select_model(interactive: bool = True) -> ModelChoice:
    """
    Sélectionne un modèle
    
    Args:
        interactive: Menu interactif ou auto
    
    Returns:
        Nom du modèle choisi
    """
    selector = ModelSelector()
    
    if interactive:
        return selector.get_user_choice()
    else:
        return selector.auto_select()