"""
Sélecteur de modèle interactif pour NETY
Mistral (local GPU) + BLOOMZ (local CPU) + Groq (cloud gratuit)
"""
import os
from typing import Literal

ModelChoice = Literal["mistral", "bloomz", "groq"]

class ModelSelector:
    """Gestionnaire de sélection de modèle"""
    
    def __init__(self):
        self.available_models = {
            "1": {
                "name": "mistral",
                "display": "Mistral-7B (Local GPU - Puissant mais gourmand)",
                "backend": "local",
                "requires_gpu": True,
                "ram_gb": 8,
                "vram_gb": 4,
                "cost": "💰 Gratuit (utilise ton matériel)",
                "internet": "📶 Non requis",
                "speed": "⚡ Moyen (dépend GPU)",
                "quality": "🧠 Excellent",
            },
            "2": {
                "name": "bloomz",
                "display": "BLOOMZ-560M (Local CPU - Léger et rapide)",
                "backend": "local",
                "requires_gpu": False,
                "ram_gb": 2,
                "vram_gb": 0,
                "cost": "💰 Gratuit (utilise ton matériel)",
                "internet": "📶 Non requis",
                "speed": "⚡ Rapide (CPU uniquement)",
                "quality": "🧠 Correct",
            },
            "3": {
                "name": "groq",
                "display": "Groq Cloud - Llama 3.2 (Cloud ultra rapide)",
                "backend": "cloud",
                "requires_gpu": False,
                "ram_gb": 0,
                "vram_gb": 0,
                "cost": "💰 Gratuit (14.4k req/jour)",
                "internet": "📶 Requis",
                "speed": "⚡⚡⚡ Ultra rapide (500 tok/sec)",
                "quality": "🧠 Excellent",
            }
        }
    
    def display_menu(self) -> None:
        """Affiche le menu de sélection avec détails"""
        print("\n" + "=" * 70)
        print("🤖 SÉLECTION DU MODÈLE D'IA POUR NETY")
        print("=" * 70)
        print()
        
        for key, model in self.available_models.items():
            print(f"{key}. {model['display']}")
            print(f"   Type: {model['backend'].upper()}")
            print(f"   {model['cost']}")
            print(f"   {model['internet']}")
            print(f"   {model['speed']}")
            print(f"   {model['quality']}")
            
            if model['ram_gb'] > 0:
                print(f"   💾 RAM: {model['ram_gb']} GB", end="")
                if model['vram_gb'] > 0:
                    print(f" + {model['vram_gb']} GB VRAM (GPU)")
                else:
                    print()
            else:
                print(f"   💾 RAM: 0 GB (cloud)")
            
            if model['requires_gpu']:
                print("   ⚠️ Nécessite un GPU compatible CUDA")
            
            print()
        
        print("💡 Recommandations:")
        print("   • Pas de GPU → BLOOMZ (option 2)")
        print("   • GPU disponible → Mistral (option 1)")
        print("   • PC faible + internet → Groq (option 3)")
        print()
    
    def get_user_choice(self) -> ModelChoice:
        """Demande à l'utilisateur de choisir un modèle"""
        self.display_menu()
        
        while True:
            choice = input("👉 Choisis ton modèle (1, 2 ou 3): ").strip()
            
            if choice in self.available_models:
                selected = self.available_models[choice]
                
                # Vérifications spécifiques
                if selected["name"] == "groq":
                    # Vérifier la clé API Groq
                    api_key = os.getenv("GROQ_API_KEY", "")
                    if not api_key:
                        print("\n" + "=" * 70)
                        print("❌ ERREUR: Clé API Groq manquante!")
                        print("=" * 70)
                        print()
                        print("📝 Comment obtenir ta clé API GRATUITE:")
                        print()
                        print("  1. Va sur https://console.groq.com/")
                        print("  2. Clique sur 'Sign Up' (gratuit, pas de CB)")
                        print("  3. Une fois connecté, va dans 'API Keys'")
                        print("  4. Clique 'Create API Key'")
                        print("  5. Copie la clé (commence par 'gsk_...')")
                        print()
                        print("  6. Ajoute-la dans ton fichier .env:")
                        print("     GROQ_API_KEY=gsk_votre_clé_ici")
                        print()
                        print("  OU définis-la temporairement:")
                        print("     Windows: set GROQ_API_KEY=gsk_votre_clé")
                        print("     Linux:   export GROQ_API_KEY=gsk_votre_clé")
                        print()
                        print("💡 Limites gratuites Groq:")
                        print("   • 30 requêtes par minute")
                        print("   • 14 400 requêtes par jour")
                        print("   • Gratuit à vie (pas de CB requise)")
                        print("   • Ultra rapide (500 tokens/sec)")
                        print()
                        
                        retry = input("As-tu ajouté ta clé ? (o/n): ").strip().lower()
                        if retry == 'o':
                            # Recharger les variables d'environnement
                            from dotenv import load_dotenv
                            load_dotenv(override=True)
                            api_key = os.getenv("GROQ_API_KEY", "")
                            if api_key:
                                print("✅ Clé API détectée!")
                            else:
                                print("❌ Clé toujours manquante. Choisis une autre option.")
                                continue
                        else:
                            print("👌 Choisis une autre option pour l'instant.")
                            continue
                    
                    # Vérifier la connexion internet
                    print("\n🌐 Vérification de la connexion internet...")
                    if not self._check_internet():
                        print("❌ Pas de connexion internet détectée!")
                        print("💡 Groq nécessite internet. Choisis Mistral ou BLOOMZ.")
                        continue
                    
                    print("✅ Connexion internet OK")
                    print(f"✅ Groq API configuré")
                    print()
                    print("📊 Informations Groq:")
                    print(f"   • Modèle: llama-3.2-3b-preview")
                    print(f"   • Vitesse: ~500 tokens/seconde")
                    print(f"   • Limite: 14 400 requêtes/jour")
                    print(f"   • Coût: 0€ (gratuit)")
                    print()
                
                elif selected["name"] == "mistral":
                    # Vérifier le GPU
                    import torch
                    if not torch.cuda.is_available():
                        print("\n⚠️ ATTENTION: Aucun GPU CUDA détecté!")
                        print("   Mistral-7B va tourner sur CPU (très lent)")
                        print()
                        confirm = input("Continuer quand même ? (o/n): ").strip().lower()
                        if confirm != 'o':
                            print("💡 Choisis BLOOMZ (option 2) pour CPU")
                            continue
                
                print(f"\n✅ Modèle sélectionné: {selected['display']}")
                print()
                return selected["name"]
            
            else:
                print("❌ Choix invalide. Entre 1, 2 ou 3.")
    
    def _check_internet(self) -> bool:
        """Vérifie la connexion internet"""
        import requests
        try:
            response = requests.get("https://api.groq.com/openai/v1/models", timeout=3)
            return True
        except:
            return False
    
    def auto_select(self, prefer_local: bool = True) -> ModelChoice:
        """Sélection automatique basée sur les ressources"""
        import torch
        
        # Si préfère cloud ET Groq disponible
        if not prefer_local:
            api_key = os.getenv("GROQ_API_KEY", "")
            if api_key and self._check_internet():
                print("🤖 Auto-sélection: Groq Cloud (API key détectée)")
                return "groq"
        
        # Si GPU disponible → Mistral
        if torch.cuda.is_available():
            print("🤖 Auto-sélection: Mistral-7B (GPU détecté)")
            return "mistral"
        
        # Sinon BLOOMZ (léger pour CPU)
        print("🤖 Auto-sélection: BLOOMZ (CPU uniquement)")
        return "bloomz"


def select_model(interactive: bool = True) -> ModelChoice:
    """
    Fonction helper pour sélectionner un modèle
    
    Args:
        interactive: Si True, affiche un menu interactif
                    Si False, fait une sélection automatique
    
    Returns:
        Le nom du modèle choisi
    """
    selector = ModelSelector()
    
    if interactive:
        return selector.get_user_choice()
    else:
        return selector.auto_select()