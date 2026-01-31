# examples/download_mistral_clean.py
"""
Téléchargement propre de Mistral avec reprise automatique
"""
from huggingface_hub import snapshot_download
import os
from pathlib import Path

def main():
    print("=" * 80)
    print("🔽 TÉLÉCHARGEMENT MISTRAL-7B-INSTRUCT-V0.2")
    print("=" * 80)
    print()
    
    # Vérifier l'espace disponible
    import shutil
    disk = shutil.disk_usage("C:\\")
    free_gb = disk.free / (1024**3)
    
    print(f"💾 Espace disque libre: {free_gb:.2f} GB")
    
    if free_gb < 20:
        print("❌ Espace insuffisant ! (besoin de 20 GB minimum)")
        print("   → Exécute d'abord: python examples/clean_cache.py")
        return
    
    print("✅ Espace suffisant")
    print()
    
    # Cache par défaut
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"📁 Cache: {cache_dir}")
    print()
    
    print("⏳ Téléchargement en cours...")
    print("   Taille: ~14 GB")
    print("   Durée estimée: 20-40 minutes")
    print()
    print("💡 Le téléchargement reprendra automatiquement si interrompu")
    print()
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        snapshot_download(
        repo_id=model_name,
        cache_dir=f"./models",
        allow_patterns="*.safetensors",
        ignore_patterns="*.bin"
)
        
        print()
        print("=" * 80)
        print("✅ TÉLÉCHARGEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 80)
        print()
        print("🚀 Tu peux maintenant lancer:")
        print("   python examples/test_mistral.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Téléchargement interrompu")
        print("💡 Relance ce script pour reprendre le téléchargement")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("💡 Vérifie ta connexion internet et réessaye")

if __name__ == "__main__":
    main()