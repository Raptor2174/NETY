# examples/clean_cache.py
"""
Script pour nettoyer le cache HuggingFace
"""
from pathlib import Path
import shutil
import os

def format_bytes(bytes):
    """Formate les bytes en GB/MB"""
    if bytes > 1024**3:
        return f"{bytes / (1024**3):.2f} GB"
    else:
        return f"{bytes / (1024**2):.2f} MB"

def get_folder_size(path):
    """Calcule la taille d'un dossier"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total

def delete_folder(path):
    """Supprime un dossier de manière sécurisée"""
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def main():
    print("=" * 80)
    print("🗑️ NETTOYAGE DU CACHE HUGGINGFACE")
    print("=" * 80)
    print()
    
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_path.exists():
        print("❌ Aucun cache trouvé")
        return
    
    print(f"📁 Cache: {cache_path}")
    print()
    
    # Trouver tous les modèles
    models_to_delete = []
    models_to_keep = []
    
    for item in cache_path.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            size = get_folder_size(item)
            model_name = item.name.replace("models--", "").replace("--", "/")
            
            # Vérifier si complet
            snapshots_dir = item / "snapshots"
            is_complete = snapshots_dir.exists() and any(snapshots_dir.iterdir())
            
            model_info = {
                "name": model_name,
                "path": item,
                "size": size,
                "complete": is_complete
            }
            
            # Stratégie de nettoyage
            if "mistral" in model_name.lower():
                if not is_complete:
                    models_to_delete.append(model_info)
                else:
                    models_to_keep.append(model_info)
            elif not is_complete:
                models_to_delete.append(model_info)
            else:
                models_to_keep.append(model_info)
    
    # Afficher ce qui sera supprimé
    print("🗑️ MODÈLES À SUPPRIMER (incomplets)")
    print("-" * 80)
    
    if not models_to_delete:
        print("✅ Aucun modèle incomplet à supprimer")
        print()
    else:
        total_freed = 0
        for model in models_to_delete:
            print(f"❌ {model['name']}")
            print(f"   Taille: {format_bytes(model['size'])}")
            print(f"   Statut: {'Complet' if model['complete'] else 'Incomplet'}")
            total_freed += model['size']
        
        print()
        print(f"💾 Espace à libérer: {format_bytes(total_freed)}")
        print()
    
    # Afficher ce qui sera gardé
    print("✅ MODÈLES À CONSERVER")
    print("-" * 80)
    
    if not models_to_keep:
        print("ℹ️ Aucun modèle complet trouvé")
    else:
        for model in models_to_keep:
            print(f"✓ {model['name']}")
            print(f"   Taille: {format_bytes(model['size'])}")
        print()
    
    # Demander confirmation
    if models_to_delete:
        print("=" * 80)
        response = input("⚠️ Confirmer la suppression ? (oui/non): ").strip().lower()
        
        if response in ["oui", "o", "yes", "y"]:
            print()
            print("🗑️ Nettoyage en cours...")
            print()
            
            freed_space = 0
            success_count = 0
            
            for model in models_to_delete:
                print(f"🗑️ Suppression de {model['name']}...", end=" ")
                if delete_folder(model['path']):
                    print("✅")
                    freed_space += model['size']
                    success_count += 1
                else:
                    print("❌")
            
            print()
            print("=" * 80)
            print(f"✅ Nettoyage terminé!")
            print(f"   Modèles supprimés: {success_count}/{len(models_to_delete)}")
            print(f"   Espace libéré: {format_bytes(freed_space)}")
            print("=" * 80)
        else:
            print("\n❌ Nettoyage annulé")
    else:
        print("✅ Rien à nettoyer!")

if __name__ == "__main__":
    main()