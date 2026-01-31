# examples/check_cache.py
"""
Script pour diagnostiquer et nettoyer le cache HuggingFace
"""
from pathlib import Path
import shutil
import os

def format_bytes(bytes):
    """Formate les bytes en GB/MB"""
    if bytes > 1024**3:
        return f"{bytes / (1024**3):.2f} GB"
    elif bytes > 1024**2:
        return f"{bytes / (1024**2):.2f} MB"
    else:
        return f"{bytes / 1024:.2f} KB"

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

def count_files(path):
    """Compte les fichiers dans un dossier"""
    count = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                count += 1
            elif entry.is_dir():
                count += count_files(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return count

def main():
    print("=" * 80)
    print("🔍 DIAGNOSTIC DU CACHE HUGGINGFACE")
    print("=" * 80)
    print()
    
    # Emplacement du cache
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_path.exists():
        print("❌ Aucun cache HuggingFace trouvé")
        print(f"   Chemin recherché: {cache_path}")
        return
    
    print(f"📁 Cache HuggingFace: {cache_path}")
    print()
    
    # Analyser tous les modèles
    models = []
    total_size = 0
    
    for item in cache_path.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            size = get_folder_size(item)
            files = count_files(item)
            
            # Extraire le nom du modèle
            model_name = item.name.replace("models--", "").replace("--", "/")
            
            # Vérifier si c'est un téléchargement complet
            snapshots_dir = item / "snapshots"
            is_complete = snapshots_dir.exists() and any(snapshots_dir.iterdir())
            
            models.append({
                "name": model_name,
                "path": item,
                "size": size,
                "files": files,
                "complete": is_complete
            })
            total_size += size
    
    # Trier par taille (plus gros en premier)
    models.sort(key=lambda x: x["size"], reverse=True)
    
    # Afficher les résultats
    print(f"📊 RÉSUMÉ")
    print(f"  Total modèles: {len(models)}")
    print(f"  Espace total utilisé: {format_bytes(total_size)}")
    print()
    
    print("📦 DÉTAILS DES MODÈLES")
    print("-" * 80)
    
    mistral_count = 0
    mistral_total = 0
    
    for i, model in enumerate(models, 1):
        status = "✅ Complet" if model["complete"] else "⚠️ Incomplet"
        print(f"{i}. {model['name']}")
        print(f"   Taille: {format_bytes(model['size'])}")
        print(f"   Fichiers: {model['files']}")
        print(f"   Statut: {status}")
        print(f"   Chemin: {model['path']}")
        
        # Compter les Mistral
        if "mistral" in model['name'].lower():
            mistral_count += 1
            mistral_total += model['size']
            print(f"   🔴 MISTRAL DÉTECTÉ #{mistral_count}")
        
        print()
    
    # Résumé Mistral
    if mistral_count > 0:
        print("=" * 80)
        print(f"⚠️ ALERTE: {mistral_count} copie(s) de Mistral détectée(s)")
        print(f"   Espace total Mistral: {format_bytes(mistral_total)}")
        
        if mistral_count > 1:
            print(f"   🗑️ Tu peux libérer ~{format_bytes(mistral_total)} en nettoyant !")
        print("=" * 80)
        print()
    
    # Espace disque
    disk = shutil.disk_usage(str(cache_path))
    print(f"💾 ESPACE DISQUE {cache_path.drive}")
    print(f"  Total: {format_bytes(disk.total)}")
    print(f"  Utilisé: {format_bytes(disk.used)}")
    print(f"  Libre: {format_bytes(disk.free)}")
    print()
    
    # Recommandations
    print("💡 RECOMMANDATIONS")
    print("-" * 80)
    
    incomplete = [m for m in models if not m["complete"]]
    if incomplete:
        print(f"⚠️ {len(incomplete)} téléchargement(s) incomplet(s) détecté(s)")
        print("   → Tu peux les supprimer sans risque")
        print()
    
    if mistral_count > 1:
        print(f"🗑️ {mistral_count} copies de Mistral trouvées")
        print("   → Garde seulement la version complète")
        print()
    
    if disk.free < 20 * 1024**3:  # Moins de 20 GB
        print(f"💾 Espace disque faible: {format_bytes(disk.free)}")
        print(f"   → Nettoie le cache pour libérer ~{format_bytes(total_size)}")
        print()

if __name__ == "__main__":
    main()