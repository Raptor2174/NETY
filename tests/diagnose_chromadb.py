"""
Script de diagnostic ChromaDB pour NETY
"""
import sys
print(f"🐍 Python version: {sys.version}")
print()

# Test 1: Import de chromadb
print("Test 1: Import de chromadb...")
try:
    import chromadb
    print(f"✅ ChromaDB importé (version: {chromadb.__version__})")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

print()

# Test 2: Création d'un client
print("Test 2: Création d'un client ChromaDB...")
try:
    from pathlib import Path
    test_path = Path("./test_chroma_diagnostic")
    test_path.mkdir(exist_ok=True)
    
    client = chromadb.PersistentClient(path=str(test_path))
    print(f"✅ Client créé dans: {test_path}")
except Exception as e:
    print(f"❌ Erreur de création du client: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Création d'une collection
print("Test 3: Création d'une collection...")
try:
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"description": "Test collection"}
    )
    print(f"✅ Collection créée: {collection.name}")
except Exception as e:
    print(f"❌ Erreur de création de collection: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Ajout et recherche de données
print("Test 4: Ajout et recherche de données...")
try:
    collection.add(
        documents=["Ceci est un test"],
        ids=["test1"]
    )
    results = collection.query(
        query_texts=["test"],
        n_results=1
    )
    print(f"✅ Données ajoutées et recherchées avec succès")
except Exception as e:
    print(f"❌ Erreur lors de l'ajout/recherche: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✅ TOUS LES TESTS PASSÉS - ChromaDB fonctionne correctement!")
print("=" * 60)

# Nettoyage
import shutil
try:
    shutil.rmtree(test_path)
    print("🧹 Fichiers de test nettoyés")
except:
    pass