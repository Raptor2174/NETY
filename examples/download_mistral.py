# download_mistral.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

print("📥 Téléchargement de Mistral-7B...")
print("⚠️ Cela peut prendre 30-120 minutes selon ta connexion")
print()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Télécharger le tokenizer (rapide)
print("1️⃣ Téléchargement du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ Tokenizer téléchargé")

# Télécharger le modèle (long)
print("\n2️⃣ Téléchargement du modèle (14 GB)...")
print("📍 Destination:", os.path.expanduser("~/.cache/huggingface/hub/"))

# Télécharger sans charger en mémoire
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=model_name,
    cache_dir=f"./models",
    allow_patterns="*.safetensors",
    ignore_patterns="*.bin"
)

print("\n✅ Téléchargement terminé!")