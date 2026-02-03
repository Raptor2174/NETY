"""
NETY V2-Maxx - Setup Data
==========================

Prépare les données et le tokenizer sans faire l'entraînement complet.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nety.preprocessing.text_preprocessor import Preprocessor
from nety.settings import NETYSettings


def main():
    print("=" * 80)
    print("NETY V2-Maxx - Setup Data")
    print("=" * 80)
    
    # Charger settings
    settings = NETYSettings()
    
    # Charger conversations
    print("\n📥 Chargement des conversations...")
    with open(settings.data.train_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data['conversations']
    print(f"✓ Chargé {len(conversations)} conversations")
    
    # Créer preprocessor
    print("\n🔧 Création du preprocessor...")
    preprocessor = Preprocessor(
        vocab_size=settings.model.vocab_size,
        max_length=settings.model.max_seq_length,
        lowercase=settings.data.lowercase
    )
    
    # Extraire tous les textes
    all_texts = []
    for conv in conversations:
        all_texts.append(conv['input'])
        all_texts.append(conv['output'])
    
    print(f"✓ Textes extraits: {len(all_texts)}")
    
    # Fit preprocessor
    print("\n📝 Construction du vocabulaire...")
    preprocessor.fit(all_texts, min_frequency=2)
    
    # Sauvegarder
    print("\n💾 Sauvegarde du preprocessor...")
    preprocessor.save("data/tokenizer")
    
    # Stats
    vocab_size = len(preprocessor.tokenizer.token_to_id)
    print(f"\n📊 Statistiques:")
    print(f"  Vocabulaire: {vocab_size:,} tokens")
    print(f"  Max sequence length: {settings.model.max_seq_length}")
    
    # Test encoding/decoding
    print("\n✅ Test encoding/decoding:")
    test_text = "Bonjour, comment vas-tu ?"
    encoded = preprocessor(test_text)
    decoded = preprocessor.decode(encoded['input_ids'])
    print(f"  Original: {test_text}")
    print(f"  Décodé: {decoded}")
    
    print("\n" + "=" * 80)
    print("✓ Setup terminé avec succès!")
    print("=" * 80)


if __name__ == "__main__":
    main()
