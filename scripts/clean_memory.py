#!/usr/bin/env python3
"""
Script pour nettoyer et trier la mémoire JSONL
- Supprime les doublons basés sur le texte
- Garde les souvenirs uniques et pertinents
- Priorise par catégorie et pertinence
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_memory(filepath):
    """Charge tous les souvenirs du fichier JSONL"""
    memories = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return memories

def is_relevant(memory):
    """Détermine si un souvenir est pertinent"""
    text = memory.get('text', '').strip()
    categories = memory.get('categories', [])
    facts = memory.get('facts', {})
    
    # Ignorer les question/réponses génériques générées automatiquement
    generic_questions = [
        "Quel est le plus grand",
        "Quelle est la capitale",
        "Qui a peint",
        "Qui a écrit",
        "Qui a découvert",
        "Qui a inventé",
        "Combien de",
        "Quel est l'élément",
        "Quelle est la monnaie",
        "Quelle planète",
        "Quelle langue",
        "Quelle est la formule",
        "Quelle est la vitesse",
        "En quelle année",
        "Quel est l'océan"
    ]
    
    # Vérifier si c'est une question générique
    for generic in generic_questions:
        if text.startswith(generic):
            return False
    
    # Ignorer les réponses vides ou très courtes sans contexte
    if len(text) < 5 and not facts and 'other' in categories:
        return False
    
    # Garder les souvenirs avec des facts/informations pertinentes
    if facts or 'identity' in categories or 'preferences' in categories or 'goals' in categories or 'work' in categories:
        return True
    
    # Garder les interactions significatives
    if len(text) > 30 and categories not in [['other'], []]:
        return True
    
    return False

def deduplicate(memories):
    """Supprime les doublons en gardant le premier occurrence"""
    seen_texts = set()
    unique_memories = []
    duplicates = []
    
    for memory in memories:
        text = memory.get('text', '').strip()
        
        # Clé pour la déduplication: texte normalisé
        key = text.lower().strip()
        
        if key not in seen_texts:
            seen_texts.add(key)
            unique_memories.append(memory)
        else:
            duplicates.append(memory)
    
    return unique_memories, duplicates

def score_memory(memory):
    """Assigne un score de pertinence au souvenir"""
    score = 0
    categories = memory.get('categories', [])
    facts = memory.get('facts', {})
    meta = memory.get('meta', {})
    text_len = len(memory.get('text', ''))
    
    # Bonus par catégorie
    if 'identity' in categories:
        score += 100
    if 'preferences' in categories:
        score += 80
    if 'goals' in categories:
        score += 90
    if 'work' in categories:
        score += 70
    if 'contact' in categories:
        score += 75
    
    # Bonus pour les facts
    if facts:
        score += len(facts) * 30
    
    # Bonus pour urgence haute
    if meta.get('urgency') == 'high':
        score += 50
    
    # Bonus pour sentiment positif
    if meta.get('sentiment') == 'positive':
        score += 20
    
    # Bonus pour texte plus long (plus d'infos)
    if text_len > 50:
        score += 10
    if text_len > 100:
        score += 20
    
    return score

def clean_and_sort_memory(input_path, output_path):
    """Nettoie et trie la mémoire"""
    print("📖 Chargement de la mémoire...")
    memories = load_memory(input_path)
    print(f"   Total: {len(memories)} souvenirs")
    
    # Déduplique
    print("\n🔍 Suppression des doublons...")
    unique_memories, duplicates = deduplicate(memories)
    print(f"   Souvenirs uniques: {len(unique_memories)}")
    print(f"   Doublons supprimés: {len(duplicates)}")
    
    # Filtre les souvenirs pertinents
    print("\n✨ Filtrage des souvenirs pertinents...")
    relevant_memories = [m for m in unique_memories if is_relevant(m)]
    irrelevant = [m for m in unique_memories if not is_relevant(m)]
    print(f"   Souvenirs pertinents: {len(relevant_memories)}")
    print(f"   Souvenirs non pertinents: {len(irrelevant)}")
    
    # Trie par score de pertinence (descendant)
    print("\n⭐ Tri par pertinence...")
    relevant_memories.sort(key=score_memory, reverse=True)
    
    # Groupe par catégorie
    print("\n📊 Analyse par catégorie:")
    by_category = defaultdict(list)
    for memory in relevant_memories:
        for cat in memory.get('categories', ['other']):
            by_category[cat].append(memory)
    
    for cat, mems in sorted(by_category.items()):
        print(f"   {cat}: {len(mems)} souvenirs")
    
    # Sauvegarde la mémoire nettoyée
    print(f"\n💾 Sauvegarde de la mémoire nettoyée: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for memory in relevant_memories:
            f.write(json.dumps(memory, ensure_ascii=False) + '\n')
    
    # Rapport
    print("\n" + "="*50)
    print("📋 RAPPORT DE NETTOYAGE")
    print("="*50)
    print(f"Souvenirs originaux:     {len(memories)}")
    print(f"Après déduplication:    {len(unique_memories)}")
    print(f"Souvenirs gardés:       {len(relevant_memories)}")
    print(f"Réduction:              {100 * (1 - len(relevant_memories) / len(memories)):.1f}%")
    print("="*50)
    
    # Affiche les meilleurs souvenirs
    print("\n⭐ TOP 10 souvenirs les plus pertinents:")
    for i, memory in enumerate(relevant_memories[:10], 1):
        text = memory.get('text', '').replace('\n', ' ')[:80]
        score = score_memory(memory)
        print(f"   {i}. [{score}pts] {text}...")
    
    return relevant_memories

if __name__ == "__main__":
    input_file = Path(__file__).parent.parent / "data" / "processed" / "ml_engine" / "memory.jsonl"
    output_file = Path(__file__).parent.parent / "data" / "processed" / "ml_engine" / "memory_cleaned.jsonl"
    
    memories = clean_and_sort_memory(input_file, output_file)
    
    # Option: remplacer l'original
    print("\n🔄 Remplacement du fichier original? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        import shutil
        shutil.copy(input_file, input_file.with_suffix('.jsonl.backup'))
        shutil.move(output_file, input_file)
        print(f"✅ Sauvegarde: {input_file.with_suffix('.jsonl.backup')}")
        print(f"✅ Original remplacé: {input_file}")
