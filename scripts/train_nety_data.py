import json
import random
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nety.core.brain import Brain

# --- Configuration ---
MAX_SECONDS = 5 * 60  # 5 minutes max
TARGET_ACCURACY = 1.0
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "ml_engine" / "dataset.jsonl"

# --- Panel large de questions/réponses ---
QA_PREDEFINED = [
    ("Quel est le plus grand océan du monde ?", "L'océan Pacifique"),
    ("Qui a inventé l'ampoule électrique ?", "Thomas Edison"),
    ("Quel est la capitale de l'Australie ?", "Canberra"),
    ("Quelle est la capitale de la France ?", "Paris"),
    ("Qui a écrit Les Misérables ?", "Victor Hugo"),
    ("Combien de continents existe-t-il ?", "7"),
    ("Quel est le plus grand mammifère ?", "La baleine bleue"),
    ("Quelle planète est la plus proche du Soleil ?", "Mercure"),
    ("Quelle est la formule de l'eau ?", "H2O"),
    ("Qui a peint la Joconde ?", "Léonard de Vinci"),
    ("Quelle est la monnaie du Japon ?", "Le yen"),
    ("En quelle année a eu lieu le premier pas sur la Lune ?", "1969"),
    ("Quel est l'organe principal de la respiration ?", "Les poumons"),
    ("Quelle est la capitale du Canada ?", "Ottawa"),
    ("Quel est l'élément chimique O ?", "Oxygène"),
    ("Qui a écrit Le Petit Prince ?", "Antoine de Saint-Exupéry"),
    ("Quel est le plus grand désert du monde ?", "Le désert de l'Antarctique"),
    ("Quelle langue parle-t-on au Brésil ?", "Le portugais"),
    ("Qui a découvert la gravitation ?", "Isaac Newton"),
    ("Quelle est la capitale de l'Allemagne ?", "Berlin"),
    ("Quel est le plus haut sommet du monde ?", "L'Everest"),
    ("Combien de côtés a un hexagone ?", "6"),
    ("Quelle est la vitesse de la lumière (approx) ?", "299792 km/s"),
    ("Quelle est la capitale de l'Italie ?", "Rome"),
    ("Quel est le plus grand pays du monde ?", "La Russie"),
    ("Quel est l'océan entre l'Afrique et l'Australie ?", "L'océan Indien"),
    ("Quel est l'élément chimique Fe ?", "Fer"),
    ("Combien de minutes dans une heure ?", "60"),
    ("Quelle est la capitale de l'Espagne ?", "Madrid"),
    ("Quel est le symbole chimique de l'or ?", "Au"),
]

def normalize(text: str) -> str:
    return text.lower().strip()

def nety_answer(brain: Brain, question: str) -> str:
    return brain.think(question)

def append_dataset(question: str, expected: str) -> None:
    # 👉 utiliser l’ingestion du MLEngine au lieu de dataset.jsonl
    pass

def train_cycle(brain: Brain) -> float:
    random.shuffle(QA_PREDEFINED)
    correct = 0
    total = 0

    for q, expected in QA_PREDEFINED:
        total += 1
        try:
            reply = nety_answer(brain, q)
        except Exception as exc:
            reply = f"__error__: {exc}"

        is_ok = normalize(reply) == normalize(expected)
        if is_ok:
            correct += 1

        brain.ml_engine.ingest_text(q)

        print(f"Q: {q}")
        print(f"Attendu: {expected}")
        print(f"NETY: {reply}")
        print(f"OK: {is_ok}")
        print("-" * 50)

    return correct / max(total, 1)

def run_training():
    # ✅ FORCER L'UTILISATION DU RNN LOCAL (pas BLOOMZ!)
    brain = Brain(model_type="rnn")
    start = time.time()
    best_acc = 0.0

    while True:
        acc = train_cycle(brain)
        best_acc = max(best_acc, acc)
        elapsed = time.time() - start

        print(f"🎯 Accuracy: {acc:.2%} | Best: {best_acc:.2%} | Temps: {elapsed:.1f}s")

        # Lancer l'entraînement ML interne si disponible
        try:
            if hasattr(brain.ml_engine, "train_from_memory"):
                brain.ml_engine.train_from_memory()
        except Exception as exc:
            print(f"⚠️ Entraînement ML interne impossible: {exc}")

        if best_acc >= TARGET_ACCURACY:
            print("✅ Objectif atteint (100%)")
            break
        if elapsed >= MAX_SECONDS:
            print("⏰ Temps limite atteint (5 minutes)")
            break

if __name__ == "__main__":
    run_training()