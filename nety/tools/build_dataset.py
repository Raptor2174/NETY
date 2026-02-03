import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

DEFAULT_MEMORY = os.path.join("data", "processed", "ml_engine", "memory.jsonl")
DEFAULT_OUTPUT = os.path.join("data", "processed", "ml_engine", "dataset.jsonl")

def _parse_ts(ts: str) -> Tuple[int, float]:
    try:
        dt = datetime.fromisoformat(ts)
        return (0, dt.timestamp())
    except Exception:
        return (1, 0.0)

def _load_memory(path: str) -> List[Dict]:
    items = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_line_idx"] = i
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items

def _group_by_user(items: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for it in items:
        user_id = it.get("user_id") or "unknown"
        grouped.setdefault(user_id, []).append(it)
    return grouped

def build_dataset(memory_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = _load_memory(memory_path)
    if not items:
        print(f"⚠️ Aucun item trouvé dans {memory_path}")
        return

    grouped = _group_by_user(items)

    total_pairs = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for user_id, msgs in grouped.items():
            msgs.sort(key=lambda x: (_parse_ts(x.get("timestamp", "")), x.get("_line_idx", 0)))
            texts = [m.get("text", "").strip() for m in msgs]
            texts = [t for t in texts if len(t) >= 2]

            for i in range(len(texts) - 1):
                src = texts[i]
                tgt = texts[i + 1]
                out_f.write(json.dumps({"input": src, "target": tgt}, ensure_ascii=False) + "\n")
                total_pairs += 1

    print(f"✅ Dataset créé: {output_path}")
    print(f"   Paires générées: {total_pairs}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", default=DEFAULT_MEMORY, help="Chemin vers memory.jsonl")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="Chemin du dataset.jsonl")
    args = parser.parse_args()
    build_dataset(args.memory, args.out)

if __name__ == "__main__":
    main()