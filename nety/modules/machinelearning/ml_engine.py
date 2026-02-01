import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nety.modules.text.tokenizer import SimpleTokenizer


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(x)
        out, _ = self.lstm(embeddings)
        last = out[:, -1, :]
        return self.fc(last)


class MLEngine:
    CATEGORY_LABELS = [
        "identity",
        "preferences",
        "location",
        "contact",
        "work",
        "goals",
        "health",
        "other",
    ]

    _STOPWORDS = {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "de",
        "du",
        "dans",
        "et",
        "ou",
        "pour",
        "par",
        "avec",
        "sans",
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        "est",
        "suis",
        "sont",
        "être",
        "avoir",
        "a",
        "ai",
        "as",
        "ce",
        "cet",
        "cette",
        "ces",
        "ça",
        "c'est",
        "sur",
        "au",
        "aux",
        "à",
    }

    _KEY_PATTERNS: List[Tuple[str, str, str]] = [
        (r"\bje m'appelle\s+([\wÀ-ÿ'\-]+)", "identity", "name"),
        (r"\bmon nom\s+est\s+([\wÀ-ÿ'\-]+)", "identity", "name"),
        (r"\bje suis\s+([\wÀ-ÿ'\- ]{2,})", "identity", "traits"),
        (r"\bj'habite\s+à\s+([\wÀ-ÿ'\- ]{2,})", "location", "location"),
        (r"\bje vis\s+à\s+([\wÀ-ÿ'\- ]{2,})", "location", "location"),
        (r"\bmon email\s+est\s+([\w.\-+]+@[\w\-]+\.[\w\-.]+)", "contact", "email"),
        (r"\b(mon|ma)\s+téléphone\s*(est|:)?\s*([\d +().-]{6,})", "contact", "phone"),
        (r"\bj'aime\s+([\wÀ-ÿ'\- ]{2,})", "preferences", "likes"),
        (r"\bje déteste\s+([\wÀ-ÿ'\- ]{2,})", "preferences", "dislikes"),
        (r"\bmon objectif\s*(est|:)?\s*([\wÀ-ÿ'\- ]{2,})", "goals", "goal"),
        (r"\bje travaille\s+chez\s+([\wÀ-ÿ'\- ]{2,})", "work", "company"),
        (r"\bje suis\s+malade\b|\bj'ai\s+mal\b", "health", "health"),
    ]

    def __init__(self, model: Optional[nn.Module] = None, data_dir: Optional[str] = None):
        """
        Initialise le moteur ML.

        Args:
            model: Modèle PyTorch (nn.Module) ou None pour un modèle par défaut
            data_dir: Répertoire de stockage des données ML
        """
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.data_dir = data_dir or os.path.join(self.root_dir, "data", "processed", "ml_engine")
        self.model_path = os.path.join(self.data_dir, "model.pt")
        self.vocab_path = os.path.join(self.data_dir, "vocab.json")
        self.labels_path = os.path.join(self.data_dir, "labels.json")
        self.memory_path = os.path.join(self.data_dir, "memory.jsonl")
        self.stats_path = os.path.join(self.data_dir, "stats.json")

        self._ensure_dirs()

        self.tokenizer = SimpleTokenizer(vocab_size=2000)

        self.model = model or self._create_default_model()
        self._load_state_if_available()

        print("✓ ML Engine initialisé")

    # ==========================================
    # 🔧 INITIALISATION & STOCKAGE
    # ==========================================
    def _ensure_dirs(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)

    def _create_default_model(self) -> nn.Module:
        """Crée un modèle simple pour la V1, mais entraînable"""
        return TextClassifier(vocab_size=2000, embed_dim=64, hidden_size=64, num_classes=len(self.CATEGORY_LABELS))

    def _load_state_if_available(self) -> None:
        if os.path.exists(self.vocab_path):
            self.tokenizer.load_vocab(self.vocab_path)

        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
                if labels:
                    self.CATEGORY_LABELS = labels

        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state)

    def _save_state(self) -> None:
        self.tokenizer.save_vocab(self.vocab_path)
        with open(self.labels_path, "w", encoding="utf-8") as f:
            json.dump(self.CATEGORY_LABELS, f, ensure_ascii=False, indent=2)
        torch.save(self.model.state_dict(), self.model_path)

    def _append_memory(self, entry: Dict) -> None:
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_key_info(self, key_info: Dict) -> None:
        key_info_path = os.path.join(self.data_dir, "key_info.jsonl")
        with open(key_info_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(key_info, ensure_ascii=False) + "\n")

    def _load_memory(self, limit: Optional[int] = None) -> List[Dict]:
        if not os.path.exists(self.memory_path):
            return []

        entries: List[Dict] = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        if limit is not None:
            return entries[-limit:]
        return entries

    def load_key_info(self) -> List[Dict]:
        key_info_path = os.path.join(self.data_dir, "key_info.jsonl")
        if not os.path.exists(key_info_path):
            return []
        entries: List[Dict] = []
        with open(key_info_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(
                        "⚠️ Ligne JSON invalide ignorée dans "
                        f"{key_info_path} (ligne {line_number}): {exc}"
                    )
        return entries

    def get_stats(self) -> Dict:
        if not os.path.exists(self.stats_path):
            return {"total_entries": 0, "category_counts": {}, "last_train_at": None}
        with open(self.stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==========================================
    # 🧠 EXTRACTION D'INFORMATIONS CLÉS
    # ==========================================
    def extract_key_info(self, text: str) -> Dict:
        facts: Dict[str, List[str]] = {}
        categories: List[str] = []

        for pattern, category, field in self._KEY_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                value = match.group(match.lastindex) if match.lastindex else match.group(0)
                value = value.strip()
                if value:
                    facts.setdefault(field, []).append(value)
                if category not in categories:
                    categories.append(category)

        keywords = self._extract_keywords(text)
        if not categories:
            categories = ["other"]

        return {
            "facts": facts,
            "categories": categories,
            "keywords": keywords,
        }

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[\wÀ-ÿ'\-]+", text.lower())
        keywords = [t for t in tokens if len(t) > 3 and t not in self._STOPWORDS]
        unique_keywords: List[str] = []
        for word in keywords:
            if word not in unique_keywords:
                unique_keywords.append(word)
        return unique_keywords[:10]

    # ==========================================
    # 📚 MÉMOIRE & APPRENTISSAGE
    # ==========================================
    def ingest_text(self, text: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        analysis = self.extract_key_info(text)
        entry = {
            "id": f"{datetime.utcnow().isoformat()}-{len(text)}",
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "facts": analysis["facts"],
            "categories": analysis["categories"],
            "keywords": analysis["keywords"],
            "user_id": user_id,
            "meta": metadata or {},
        }
        self._append_memory(entry)
        self._update_stats(entry)

        # Ajout de key_info corrélée si applicable
        if "name" in analysis["facts"] and user_id:
            key_info = {
                "type": "user_identity",
                "user_id": user_id,
                "identity": analysis["facts"]["name"][0],
                "timestamp": entry["timestamp"],
            }
            self._append_key_info(key_info)
        if "role" in analysis["facts"] and user_id:
            key_info = {
                "type": "user_role",
                "user_id": user_id,
                "roles": analysis["facts"]["role"],
                "timestamp": entry["timestamp"],
            }
            self._append_key_info(key_info)
        # ...ajoute d'autres corrélations/relations si besoin

        return entry

    def _update_stats(self, entry: Dict) -> None:
        stats = {"total_entries": 0, "category_counts": {}, "last_train_at": None}
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)

        stats["total_entries"] = stats.get("total_entries", 0) + 1
        for category in entry.get("categories", []):
            stats.setdefault("category_counts", {})
            stats["category_counts"][category] = stats["category_counts"].get(category, 0) + 1

        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def train_from_memory(self, epochs: int = 5, learning_rate: float = 0.001, min_samples: int = 5) -> bool:
        entries = self._load_memory()
        labeled = [e for e in entries if e.get("categories")]
        if len(labeled) < min_samples:
            print("ℹ️ Pas assez d'exemples pour entraîner un modèle ML.")
            return False

        texts = [e["text"] for e in labeled]
        labels = [self._category_to_label(e["categories"]) for e in labeled]

        self.tokenizer.fit(texts)
        inputs = torch.stack([self.tokenizer.encode(t) for t in texts])
        targets = torch.tensor(labels, dtype=torch.long)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        self._save_state()
        stats = self.get_stats()
        stats["last_train_at"] = datetime.utcnow().isoformat()
        with open(self.stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return True

    def _category_to_label(self, categories: List[str]) -> int:
        for category in categories:
            if category in self.CATEGORY_LABELS:
                return self.CATEGORY_LABELS.index(category)
        return self.CATEGORY_LABELS.index("other")

    def predict_category(self, text: str) -> str:
        self.model.eval()
        encoded = self.tokenizer.encode(text).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(encoded)
            predicted = torch.argmax(outputs, dim=1).item()
        return self.CATEGORY_LABELS[int(predicted)]

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Récupère les souvenirs pertinents en priorisant:
        1. Les dernières interactions
        2. Les correspondances de mots-clés
        3. Les faits corrélés
        """
        keywords = set(self._extract_keywords(query))
        all_memories = self._load_memory(limit=None)
        
        if not all_memories:
            return []

        scored: List[Tuple[int, Dict, int]] = []  # (score, entry, recency)
        
        for idx, entry in enumerate(all_memories):
            recency_score = idx  # Plus l'index est élevé, plus récent
            entry_keywords = set(entry.get("keywords", []))
            keyword_score = len(keywords.intersection(entry_keywords))
            
            # Score des faits corrélés
            facts_score = len(entry.get("facts", {}).values()) if entry.get("facts") else 0
            
            # Score total: recency + keywords + facts
            total_score = (recency_score * 2) + keyword_score + facts_score
            
            if total_score > 0 or idx >= len(all_memories) - 5:  # Inclure les 5 derniers
                scored.append((total_score, entry, recency_score))

        # Trier par score décroissant (récents d'abord)
        scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
        return [entry for _, entry, _ in scored[:limit]]

    def get_user_profile(self, user_id: Optional[str] = None) -> Dict[str, str]:
        profile: Dict[str, str] = {}
        entries = self._load_memory(limit=200)
        for entry in reversed(entries):
            if user_id and entry.get("user_id") != user_id:
                continue
            facts = entry.get("facts", {})
            for field, values in facts.items():
                if not values:
                    continue
                if field not in profile:
                    profile[field] = values[0]
        return profile

    # ==========================================
    # 🎯 MÉTHODES APPELÉES PAR BRAIN
    # ==========================================
    def transform_text(self, text: str) -> str:
        """
        Transforme/réécrit un texte
        Pour V1 : implémentation simple, sera améliorée avec ML
        """
        print(f"🔄 ML Engine transforme : {text}")
        transformed = text.upper()
        return f"[Transformé] {transformed}"

    def generate_response(self, text: str) -> str:
        """
        Génère une réponse conversationnelle
        Pour V1 : réponses basiques, sera améliorée avec ML
        """
        print(f"💬 ML Engine génère une réponse pour : {text}")

        analysis = self.extract_key_info(text)
        if analysis["facts"]:
            self.ingest_text(text)
            if "name" in analysis["facts"]:
                name = analysis["facts"]["name"][0]
                return f"Enchanté, {name}. Je note ça."
            if "likes" in analysis["facts"]:
                likes = analysis["facts"]["likes"][0]
                return f"D'accord, tu aimes {likes}. Je le retiens."
            if "location" in analysis["facts"]:
                location = analysis["facts"]["location"][0]
                return f"Merci ! J'ai noté que tu es à {location}."
            return "Merci, j'ai enregistré cette information."

        responses = {
            "bonjour": "Bonjour ! Comment puis-je vous aider ?",
            "salut": "Salut ! Que puis-je faire pour toi ?",
            "comment ça va": "Je vais bien, merci ! Et toi ?",
        }

        text_lower = text.lower()
        for keyword, response in responses.items():
            if keyword in text_lower:
                return response

        predicted = self.predict_category(text)
        return f"Je comprends. Catégorie détectée : {predicted}. Peux-tu préciser ?"

    # ==========================================
    # 🧠 MÉTHODES ML ORIGINALES
    # ==========================================
    
    def assign_memory_labels(self, text: str, user_id: Optional[str] = None) -> Dict:
        """
        Assigne des labels contextuels aux souvenirs et enregistre les corrélations.
        Améliore le système en catégorisant les interactions et en créant des liens.
        """
        analysis = self.extract_key_info(text)
        
        # Déterminer les labels contextuels
        labels = []
        if "name" in analysis["facts"]:
            labels.append("identity_info")
        if analysis["keywords"] and any(word in ["aime", "préfère", "adore"] for word in analysis["keywords"]):
            labels.append("preference")
        if "goals" in analysis["categories"]:
            labels.append("goal")
        if "health" in analysis["categories"]:
            labels.append("health_update")
        if len(text.split()) > 50:
            labels.append("detailed_context")
        else:
            labels.append("short_interaction")
        
        # Créer l'entrée mémoire enrichie avec labels
        entry = {
            "id": f"{datetime.utcnow().isoformat()}-{len(text)}",
            "timestamp": datetime.utcnow().isoformat(),
            "text": text,
            "facts": analysis["facts"],
            "categories": analysis["categories"],
            "keywords": analysis["keywords"],
            "user_id": user_id,
            "labels": labels,  # ✨ Nouveaux labels contextuels
            "meta": {
                "sentiment": self._analyze_sentiment(text),
                "urgency": self._determine_urgency(text),
            },
        }
        
        self._append_memory(entry)
        self._update_stats(entry)
        
        # Enregistrer les corrélations entre les informations clés
        self._store_correlations(entry, user_id)
        
        return entry
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyse le sentiment du texte"""
        positive_words = ["merci", "super", "génial", "content", "heureux", "aime", "formidable", "excellent", "cool"]
        negative_words = ["triste", "nul", "mauvais", "déçu", "horrible", "déteste", "frustré", "angry"]
        
        text_lower = text.lower()
        pos = sum(1 for word in positive_words if word in text_lower)
        neg = sum(1 for word in negative_words if word in text_lower)
        
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        else:
            return "neutral"
    
    def _determine_urgency(self, text: str) -> str:
        """Détermine le niveau d'urgence"""
        urgent_words = ["urgent", "aide", "problème", "bug", "crash", "immédiatement", "vite", "rapidement"]
        text_lower = text.lower()
        
        if any(word in text_lower for word in urgent_words):
            return "high"
        elif "bientôt" in text_lower or "à faire" in text_lower:
            return "medium"
        else:
            return "low"
    
    def _store_correlations(self, current_entry: Dict, user_id: Optional[str] = None) -> None:
        """
        Enregistre les corrélations entre les informations clés et les interactions précédentes.
        Crée des liens sémantiques entre les souvenirs.
        """
        key_info_path = os.path.join(self.data_dir, "key_info.jsonl")
        current_facts = current_entry.get("facts", {})
        
        if not current_facts:
            return
        
        # Créer une corrélation pour chaque fait important
        for field, values in current_facts.items():
            for value in values:
                correlation = {
                    "type": "correlation",
                    "field": field,
                    "value": value,
                    "user_id": user_id,
                    "memory_id": current_entry["id"],
                    "timestamp": current_entry["timestamp"],
                    "category": current_entry.get("categories", ["other"])[0],
                    "labels": current_entry.get("labels", []),
                    "sentiment": current_entry.get("meta", {}).get("sentiment", "neutral"),
                }
                
                with open(key_info_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(correlation, ensure_ascii=False) + "\n")
    
    def get_memory_with_context(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """
        Récupère les souvenirs d'un utilisateur avec le contexte complet:
        - Dernières interactions
        - Labels et corrélations associées
        - Faits clés avec sentiment
        """
        memories = self._load_memory(limit=None)
        user_memories = []
        
        for entry in reversed(memories):  # Récent d'abord
            if user_id is None or entry.get("user_id") == user_id:
                user_memories.append(entry)
        
        return user_memories[:limit]
    
    def get_related_memories(self, memory_id: str) -> List[Dict]:
        """
        Récupère tous les souvenirs corrélés à une entrée mémoire spécifique.
        Utile pour reconstituer le contexte complet d'une interaction.
        """
        key_info = self.load_key_info()
        related_ids = set()
        
        # Trouver tous les liens de corrélation vers ce memory_id
        for info in key_info:
            if info.get("type") == "correlation" and info.get("memory_id") == memory_id:
                related_ids.add(info.get("memory_id"))
        
        all_memories = self._load_memory()
        related = [m for m in all_memories if m.get("id") in related_ids]
        
        return related

    def train(self, data, labels, epochs: int = 10, learning_rate: float = 0.01):
        """Entraîne le modèle (API brute)"""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def evaluate(self, data, labels):
        """Évalue le modèle"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def predict(self, data):
        """Fait une prédiction brute"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
        return predicted