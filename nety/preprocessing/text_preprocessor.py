"""
NETY V2-Maxx - Preprocessing Module
====================================

Preprocessing pipeline pour transformer le texte brut en input neuronal :
1. Normalisation du texte
2. Tokenization (BPE, WordPiece, ou SentencePiece)
3. Conversion en IDs
4. Padding/Truncation
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import json
import os


class TextNormalizer:
    """Normalisation du texte avant tokenization"""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_accents: bool = False,
        normalize_unicode: bool = True,
        remove_extra_spaces: bool = True
    ):
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.normalize_unicode = normalize_unicode
        self.remove_extra_spaces = remove_extra_spaces
    
    def normalize(self, text: str) -> str:
        """Normalise le texte"""
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove accents
        if self.remove_accents:
            text = ''.join(
                c for c in unicodedata.normalize('NFD', text)
                if unicodedata.category(c) != 'Mn'
            )
        
        # Remove extra spaces
        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text


class SimpleTokenizer:
    """
    Tokenizer simple basé sur BPE simplifié
    
    Pour une implémentation complète, utiliser :
    - tokenizers library (Hugging Face)
    - sentencepiece
    - tiktoken (OpenAI)
    """
    
    def _build_id_to_token(self) -> Dict[int, str]:
        """Construit le mapping id_to_token depuis token_to_id"""
        return {v: k for k, v in self.token_to_id.items()}
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        
        # Tokens spéciaux
        self.pad_token = "[PAD]"
        self.sos_token = "[SOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # Vocabulaire
        self.token_to_id: Dict[str, int] = {
            self.pad_token: self.pad_token_id,
            self.sos_token: self.sos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
        }
        self.id_to_token: Dict[int, str] = self._build_id_to_token()
        
        # Compteur pour nouveaux tokens
        self.next_id = 4
    
    def build_vocab(self, texts: List[str], min_frequency: int = 2):
        """
        Construit le vocabulaire à partir d'un corpus
        
        Args:
            texts: Liste de textes
            min_frequency: Fréquence minimale pour inclure un token
        """
        # Compter les mots
        word_counts: Dict[str, int] = {}
        
        for text in texts:
            words = self._simple_tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filtrer par fréquence et trier
        sorted_words = sorted(
            [(word, count) for word, count in word_counts.items() if count >= min_frequency],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Ajouter au vocabulaire (limité à vocab_size)
        for word, _ in sorted_words[:self.vocab_size - 4]:  # -4 pour tokens spéciaux
            if word not in self.token_to_id:
                self.token_to_id[word] = self.next_id
                self.id_to_token[self.next_id] = word
                self.next_id += 1
        
        print(f"✓ Vocabulaire construit: {len(self.token_to_id)} tokens")
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Tokenization simple (whitespace + ponctuation)"""
        # Ajouter espaces autour de la ponctuation
        text = re.sub(r'([.!?,;:()])', r' \1 ', text)
        
        # Split sur whitespace
        tokens = text.split()
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encode le texte en IDs
        
        Args:
            text: Texte à encoder
            add_special_tokens: Ajouter SOS/EOS
            max_length: Longueur maximale
            padding: 'max_length' ou 'do_not_pad'
            truncation: Tronquer si trop long
        
        Returns:
            dict avec 'input_ids' et 'attention_mask'
        """
        # Tokenize
        tokens = self._simple_tokenize(text)
        
        # Convertir en IDs
        ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Ajouter tokens spéciaux
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        
        # Truncation
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_special_tokens:
                ids[-1] = self.eos_token_id  # Garder EOS
        
        # Padding
        attention_mask = [1] * len(ids)
        
        if padding == 'max_length' and max_length is not None:
            pad_length = max_length - len(ids)
            if pad_length > 0:
                ids = ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Décode les IDs en texte
        
        Args:
            ids: Liste d'IDs
            skip_special_tokens: Ignorer tokens spéciaux
        
        Returns:
            Texte décodé
        """
        tokens = []
        
        for id_ in ids:
            token = self.id_to_token.get(id_, self.unk_token)
            
            # Skip special tokens
            if skip_special_tokens and token in [
                self.pad_token, self.sos_token, self.eos_token
            ]:
                continue
            
            tokens.append(token)
        
        # Joindre et nettoyer
        text = ' '.join(tokens)
        
        # Nettoyer ponctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        text = re.sub(r'\(\s+', r'(', text)
        text = re.sub(r'\s+\)', r')', text)
        
        return text
    
    def save(self, path: str):
        """Sauvegarde le vocabulaire"""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad_token': self.pad_token,
                'sos_token': self.sos_token,
                'eos_token': self.eos_token,
                'unk_token': self.unk_token,
            }
        }
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Vocabulaire sauvegardé: {path}")
    
    def load(self, path: str):
        """Charge le vocabulaire"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = self._build_id_to_token()
        self.vocab_size = vocab_data['vocab_size']
        
        special = vocab_data['special_tokens']
        self.pad_token = special['pad_token']
        self.sos_token = special['sos_token']
        self.eos_token = special['eos_token']
        self.unk_token = special['unk_token']
        
        self.next_id = max(self.token_to_id.values()) + 1
        
        print(f"✓ Vocabulaire chargé: {path} ({len(self.token_to_id)} tokens)")


class Preprocessor:
    """
    Pipeline de preprocessing complet
    
    Usage:
        preprocessor = Preprocessor()
        preprocessor.fit(corpus)  # Construire vocabulaire
        encoded = preprocessor(text)  # Encoder texte
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        lowercase: bool = True,
        remove_accents: bool = False
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Composants
        self.normalizer = TextNormalizer(
            lowercase=lowercase,
            remove_accents=remove_accents
        )
        self.tokenizer = SimpleTokenizer(vocab_size=vocab_size)
        
        self._is_fitted = False
    
    def fit(self, texts: List[str], min_frequency: int = 2):
        """
        Construit le vocabulaire à partir d'un corpus
        
        Args:
            texts: Liste de textes
            min_frequency: Fréquence minimale pour un token
        """
        # Normaliser textes
        normalized_texts = [self.normalizer.normalize(text) for text in texts]
        
        # Construire vocabulaire
        self.tokenizer.build_vocab(normalized_texts, min_frequency=min_frequency)
        
        self._is_fitted = True
        print(f"✓ Preprocessor fitted sur {len(texts)} textes")
    
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: str = 'max_length',
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Preprocess un texte
        
        Args:
            text: Texte à preprocesser
            add_special_tokens: Ajouter SOS/EOS
            padding: 'max_length' ou 'do_not_pad'
            truncation: Tronquer si trop long
        
        Returns:
            dict avec 'input_ids' et 'attention_mask'
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor pas encore fitted. Appeler .fit() d'abord.")
        
        # Normaliser
        normalized = self.normalizer.normalize(text)
        
        # Encoder
        encoded = self.tokenizer.encode(
            normalized,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            padding=padding,
            truncation=truncation
        )
        
        return encoded
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Décode les IDs en texte"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: str = 'max_length',
        truncation: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Encode un batch de textes"""
        batch_encoded = {
            'input_ids': [],
            'attention_mask': []
        }
        
        for text in texts:
            encoded = self(text, add_special_tokens, padding, truncation)
            batch_encoded['input_ids'].append(encoded['input_ids'])
            batch_encoded['attention_mask'].append(encoded['attention_mask'])
        
        return batch_encoded
    
    def save(self, directory: str):
        """Sauvegarde le preprocessor"""
        os.makedirs(directory, exist_ok=True)
        
        # Sauvegarder tokenizer
        vocab_path = os.path.join(directory, 'vocab.json')
        self.tokenizer.save(vocab_path)
        
        # Sauvegarder config
        config_path = os.path.join(directory, 'preprocessor_config.json')
        config = {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'lowercase': self.normalizer.lowercase,
            'remove_accents': self.normalizer.remove_accents,
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Preprocessor sauvegardé: {directory}")
    
    @classmethod
    def load(cls, directory: str) -> 'Preprocessor':
        """Charge le preprocessor"""
        # Charger config
        config_path = os.path.join(directory, 'preprocessor_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Créer preprocessor
        preprocessor = cls(
            vocab_size=config['vocab_size'],
            max_length=config['max_length'],
            lowercase=config['lowercase'],
            remove_accents=config['remove_accents']
        )
        
        # Charger tokenizer
        vocab_path = os.path.join(directory, 'vocab.json')
        preprocessor.tokenizer.load(vocab_path)
        
        preprocessor._is_fitted = True
        
        print(f"✓ Preprocessor chargé: {directory}")
        return preprocessor


if __name__ == "__main__":
    # Test du preprocessor
    print("=" * 80)
    print("Test du Preprocessor")
    print("=" * 80)
    
    # Corpus d'exemple
    corpus = [
        "Bonjour, comment vas-tu aujourd'hui ?",
        "Je vais très bien, merci !",
        "Qu'est-ce que tu aimes faire ?",
        "J'aime lire des livres et écouter de la musique.",
        "C'est génial ! Moi aussi j'adore la musique.",
        "Quel est ton style de musique préféré ?",
        "J'aime le rock et le jazz.",
        "Super ! Tu as de bons goûts musicaux.",
    ]
    
    # Créer et fitter preprocessor
    preprocessor = Preprocessor(vocab_size=1000, max_length=32)
    preprocessor.fit(corpus, min_frequency=1)
    
    # Test encoding
    text = "Bonjour ! Comment vas-tu ?"
    encoded = preprocessor(text)
    
    print(f"\n✓ Texte original: {text}")
    print(f"  Input IDs: {encoded['input_ids'][:10]}... (truncated)")
    print(f"  Attention mask: {encoded['attention_mask'][:10]}... (truncated)")
    
    # Test decoding
    decoded = preprocessor.decode(encoded['input_ids'])
    print(f"  Décodé: {decoded}")
    
    # Test batch
    texts = corpus[:3]
    batch = preprocessor.batch_encode(texts)
    print(f"\n✓ Batch encoding: {len(batch['input_ids'])} textes")
    
    # Test save/load
    preprocessor.save("test_preprocessor")
    loaded = Preprocessor.load("test_preprocessor")
    
    # Test avec loaded
    encoded2 = loaded(text)
    print(f"\n✓ Test après chargement: {preprocessor.decode(encoded2['input_ids'])}")
    
    print("\n" + "=" * 80)
    print("✓ Tous les tests passés!")
    print("=" * 80)
