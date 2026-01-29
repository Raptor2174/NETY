"""
Tokenizer simple pour le prétraitement de texte
"""
import torch
from typing import List, Dict

class SimpleTokenizer:
    """Tokenizer basique pour convertir du texte en tenseurs"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.idx_to_word: Dict[int, str] = {0: "<pad>", 1: "<unk>"}
        self.next_idx = 2
    
    def fit(self, texts: List[str]):
        """
        Construit le vocabulaire à partir d'une liste de textes
        
        Note: Seuls les premiers (vocab_size - 2) mots uniques seront ajoutés
        au vocabulaire (indices 0 et 1 sont réservés pour <pad> et <unk>).
        Les mots suivants seront traités comme des tokens inconnus.
        """
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in self.word_to_idx and self.next_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_idx
                    self.idx_to_word[self.next_idx] = word
                    self.next_idx += 1
    
    def encode(self, text: str, max_length: int = 20) -> torch.Tensor:
        """
        Convertit un texte en tensor d'indices
        
        Note: Les textes vides retournent un tensor rempli de tokens <pad>
        """
        if not text or not text.strip():
            # Handle empty text
            return torch.tensor([0] * max_length, dtype=torch.long)
        
        words = text.lower().split()
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 = <unk>
        
        # Padding ou truncation
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Convertit un tensor d'indices en texte"""
        words = [self.idx_to_word.get(idx.item(), "<unk>") for idx in indices]
        return " ".join([w for w in words if w != "<pad>"])
