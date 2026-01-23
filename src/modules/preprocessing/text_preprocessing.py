"""
Module de prétraitement de texte pour le NLP.

Ce module fournit des fonctions et classes pour le prétraitement de données textuelles
en utilisant spaCy et PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import spacy
import string
from typing import List, Any

# Utilisez le modèle de langue spaCy pour le prétraitement (assurez-vous de l'installer)
# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")


def preprocess_input(text: str) -> List[str]:
    """
    Prétraite une seule phrase de texte.
    
    Effectue la tokenisation et le nettoyage en retirant la ponctuation
    et en convertissant en minuscules.
    
    Args:
        text: Texte brut à prétraiter
        
    Returns:
        Liste de tokens nettoyés et en minuscules
    """
    # Utilisez spaCy pour tokeniser et prétraiter le texte
    tokens = [token.text.lower() for token in nlp(text) if token.text not in string.punctuation]
    return tokens


class TextDataset(Dataset):
    """
    Jeu de données personnalisé pour le traitement de texte avec PyTorch.
    
    Convertit le texte en indices de vocabulaire pour l'entraînement de modèles.
    
    Args:
        data: Liste de textes bruts
        vocab: Vocabulaire pour la conversion token -> indice
    """
    
    def __init__(self, data: List[str], vocab: Any) -> None:
        self.data = data
        self.vocab = vocab

    def __len__(self) -> int:
        """Retourne la taille du dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Récupère un élément du dataset.
        
        Args:
            idx: Indice de l'élément à récupérer
            
        Returns:
            Tensor contenant les indices de vocabulaire pour le texte
        """
        text = self.data[idx]
        # Tokenisez et convertissez le texte en indices de vocabulaire
        tokens = preprocess_input(text)
        # Gérer les tokens hors vocabulaire avec un token spécial <unk>
        text_indices = [self.vocab.get(token, self.vocab.get('<unk>', 0)) for token in tokens]
        return torch.tensor(text_indices)


def create_text_loader(data: List[str], batch_size: int = 32) -> DataLoader:
    """
    Crée un DataLoader pour le texte.
    
    Construit un vocabulaire à partir des données et crée un DataLoader
    pour l'entraînement par mini-lots.
    
    Args:
        data: Liste de textes bruts
        batch_size: Taille des mini-lots (défaut: 32)
        
    Returns:
        DataLoader PyTorch configuré pour le texte
    """
    # Créez un vocabulaire à partir des données
    vocab = build_vocab_from_iterator(preprocess_input(text) for text in data)
    # Créez un jeu de données personnalisé
    dataset = TextDataset(data, vocab)
    # Utilisez un DataLoader pour gérer les mini-lots
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    return loader
