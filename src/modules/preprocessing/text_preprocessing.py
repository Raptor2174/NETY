import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import spacy
import string

# Utilisez le modèle de langue spaCy pour le prétraitement (assurez-vous de l'installer)
# Charger le modèle de langue français
nlp = spacy.load("fr_core_news_sm")

# Fonction pour le prétraitement d'une seule phrase
def preprocess_text(text):
    # Utilisez spaCy pour tokeniser et prétraiter le texte
    tokens = [token.text.lower() for token in nlp(text) if token.text not in string.punctuation]
    return tokens

# Classe du jeu de données personnalisé pour le traitement du texte
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # Tokenisez et convertissez le texte en indices de vocabulaire
        tokens = preprocess_text(text)
        text_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(text_indices)

# Fonction pour créer un DataLoader pour le texte
def create_text_loader(data, batch_size=32):
    # Créez un vocabulaire à partir des données
    vocab = build_vocab_from_iterator(preprocess_text(text) for text in data)
    # Créez un jeu de données personnalisé
    dataset = TextDataset(data, vocab)
    # Utilisez un DataLoader pour gérer les mini-lots
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence)
    return loader
