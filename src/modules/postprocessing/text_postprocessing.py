import torch

# Classe pour le post-traitement du texte
class TextPostprocessor:
    def __init__(self, vocab):
        self.vocab = vocab

    def postprocess_output(self, predicted_indices):
        # Convertissez les indices en texte en utilisant le vocabulaire
        predicted_text = [self.vocab.itos[idx] for idx in predicted_indices]
        # Rejoignez les tokens pour former une phrase
        predicted_sentence = ' '.join(predicted_text)
        return predicted_sentence

# Exemple d'utilisation
if __name__ == '__main__':
    # Créez un exemple de vocabulaire (vous devrez adapter cela à votre propre vocabulaire)
    vocab = torch.load('vocab.pth')

    # Créez une instance du post-processeur
    postprocessor = TextPostprocessor(vocab)

    # Supposons que vous ayez des indices prédits par votre modèle (liste d'entiers)
    predicted_indices = [1, 2, 3, 4, 5]

    # Post-traitez les indices prédits en texte
    predicted_text = postprocessor.postprocess_output(predicted_indices)

    # Affichez le texte prédit
    print(predicted_text)
