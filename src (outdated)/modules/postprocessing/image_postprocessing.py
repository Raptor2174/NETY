import torchvision.transforms as transforms
import torch

# Classe pour le post-traitement d'image
class ImagePostprocessor:
    def __init__(self):
        # Spécifiez les transformations d'image inverses (le cas échéant)
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        ])

    def postprocess_output(self, predicted_image):
        # Appliquez les transformations inverses (le cas échéant)
        postprocessed_image = self.transforms(predicted_image)
        # Assurez-vous que les valeurs sont dans la plage [0, 1]
        postprocessed_image = torch.clamp(postprocessed_image, 0, 1)
        return postprocessed_image

# Exemple d'utilisation
if __name__ == '__main__':
    # Créez une instance du post-processeur
    postprocessor = ImagePostprocessor()

    # Supposons que vous ayez une image prédite par votre modèle (un torch.Tensor)
    predicted_image = torch.randn(3, 256, 256)  # Exemple aléatoire

    # Post-traitez l'image prédite
    postprocessed_image = postprocessor.postprocess_output(predicted_image)

    # Affichez l'image post-traitée (pour le débogage)
    print(postprocessed_image)
