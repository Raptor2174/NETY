import torchvision.transforms as transforms
from PIL import Image

# Classe pour le prétraitement d'image
class ImagePreprocessor:
    def __init__(self):
        # Spécifiez les transformations d'image que vous souhaitez appliquer
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_input(self, image_path):
        # Chargez l'image depuis le chemin du fichier
        image = Image.open(image_path)
        # Appliquez les transformations définies
        preprocessed_image = self.transforms(image)
        return preprocessed_image

# Exemple d'utilisation
if __name__ == '__main__':
    # Créez une instance du préprocesseur
    preprocessor = ImagePreprocessor()

    # Supposons que vous ayez un chemin d'image en entrée
    image_path = 'image.jpg'

    # Prétraitez l'image
    preprocessed_image = preprocessor.preprocess_input(image_path)

    # Affichez l'image prétraitée (pour le débogage)
    print(preprocessed_image)
