# Guide d'utilisation de main.py

## Description

Le fichier `main.py` est le point d'entrée principal de l'application NETY. Il fournit une interface en ligne de commande interactive pour tester et démontrer les différents modules d'intelligence artificielle du projet.

## Prérequis

### Dépendances Python

Avant d'exécuter `main.py`, assurez-vous d'avoir installé les dépendances nécessaires :

```bash
# Installer toutes les dépendances
pip install -r requirements.txt

# Ou installer manuellement les dépendances essentielles
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install numpy
```

### Modèles spaCy (optionnel pour le module texte)

Pour le traitement de texte avancé :

```bash
python -m spacy download fr_core_news_sm
```

## Lancement de l'application

### Méthode simple

```bash
python main.py
```

### Avec Python 3 explicitement

```bash
python3 main.py
```

### En tant qu'exécutable (Unix/Linux/Mac)

```bash
chmod +x main.py
./main.py
```

## Structure du menu

Lorsque vous lancez `main.py`, vous verrez un menu principal avec les options suivantes :

```
======================================================================
               NETY - Intelligence Artificielle
          Traitement Multimodal : Texte, Image, Audio
======================================================================

──────────────────────────────────────────────────────────────────────
MENU PRINCIPAL
──────────────────────────────────────────────────────────────────────
1. Module de traitement de texte (RNN/LSTM)
2. Module de traitement d'images (CNN)
3. Module de traitement audio (STT)
4. À propos de NETY
0. Quitter l'application
──────────────────────────────────────────────────────────────────────
```

## Utilisation des modules

### 1. Module de traitement de texte (RNN/LSTM)

**Objectif** : Démonstration du modèle de réseau de neurones récurrent pour le traitement séquentiel.

**Ce que fait cette option** :
- Charge le modèle RNN depuis `src/modules/module_text/modele_rnn.py`
- Affiche les paramètres du modèle (taille d'entrée, couches cachées, etc.)
- Exécute une prédiction de test avec des données aléatoires
- Montre la structure et les capacités du modèle

**Exemple de sortie** :
```
Initialisation du modèle RNN...
✓ Modèle RNN créé avec succès!
  - Taille d'entrée: 10
  - Taille cachée: 64
  - Nombre de couches LSTM: 2
  - Taille de sortie: 1

Test du modèle avec des données aléatoires...
✓ Prédiction effectuée avec succès!
  - Forme de sortie: torch.Size([1, 1])
  - Valeur de sortie: 0.0234
```

### 2. Module de traitement d'images (CNN)

**Objectif** : Démonstration du réseau de neurones convolutionnel pour la classification d'images.

**Ce que fait cette option** :
- Charge le modèle CNN depuis `src/modules/module_image/modele_cnn.py`
- Affiche l'architecture du modèle (5 couches de convolution)
- Exécute une classification de test sur une image aléatoire
- Montre les prédictions du modèle

**Exemple de sortie** :
```
Initialisation du modèle CNN...
✓ Modèle CNN créé avec succès!
  - Nombre de classes: 10
  - Architecture: 5 couches de convolution
  - Pooling: Global Average Pooling

Test du modèle avec une image aléatoire...
✓ Classification effectuée avec succès!
  - Forme de sortie: torch.Size([1, 10])
  - Classe prédite: 3
```

### 3. Module de traitement audio (STT)

**Objectif** : Démonstration du modèle Speech-to-Text basé sur CNN.

**Ce que fait cette option** :
- Charge le modèle STT depuis `src/modules/module_audio/module_stt.py`
- Affiche l'architecture du modèle pour traiter les spectrogrammes
- Montre le résumé détaillé du modèle Keras
- Explique les capacités de traitement audio

**Exemple de sortie** :
```
Initialisation du modèle de traitement audio...
✓ Modèle STT créé avec succès!
  - Forme d'entrée: (128, 128, 1)
  - Nombre de classes: 10
  - Architecture: CNN pour spectrogrammes

Résumé du modèle:
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
...
```

### 4. À propos de NETY

**Objectif** : Afficher les informations sur le projet NETY.

**Ce que montre cette option** :
- Description du projet
- Objectifs principaux
- Technologies utilisées
- Informations sur le chef de projet
- Lien vers le repository GitHub

## Navigation

- **Sélectionner une option** : Tapez le numéro (1-4) et appuyez sur Entrée
- **Quitter l'application** : Tapez 0 et appuyez sur Entrée
- **Interruption d'urgence** : Appuyez sur `Ctrl+C` pour quitter immédiatement

## Gestion des erreurs

Le fichier `main.py` gère plusieurs types d'erreurs :

### Erreurs d'importation

Si un module n'est pas trouvé :
```
✗ Erreur d'importation: No module named 'torch'
  Vérifiez que le module existe dans src/modules/module_text/modele_rnn.py
  et que PyTorch est installé (pip install torch).
```

**Solution** : Installez la dépendance manquante avec pip.

### Erreurs d'attribut

Si un module n'a pas la structure attendue :
```
✗ Erreur d'attribut: ...
  Le module RNN n'a pas la structure attendue.
```

**Solution** : Vérifiez que les fichiers dans `src/modules/` sont à jour.

### Autres erreurs

Pour toute autre erreur, un message générique sera affiché avec les détails de l'exception.

## Notes importantes

1. **Modèles non entraînés** : Les modèles affichés dans `main.py` sont des architectures vides. Pour des prédictions réelles, ils doivent être entraînés avec des données appropriées.

2. **Données de test** : Les démonstrations utilisent des données aléatoires uniquement pour montrer que les modèles fonctionnent structurellement.

3. **Performance** : L'exécution peut être lente lors du premier lancement car PyTorch et TensorFlow initialisent leurs backends.

4. **Compatibilité** : Testé avec Python 3.12+. Des versions antérieures peuvent fonctionner mais ne sont pas officiellement supportées.

## Dépannage

### Le menu ne s'affiche pas correctement

**Problème** : Les caractères spéciaux ne s'affichent pas.

**Solution** : Assurez-vous que votre terminal supporte l'UTF-8.

### Erreur "No space left on device"

**Problème** : Pas assez d'espace disque pour installer les dépendances.

**Solution** : Libérez de l'espace ou utilisez un environnement avec plus de stockage.

### Les modèles ne se chargent pas

**Problème** : Erreurs d'importation pour torch ou tensorflow.

**Solution** : Réinstallez les dépendances :
```bash
pip install --upgrade torch torchvision tensorflow keras
```

## Développement futur

Le fichier `main.py` pourra être étendu avec :
- Interface web (Flask/FastAPI)
- Chargement de modèles pré-entraînés
- Support pour des données réelles (images, audio, texte)
- Sauvegarde et chargement de configurations
- Mode batch pour traiter plusieurs fichiers

## Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub : https://github.com/Raptor2174/NETY
- Consulter la documentation dans le dossier `documentation/`
- Contacter Raptor_ via GitHub

---

*"On n'a jamais fini d'apprendre" 🚀*
