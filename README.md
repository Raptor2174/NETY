# NETY - IA de Traitement de Langage Naturel

## 📋 Description du Projet

**NETY** est un projet de création d'intelligence artificielle axé sur le traitement du langage naturel (NLP), le traitement d'images (CNN) et le traitement audio (Speech-to-Text). Ce projet est avant tout un projet d'apprentissage pour explorer l'univers de l'IA et le langage Python.

## 🎯 Objectifs

- Créer une IA multimodale capable de traiter du texte, des images et de l'audio
- Apprendre les concepts fondamentaux du machine learning et du deep learning
- Développer une architecture modulaire et extensible
- Participer à l'innovation dans le domaine de l'IA

## 👥 Équipe

- **Chef de projet** : Raptor_ (propriétaire de NETY)

## 🏗️ Architecture du Projet 

## 🏗️ Architecture du Projet

```
NETY/
├── src/
│   ├── modules/
│   │   ├── module_text/       # Traitement de texte avec RNN/LSTM
│   │   ├── module_image/      # Traitement d'images avec CNN
│   │   ├── module_audio/      # Traitement audio et STT
│   │   ├── preprocessing/     # Prétraitement des données
│   │   ├── postprocessing/    # Post-traitement des résultats
│   │   └── module_interface/  # Interfaces utilisateur
│   └── utils/                 # Fonctions utilitaires
├── data/                      # Données d'entraînement
├── documentation/             # Documentation technique
└── main.py                    # Point d'entrée principal
```

## 📝 Étapes du Projet

1. ✅ **Structuration du projet** - Organisation des dossiers et modules
2. ✅ **Création des modules essentiels** - Modules de base pour le développement
3. ⏳ **Entraînement de l'IA** - Avec les données d'entraînement
4. ⏳ **Évaluation des performances** - Tests et métriques
5. ⏳ **Extension des modules** - Ajout de nouvelles fonctionnalités
6. ⏳ **Amélioration continue** - Optimisation des composants
7. ⏳ **Déploiement** - Mise en production potentielle
8. ⏳ **Maintenance** - Résolution de bugs et améliorations continues

## 🚀 Installation

### Prérequis

- Python 3.12+
- pip

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/Raptor2174/NETY.git
cd NETY

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle de langue française pour spaCy
python -m spacy download fr_core_news_sm
```

## 💻 Utilisation

### Lancer le Dashboard NETY

```bash
# Lancer uniquement l'interface Dashboard
python app/main.py

# OU lancer le système complet (Dashboard + Backend NETY)
python run.py
```

### Fonctionnalités du Dashboard

Le **NETY Dashboard** propose trois pages principales :

1. **📊 Dashboard** - Contrôle de l'IA et visualisation des modules
   - Bouton toggle pour démarrer/arrêter l'IA NETY
   - Affichage de l'état des 4 modules principaux
   - Indicateur de statut en temps réel

2. **🔧 Admin Room** - Communication avec l'IA
   - Chatbox pour messages simples
   - Prompt Expeditor pour requêtes complexes
   - Historique des conversations

3. **📊 Logs/Terminal** - Surveillance des actions (NOUVEAU! 🆕)
   - Affichage temps réel des logs de l'IA
   - Style terminal noir/vert
   - Timestamps sur chaque action
   - Boutons Rafraîchir et Effacer
   - Auto-scroll vers les derniers logs
   - Limite de 1000 entrées

Pour plus de détails, consultez le [Guide Utilisateur](documentation/guide_utilisateur.md).

## 📊 Collecte de Données

Les données seront collectées à partir de :
- Bases de données publiques
- APIs ouvertes
- Données Open Source

## 🧠 Modèles Utilisés

### Traitement de Texte (RNN)
- Architecture : LSTM (Long Short-Term Memory)
- Framework : PyTorch
- Prétraitement : spaCy pour le français

### Traitement d'Images (CNN)
- Architecture : Réseau de neurones convolutionnel multicouche
- Framework : PyTorch
- Prétraitement : normalisation et augmentation

### Traitement Audio (STT)
- Architecture : CNN pour spectrogrammes
- Framework : TensorFlow/Keras
- Prétraitement : librosa pour l'extraction de features

## 🔧 Développement

### Linting

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Tests

```bash
pytest
```

## 📚 Documentation

Consultez le dossier `documentation/` pour plus de détails sur :
- La structure du projet
- Les modèles RNN, CNN et STT
- Les guides d'intégration

## 🔒 Sécurité

Voir [SECURITY.md](SECURITY.md) pour les politiques de sécurité.

## 📄 Licence

Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues ou des pull requests.

## 📧 Contact

Pour toute question, contactez Raptor_ via GitHub.

---

*"On n'a jamais fini d'apprendre"* 🚀
