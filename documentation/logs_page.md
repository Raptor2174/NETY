# Page Logs/Terminal - Documentation

## Vue d'ensemble

La page **Logs/Terminal** affiche en temps réel toutes les actions de l'IA NETY dans une interface de type terminal. Cette page permet de suivre l'activité du système et de déboguer les problèmes.

## Fonctionnalités

### Affichage des Logs
- **Interface terminal noir/vert** : Style terminal classique pour une meilleure lisibilité
- **Horodatage** : Chaque entrée de log est préfixée avec un timestamp `[YYYY-MM-DD HH:MM:SS]`
- **Icônes émoji** : Utilisation d'icônes pour identifier rapidement le type d'action
- **Auto-scroll** : Défilement automatique vers les derniers logs

### Boutons de Contrôle

1. **🔄 Rafraîchir** : Actualise l'affichage des logs
2. **🗑️ Effacer** : Supprime tous les logs
3. **Auto-scroll** : Case à cocher pour activer/désactiver le défilement automatique

### Indicateur de Statut
- **● IA EN MARCHE** : L'IA est active et surveille les actions
- **○ IA ARRÊTÉE** : L'IA est inactive

## Types de Logs

### Logs Système
- 💡 **Initialisation** : Démarrage du système Dashboard
- 🗑️ **Effacement** : Logs effacés par l'utilisateur

### Logs de l'IA
- 🚀 **Démarrage** : L'IA NETY démarre
- ⏹️ **Arrêt** : L'IA NETY s'arrête
- ✓ **Initialisation** : Modules NETY en cours d'initialisation
- 🤖 **Traitement** : Un prompt est en cours de traitement
- ✅ **Succès** : Traitement terminé avec succès
- ⚠️ **Avertissement** : Tentative d'action avec IA arrêtée

### Logs de Communication
- 📤 **Envoi** : Données envoyées vers NETY
- ❌ **Erreur** : Échec d'une opération

## Implémentation Technique

### Fichier : `app/ui/logs.py`

La classe `LogsView` gère l'affichage de la page logs :
- Utilise `scrolledtext.ScrolledText` pour l'affichage
- Rafraîchit les logs depuis `NetyAdapter.get_logs()`
- Supporte l'auto-scroll optionnel

### Fichier : `app/services/nety_adapter.py`

Le `NetyAdapter` a été enrichi avec :
- `logs: List[str]` : Liste des entrées de log
- `_add_log(message)` : Ajoute une entrée horodatée
- `get_logs()` : Retourne une copie des logs
- `clear_logs()` : Efface tous les logs
- Limitation à 1000 entrées maximum

### Navigation

La page Logs est accessible via :
- **Taskbar** : Bouton "Logs" dans la barre de navigation
- **Route** : `page_id = "logs"`

## Exemple d'Utilisation

1. Lancer l'application NETY Dashboard
2. Cliquer sur le bouton "Logs" dans la taskbar
3. Démarrer l'IA depuis le Dashboard
4. Observer les logs en temps réel
5. Envoyer un prompt depuis Admin Room
6. Voir les actions se refléter dans les logs

## Limitations

- **Mémoire** : Maximum 1000 entrées de log en mémoire
- **Persistance** : Les logs ne sont pas sauvegardés (en mémoire uniquement)
- **Performance** : Rafraîchissement manuel ou via bouton

## Améliorations Futures

- [ ] Sauvegarde des logs dans un fichier
- [ ] Filtrage par type de log (système, IA, erreurs)
- [ ] Recherche dans les logs
- [ ] Export des logs (CSV, TXT)
- [ ] Rafraîchissement automatique en temps réel
- [ ] Couleurs différentes par type de log
- [ ] Niveaux de log (DEBUG, INFO, WARNING, ERROR)
