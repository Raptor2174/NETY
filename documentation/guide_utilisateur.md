# NETY Dashboard - Guide d'Utilisation Complet

## 🎯 Vue d'ensemble

Le NETY Dashboard est une interface graphique complète pour interagir avec l'IA NETY. Il propose trois pages principales accessibles via une barre de navigation.

## 📱 Pages Disponibles

### 1. Dashboard (Page d'accueil)
**Objectif**: Contrôler l'état de l'IA et visualiser les modules

**Fonctionnalités**:
- 🟢 **Bouton Toggle IA**: Démarrer/Arrêter l'IA NETY
- 📊 **État des Modules**: Visualisation en temps réel de 4 modules:
  - CCM (Cortex Cérébral) - TPM
  - LCM (Cortex Limbique) - ESM
  - BCM (Tronc Cérébral) - IMCM
  - CBM (Cervelet) - LM
- 🔴/🟢 **Indicateur de statut**: Affiche si l'IA est en marche ou arrêtée

**Utilisation**:
1. Cliquez sur "▶ Démarrer l'IA" pour activer le système
2. L'indicateur passe de "⚫ IA ARRÊTÉE" à "🟢 IA EN MARCHE"
3. Le bouton devient "⏸ Arrêter l'IA"
4. Les modules s'affichent avec leur statut (running, idle, inactive)

### 2. Admin Room
**Objectif**: Communiquer avec l'IA via chatbox ou prompts

**Fonctionnalités**:
- 💬 **Chatbox**: Conversation simple avec timestamps
- 🤖 **Prompt Expeditor**: Envoyer des prompts complexes à l'IA
- 📝 **Historique**: Affichage de tous les messages envoyés
- 🟢 **Indicateur de statut**: Rappel de l'état de l'IA

**Utilisation**:
1. **Via Chatbox**:
   - Tapez votre message dans le champ texte
   - Appuyez sur Enter ou cliquez "Envoyer"
   - Le message s'affiche avec un timestamp
2. **Via Prompt Expeditor**:
   - Écrivez un prompt dans la zone de texte
   - Cliquez "Envoyer vers l'IA"
   - La réponse s'affiche dans la chatbox

**Note**: L'IA doit être démarrée pour traiter les prompts

### 3. Logs / Terminal (NOUVEAU! 🆕)
**Objectif**: Visualiser en temps réel toutes les actions de l'IA

**Fonctionnalités**:
- 📊 **Affichage Terminal**: Style noir/vert type "Matrix"
- ⏱️ **Timestamps**: Chaque log avec date et heure
- 🔄 **Rafraîchir**: Mettre à jour l'affichage
- 🗑️ **Effacer**: Supprimer tous les logs
- ✅ **Auto-scroll**: Défilement automatique vers les derniers logs
- 🟢 **Indicateur de statut**: État de l'IA

**Types de logs affichés**:
- 💡 Initialisation du système
- 🚀 Démarrage de l'IA
- ⏹️ Arrêt de l'IA
- 🤖 Traitement de prompt
- 📤 Envoi de données
- ✅ Succès d'opération
- ⚠️ Avertissements
- ❌ Erreurs

**Utilisation**:
1. Cliquez sur "Logs" dans la barre de navigation
2. Les logs apparaissent automatiquement
3. Utilisez "🔄 Rafraîchir" pour mettre à jour
4. Cochez "Auto-scroll" pour suivre les derniers logs
5. Cliquez "🗑️ Effacer" pour nettoyer l'affichage

**Format des logs**:
```
[YYYY-MM-DD HH:MM:SS] [ICÔNE] Message descriptif
```

Exemple:
```
[2026-01-28 17:20:51] 🚀 IA NETY démarrée
[2026-01-28 17:20:51] ✓ Initialisation des modules NETY...
[2026-01-28 17:20:51] 🤖 Traitement du prompt (15 caractères)
```

## 🔄 Workflow Recommandé

### Scénario 1: Démarrage et test de l'IA

1. **Dashboard**: Cliquez sur "▶ Démarrer l'IA"
2. **Logs**: Vérifiez que l'IA démarre correctement
3. **Admin Room**: Envoyez un prompt de test
4. **Logs**: Observez le traitement du prompt

### Scénario 2: Debug et surveillance

1. **Logs**: Ouvrez la page Logs
2. **Cochez Auto-scroll**: Pour suivre en temps réel
3. **Dashboard/Admin Room**: Effectuez vos actions
4. **Logs**: Les actions s'affichent automatiquement
5. **Rafraîchir**: Si besoin de recharger

### Scénario 3: Nettoyage

1. **Logs**: Cliquez "🗑️ Effacer"
2. Un log "🗑️ Logs effacés" apparaît
3. Historique propre pour nouvelle session

## 🎨 Raccourcis Clavier

- **Enter** dans Chatbox: Envoyer le message
- **Clic droit**: (Fonctionnalité future)

## ⚙️ Configuration

### Limite de Logs
- Maximum: **1000 entrées**
- Comportement: FIFO (les plus anciens sont supprimés)
- Stockage: Mémoire uniquement (non persistant)

### Modules Surveillés
Les 4 modules affichés dans le Dashboard:
1. **CCM - TPM** (Text Processing Module)
2. **LCM - ESM** (Emotion Simulation Module)
3. **BCM - IMCM** (Inter-Module Communication Module)
4. **CBM - LM** (Learning Module)

## 🚀 Lancement de l'Application

```bash
# Depuis la racine du projet
cd /home/runner/work/NETY/NETY

# Lancer le dashboard
python3 app/main.py

# OU lancer le système complet (Dashboard + Backend NETY)
python3 run.py
```

## 📚 Documentation Supplémentaire

- **Architecture**: `documentation/logs_page.md`
- **Mockups UI**: `documentation/ui_mockup_with_logs.md`
- **README Principal**: `README.md`

## 🐛 Dépannage

### L'IA ne démarre pas
1. Vérifiez dans les Logs s'il y a des erreurs
2. Essayez d'arrêter puis redémarrer
3. Vérifiez que tkinter est installé

### Les logs n'apparaissent pas
1. Cliquez sur "🔄 Rafraîchir"
2. Vérifiez que vous êtes sur la page Logs
3. Vérifiez qu'il y a eu des actions (démarrage IA, prompts, etc.)

### Le bouton toggle ne fonctionne pas
1. Vérifiez dans les Logs l'état réel
2. Rafraîchissez la page Dashboard
3. Redémarrez l'application si nécessaire

## 🔐 Sécurité

- Les logs peuvent contenir des informations sensibles
- Utilisez "🗑️ Effacer" régulièrement
- Les logs ne sont pas sauvegardés sur disque

## 🎯 Prochaines Améliorations

- [ ] Sauvegarde des logs dans un fichier
- [ ] Filtrage par type de log
- [ ] Recherche dans les logs
- [ ] Export CSV/TXT
- [ ] Rafraîchissement automatique
- [ ] Couleurs par niveau (DEBUG, INFO, WARNING, ERROR)
- [ ] Logs persistants entre sessions

## 📞 Support

Pour toute question ou problème:
- Consultez la documentation dans `/documentation`
- Ouvrez une issue sur GitHub
- Contactez l'équipe via GitHub

---

**Version**: 0.1.4  
**Dernière mise à jour**: 2026-01-28  
**Auteur**: Raptor2174
