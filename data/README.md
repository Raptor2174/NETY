# NETY Data Directory

Ce répertoire contient le **stockage et la gestion des données** pour le système NETY.

## Structure

- **raw/** : Données brutes non traitées
- **processed/** : Données prétraitées et prêtes à l'utilisation
- **models/** : Modèles ML entraînés et sérialisés
- **logs/** : Fichiers de log et historique

## Notes

- Ce dossier est distinct de `nety/` qui contient la **logique et le fonctionnement** du système
- Les données sensibles ne doivent PAS être commitées (ajoutées au .gitignore)
