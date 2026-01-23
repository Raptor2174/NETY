# Guide de Contribution à NETY

Merci de votre intérêt pour contribuer à NETY ! 🎉

## Comment Contribuer

### Signaler des Bugs

Si vous trouvez un bug, veuillez :

1. Vérifier qu'il n'a pas déjà été signalé dans les [Issues](https://github.com/Raptor2174/NETY/issues)
2. Créer une nouvelle issue avec :
   - Un titre descriptif
   - Les étapes pour reproduire le bug
   - Le comportement attendu vs le comportement actuel
   - Votre environnement (OS, version Python, etc.)

### Proposer des Améliorations

Pour proposer une nouvelle fonctionnalité :

1. Ouvrez une issue pour discuter de votre idée
2. Attendez les retours avant de commencer le développement
3. Assurez-vous que votre proposition s'aligne avec les objectifs du projet

### Soumettre des Pull Requests

1. **Fork** le repository
2. **Créez une branche** pour votre fonctionnalité :
   ```bash
   git checkout -b feature/ma-super-fonctionnalite
   ```
3. **Commitez** vos changements :
   ```bash
   git commit -m "Ajout de ma super fonctionnalité"
   ```
4. **Poussez** vers votre fork :
   ```bash
   git push origin feature/ma-super-fonctionnalite
   ```
5. **Ouvrez une Pull Request** avec une description détaillée

## Standards de Code

### Style Python

- Suivez la [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Utilisez des noms de variables descriptifs
- Ajoutez des docstrings à toutes les fonctions et classes
- Limitez les lignes à 127 caractères maximum

### Tests

- Écrivez des tests pour toute nouvelle fonctionnalité
- Assurez-vous que tous les tests passent avant de soumettre
- Utilisez `pytest` pour exécuter les tests :
  ```bash
  pytest
  ```

### Linting

Avant de soumettre, exécutez :

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Structure des Commits

Utilisez des messages de commit clairs et descriptifs :

- `feat: ` pour une nouvelle fonctionnalité
- `fix: ` pour une correction de bug
- `docs: ` pour la documentation
- `style: ` pour le formatage
- `refactor: ` pour le refactoring
- `test: ` pour les tests
- `chore: ` pour les tâches de maintenance

Exemple :
```
feat: Ajout du support pour l'audio en temps réel
```

## Documentation

- Mettez à jour la documentation si nécessaire
- Ajoutez des commentaires pour le code complexe
- Utilisez des docstrings au format [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

Exemple :
```python
def ma_fonction(param1: str, param2: int) -> bool:
    """
    Description courte de la fonction.
    
    Description plus détaillée si nécessaire.
    
    Args:
        param1: Description du premier paramètre
        param2: Description du second paramètre
        
    Returns:
        Description de la valeur de retour
        
    Raises:
        ValueError: Quand et pourquoi cette exception est levée
    """
    pass
```

## Processus de Review

1. Un mainteneur examinera votre PR
2. Des changements peuvent être demandés
3. Une fois approuvée, votre PR sera mergée
4. Votre contribution sera créditée dans le projet

## Code de Conduite

- Soyez respectueux envers tous les contributeurs
- Acceptez les critiques constructives
- Concentrez-vous sur ce qui est meilleur pour le projet
- Faites preuve d'empathie envers les autres membres de la communauté

## Questions ?

N'hésitez pas à ouvrir une issue avec le tag `question` si vous avez besoin d'aide !

---

Merci de contribuer à NETY ! 🚀
