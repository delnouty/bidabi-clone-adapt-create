# Instructions pour cloner et utiliser le dépôt dans Jupyter / Colab

1. Dans une cellule Jupyter ou Google Colab, exécutez :
   ```
   !git clone https://github.com/delnouty/bidabi-clone-adapt-create.git
   ```
2. Puis entrez dans le dossier :
```
%cd bidabi-clone-adapt-create
```
3. Installer les dépendances dans le notebook
   ```
   !pip install -r requirements.txt

   ```
   Sinon, installez les bibliothèques nécessaires au fur et à mesure, par exemple :
   ```
   !pip install pillow pandas torch torchvision

   ```
4. Vérifier la structure du projet
   ```
   from pathlib import Path
   for p in Path(".").rglob("*"):
       print(p)
   ```
   Vous devriez voir les dossiers :
- src — code à adapter
- data — votre jeu de données
- notebooks — notebooks d’exemples
- reports — rapport final

