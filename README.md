# Deep-Learning-Cardiology-Service
Heart attack detection system developed for a cardiology service. This project implements and compares MLP, CNN, and RNN architectures using the ECG200 dataset.

##  Structure du projet

* **`core/data_loader.py`** : Charge les données (ECG200), nettoie les labels (0/1) et normalise les signaux.

* **`models/`** : Contient les squelettes vides pour chaque archi :
    * `mlp.py` 
    * `cnn.py` 
    * `rnn.py` 

* **`main.py`** : C'est le seul fichier qu'on lance pour l'entraînement.

* **`data/`** : Présence des fichiers `.tsv` ici .

* **`production/`** : Dossier réservé au TP de **M. Hassenforder**. Contient la partie DevOps (Docker, API, Maven). C'est ici qu'on mettra le nécessaire pour le déploiement.

## Comment l'utiliser ?

1. **Ton modèle** : Va dans `models/` et remplis ton fichier. Garde bien le nom de la fonction et l'input.

2. **Test** : Dans `main.py`, tout en bas, décommente la ligne de ton modèle 
(ex: `run_experiment("cnn")`).

3. **Run** : Lance la commande `python main.py` dans ton terminal.

##  Utilisation de l'IA

Conformément aux consignes du prof,j'ai utilisé un peu l'IA pour gagner du temps et pour m'aider à :
* Concevoir l'**architecture modulaire** du code.
* Gérer le **Reshaping** des données (2D pour le MLP, 3D pour CNN/RNN).
* Coder proprement le chargement des fichiers TSV via **NumPy/Pandas**.

