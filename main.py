# main.py
from core.data_loader import load_ecg_dataset
from models.mlp import build_mlp
from models.cnn import build_cnn
from models.rnn import build_rnn

def run_experiment(model_type="mlp"):
    # 1. Chargement des données
    # True si CNN/RNN (3D), False si MLP (2D)
    is_3d = True if model_type in ["cnn", "rnn"] else False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)
    
    # 2. Construction du modèle
    input_shape = x_train.shape[1:]
    
    if model_type == "mlp":
        model = build_mlp(input_shape)
    elif model_type == "cnn":
        model = build_cnn(input_shape)
    elif model_type == "rnn":
        model = build_rnn(input_shape)

    # 3. Entraînement
    # JUSTIFICATION DES PARAMÈTRES :
    # - epochs=5 : Faible pour tester le code. À augmenter pour les résultats finaux.
    # - batch_size=16 : Adapté à la petite taille du dataset (100 signaux) pour avoir assez de mises à jour.
    # - validation_split=0.2 : 20% des données servent de test "interne" pour surveiller l'overfitting.

    model.summary()
    model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    # 4. Évaluation
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Result {model_type} - Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Décommenter le modèle à tester
     run_experiment("mlp")
    # run_experiment("cnn")
    # run_experiment("rnn")