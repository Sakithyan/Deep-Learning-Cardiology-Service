# main.py
from core.data_loader import load_ecg_dataset
from models.mlp import build_mlp
from models.cnn import build_cnn
from models.rnn import build_rnn
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _ensure_output_dir():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def _plot_training_curves(history, model_type, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="validation")
    ax1.set_title(f"{model_type.upper()} accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="validation")
    ax2.set_title(f"{model_type.upper()} loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model_type}_training_curves.png"), dpi=150)
    plt.close(fig)

def _plot_confusion_matrix(y_true, y_pred, model_type, output_dir):
    cm = np.zeros((2, 2), dtype=int)
    for target, pred in zip(y_true, y_pred):
        cm[int(target), int(pred)] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_type.upper()} confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pas malade", "malade"])
    ax.set_yticklabels(["pas malade", "malade"])

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model_type}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

def _plot_score_distribution(y_true, y_prob, model_type, output_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob[y_true == 0], bins=15, alpha=0.7, label="pas malade", color="tab:blue")
    ax.hist(y_prob[y_true == 1], bins=15, alpha=0.7, label="malade", color="tab:red")
    ax.set_title(f"{model_type.upper()} score distribution")
    ax.set_xlabel("score malade")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model_type}_score_distribution.png"), dpi=150)
    plt.close(fig)

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
    history = model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2)

    # 4. Évaluation
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    output_dir = _ensure_output_dir()
    _plot_training_curves(history, model_type, output_dir)
    _plot_confusion_matrix(y_test, y_pred, model_type, output_dir)
    _plot_score_distribution(y_test, y_prob, model_type, output_dir)

    print(f"Result {model_type} - Accuracy: {acc:.4f}")
    print(f"Graphs saved in {output_dir}")

if __name__ == "__main__":
    # Décommenter le modèle à tester
     run_experiment("mlp")
    # run_experiment("cnn")
    # run_experiment("rnn")