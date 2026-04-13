# main.py
from core.data_loader import load_ecg_dataset
from models.mlp import build_mlp
from models.cnn import build_cnn, build_cnn_naive, build_cnn_regularized
from models.rnn import build_rnn
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)
    return "outputs"


def save_training_curves(history, model_type, output_dir, tag=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train")
    ax1.plot(history.history["val_accuracy"], label="validation")
    ax1.set_title(f"{model_type.upper()} {tag} accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="train")
    ax2.plot(history.history["val_loss"], label="validation")
    ax2.set_title(f"{model_type.upper()} {tag} loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    tag_file = f"_{tag}" if tag else ""
    graph_path = os.path.join(output_dir, f"{model_type}{tag_file}_training_curves.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


def save_confusion_matrix(y_true, y_pred, model_type, output_dir, tag=""):
    cm = np.zeros((2, 2), dtype=int)
    for target, pred in zip(y_true, y_pred):
        cm[int(target), int(pred)] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{model_type.upper()} {tag} confusion matrix")
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
    tag_file = f"_{tag}" if tag else ""
    graph_path = os.path.join(output_dir, f"{model_type}{tag_file}_confusion_matrix.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


def save_score_distribution(y_true, y_prob, model_type, output_dir, tag=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob[y_true == 0], bins=15, alpha=0.7, label="pas malade", color="tab:blue")
    ax.hist(y_prob[y_true == 1], bins=15, alpha=0.7, label="malade", color="tab:red")
    ax.set_title(f"{model_type.upper()} {tag} score distribution")
    ax.set_xlabel("score malade")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    tag_file = f"_{tag}" if tag else ""
    graph_path = os.path.join(output_dir, f"{model_type}{tag_file}_score_distribution.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


def open_graphs(graph_paths):
    if not hasattr(os, "startfile"):
        return
    for graph_path in graph_paths:
        try:
            os.startfile(os.path.abspath(graph_path))
        except OSError:
            pass


def save_cnn_variant_comparison(history_naive, history_regularized, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history_naive.history["val_accuracy"], label="naive", color="tab:orange")
    ax1.plot(history_regularized.history["val_accuracy"], label="regularized", color="tab:green")
    ax1.set_title("CNN variants - validation accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history_naive.history["val_loss"], label="naive", color="tab:orange")
    ax2.plot(history_regularized.history["val_loss"], label="regularized", color="tab:green")
    ax2.set_title("CNN variants - validation loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    graph_path = os.path.join(output_dir, "cnn_variant_comparison.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


def train_model(model, x_train, y_train, epochs, batch_size):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )


def pick_model(model_type, input_shape, cnn_variant):
    if model_type == "mlp":
        return build_mlp(input_shape)
    if model_type == "rnn":
        return build_rnn(input_shape)
    if cnn_variant == "naive":
        return build_cnn_naive(input_shape)
    if cnn_variant == "regularized":
        return build_cnn_regularized(input_shape)
    return build_cnn(input_shape)

def run_experiment(model_type="mlp", cnn_variant="regularized", compare_cnn=False, epochs=35, batch_size=16, show_graphs=False):
    # Donnees 3D pour CNN/RNN, 2D pour MLP.
    is_3d = True if model_type in ["cnn", "rnn"] else False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)

    input_shape = x_train.shape[1:]

    if compare_cnn and model_type != "cnn":
        raise ValueError("--compare-cnn peut etre utilise seulement avec --model cnn")

    if compare_cnn:
        output_dir = ensure_output_dir()
        graph_paths = []

        model_naive = build_cnn_naive(input_shape)
        history_naive = train_model(model_naive, x_train, y_train, epochs=epochs, batch_size=batch_size)
        loss_naive, acc_naive = model_naive.evaluate(x_test, y_test, verbose=0)
        y_prob_naive = model_naive.predict(x_test, verbose=0).reshape(-1)
        y_pred_naive = (y_prob_naive >= 0.5).astype(int)

        graph_paths.append(save_training_curves(history_naive, "cnn", output_dir, tag="naive"))
        graph_paths.append(save_confusion_matrix(y_test, y_pred_naive, "cnn", output_dir, tag="naive"))
        graph_paths.append(save_score_distribution(y_test, y_prob_naive, "cnn", output_dir, tag="naive"))

        model_reg = build_cnn_regularized(input_shape)
        history_reg = train_model(model_reg, x_train, y_train, epochs=epochs, batch_size=batch_size)
        loss_reg, acc_reg = model_reg.evaluate(x_test, y_test, verbose=0)
        y_prob_reg = model_reg.predict(x_test, verbose=0).reshape(-1)
        y_pred_reg = (y_prob_reg >= 0.5).astype(int)

        graph_paths.append(save_training_curves(history_reg, "cnn", output_dir, tag="regularized"))
        graph_paths.append(save_confusion_matrix(y_test, y_pred_reg, "cnn", output_dir, tag="regularized"))
        graph_paths.append(save_score_distribution(y_test, y_prob_reg, "cnn", output_dir, tag="regularized"))
        graph_paths.append(save_cnn_variant_comparison(history_naive, history_reg, output_dir))

        if show_graphs:
            open_graphs(graph_paths)

        print(f"Result cnn naive - Accuracy: {acc_naive:.4f} | Loss: {loss_naive:.4f}")
        print(f"Result cnn regularized - Accuracy: {acc_reg:.4f} | Loss: {loss_reg:.4f}")
        print(f"Graphs saved in {output_dir}")
        for graph_path in graph_paths:
            print(graph_path)
        return

    model = pick_model(model_type, input_shape, cnn_variant)

    model.summary()
    history = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    output_dir = ensure_output_dir()
    tag = cnn_variant if model_type == "cnn" else ""
    graph_paths = []
    graph_paths.append(save_training_curves(history, model_type, output_dir, tag=tag))
    graph_paths.append(save_confusion_matrix(y_test, y_pred, model_type, output_dir, tag=tag))
    graph_paths.append(save_score_distribution(y_test, y_prob, model_type, output_dir, tag=tag))

    if show_graphs:
        open_graphs(graph_paths)

    if tag:
        print(f"Result {model_type} ({tag}) - Accuracy: {acc:.4f} | Loss: {loss:.4f}")
    else:
        print(f"Result {model_type} - Accuracy: {acc:.4f} | Loss: {loss:.4f}")
    print(f"Graphs saved in {output_dir}")
    for graph_path in graph_paths:
        print(graph_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn", "rnn"], default="cnn")
    parser.add_argument("--cnn-variant", choices=["naive", "regularized"], default="regularized")
    parser.add_argument("--compare-cnn", action="store_true")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-graphs", action="store_true")
    args = parser.parse_args()
    run_experiment(
        args.model,
        cnn_variant=args.cnn_variant,
        compare_cnn=args.compare_cnn,
        epochs=args.epochs,
        batch_size=args.batch_size,
        show_graphs=args.show_graphs,
    )