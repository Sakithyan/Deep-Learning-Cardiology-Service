import argparse
import csv
import logging
import os
import random
import time
import warnings

# Reduce TensorFlow logs before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

from core.data_loader import load_ecg_dataset
<<<<<<< HEAD
from models.cnn import build_cnn, build_cnn_naive, build_cnn_regularized
from models.mlp import build_mlp, build_mlp_v1, build_mlp_v2, build_mlp_final
from models.rnn import build_rnn, build_rnn_v1, build_rnn_v2, build_rnn_final


def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)
    return "outputs"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def default_variant_for_model(model_type):
    if model_type == "cnn":
        return "regularized"
    return "final"


def resolve_is_3d(model_type):
    return model_type in ["cnn", "rnn"]


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


def save_mlp_final_diagnostic(y_prob, y_test, output_dir):
    # ROC + score distribution to preserve MLP final diagnostic analysis.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="random")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("MLP final ROC")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    healthy_scores = y_prob[y_test == 0]
    sick_scores = y_prob[y_test == 1]
    axes[1].hist(healthy_scores, bins=30, alpha=0.7, label="pas malade", color="tab:blue", edgecolor="black")
    axes[1].hist(sick_scores, bins=30, alpha=0.7, label="malade", color="tab:red", edgecolor="black")
    axes[1].axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="threshold=0.5")
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("count")
    axes[1].set_title("MLP final score distribution")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    graph_path = os.path.join(output_dir, "mlp_final_diagnostic.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)

    return graph_path, roc_auc


def open_graphs(graph_paths):
    if not hasattr(os, "startfile"):
        return
    for graph_path in graph_paths:
        try:
            os.startfile(os.path.abspath(graph_path))
        except OSError:
            pass


def save_variant_comparison(history_a, history_b, model_type, output_dir, label_a="naive", label_b="regularized"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history_a.history["val_accuracy"], label=label_a, color="tab:orange")
    ax1.plot(history_b.history["val_accuracy"], label=label_b, color="tab:green")
    ax1.set_title(f"{model_type.upper()} variants - validation accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history_a.history["val_loss"], label=label_a, color="tab:orange")
    ax2.plot(history_b.history["val_loss"], label=label_b, color="tab:green")
    ax2.set_title(f"{model_type.upper()} variants - validation loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    graph_path = os.path.join(output_dir, f"{model_type}_variant_comparison.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    return graph_path


def save_repeats_summary_plot(summary_rows, output_dir, model_type):
    labels = [row["variant"] for row in summary_rows]
    acc_mean = [row["acc_mean"] for row in summary_rows]
    acc_std = [row["acc_std"] for row in summary_rows]
    train_mean = [row["train_time_mean"] for row in summary_rows]
    train_std = [row["train_time_std"] for row in summary_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(labels, acc_mean, yerr=acc_std, capsize=5, color=["tab:orange", "tab:green", "tab:blue"][: len(labels)])
    ax1.set_title(f"{model_type.upper()} repeats - test accuracy")
    ax1.set_ylabel("accuracy")
    ax1.grid(True, axis="y")

    ax2.bar(labels, train_mean, yerr=train_std, capsize=5, color=["tab:orange", "tab:green", "tab:blue"][: len(labels)])
    ax2.set_title(f"{model_type.upper()} repeats - train time (s)")
    ax2.set_ylabel("seconds")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    graph_path = os.path.join(output_dir, f"{model_type}_repeats_summary.png")
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
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - start
    return history, train_time


def pick_model(model_type, input_shape, variant=None):
    variant = variant or default_variant_for_model(model_type)

    if model_type == "mlp":
        if variant == "naive":
            return build_mlp_v1(input_shape)
        if variant == "regularized":
            return build_mlp_v2(input_shape)
        return build_mlp_final(input_shape)

    if model_type == "rnn":
        if variant == "naive":
            return build_rnn_v1(input_shape)
        if variant == "regularized":
            return build_rnn_v2(input_shape)
        return build_rnn_final(input_shape)

    if variant == "naive":
        return build_cnn_naive(input_shape)
    if variant == "regularized":
        return build_cnn_regularized(input_shape)
    return build_cnn(input_shape)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_summary(rows):
    summary = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        acc = np.array([row["test_accuracy"] for row in variant_rows], dtype=float)
        loss = np.array([row["test_loss"] for row in variant_rows], dtype=float)
        train_time = np.array([row["train_time_s"] for row in variant_rows], dtype=float)
        infer_time = np.array([row["inference_time_per_sample_ms"] for row in variant_rows], dtype=float)
        params = np.array([row["n_parameters"] for row in variant_rows], dtype=float)
        size_mb = np.array([row["estimated_size_mb"] for row in variant_rows], dtype=float)

        summary.append(
            {
                "variant": variant,
                "acc_mean": float(acc.mean()),
                "acc_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
                "loss_mean": float(loss.mean()),
                "loss_std": float(loss.std(ddof=1) if len(loss) > 1 else 0.0),
                "train_time_mean": float(train_time.mean()),
                "train_time_std": float(train_time.std(ddof=1) if len(train_time) > 1 else 0.0),
                "infer_ms_mean": float(infer_time.mean()),
                "infer_ms_std": float(infer_time.std(ddof=1) if len(infer_time) > 1 else 0.0),
                "params_mean": float(params.mean()),
                "size_mb_mean": float(size_mb.mean()),
            }
        )
    return summary


def evaluate_model(model, x_test, y_test):
    metrics = model.evaluate(x_test, y_test, verbose=0)
    loss = metrics[0]
    acc = metrics[1] if len(metrics) > 1 else 0.0
    recall = metrics[2] if len(metrics) > 2 else None

    start = time.perf_counter()
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    infer_time = time.perf_counter() - start
    y_pred = (y_prob >= 0.5).astype(int)

    n_parameters = int(model.count_params())
    size_mb = (n_parameters * 4) / (1024 ** 2)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "recall": float(recall) if recall is not None else None,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "n_parameters": n_parameters,
        "estimated_size_mb": float(size_mb),
        "inference_time_per_sample_ms": float((infer_time / len(x_test)) * 1000.0),
    }


def run_repeats(model_type, variants, repeats=5, base_seed=42, epochs=35, batch_size=16, show_graphs=False):
    output_dir = ensure_output_dir()
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=resolve_is_3d(model_type))
    input_shape = x_train.shape[1:]

    rows = []
    graph_paths = []
    first_histories = {}

    for run_idx in range(repeats):
        seed = base_seed + run_idx
        seed_all(seed)
        print(f"Run {run_idx + 1}/{repeats} - seed={seed}")

        for variant in variants:
            model = pick_model(model_type, input_shape, variant)
            history, train_time = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
            result = evaluate_model(model, x_test, y_test)

            rows.append(
                {
                    "run": run_idx + 1,
                    "seed": seed,
                    "variant": variant,
                    "epochs_ran": int(len(history.history["loss"])),
                    "test_accuracy": result["acc"],
                    "test_loss": result["loss"],
                    "test_recall": result["recall"],
                    "n_parameters": result["n_parameters"],
                    "estimated_size_mb": result["estimated_size_mb"],
                    "train_time_s": float(train_time),
                    "inference_time_per_sample_ms": result["inference_time_per_sample_ms"],
                }
            )

            if run_idx == 0:
                graph_paths.append(save_training_curves(history, model_type, output_dir, tag=variant))
                graph_paths.append(save_confusion_matrix(y_test, result["y_pred"], model_type, output_dir, tag=variant))
                graph_paths.append(save_score_distribution(y_test, result["y_prob"], model_type, output_dir, tag=variant))
                first_histories[variant] = history

            msg = (
                f"  {variant}: acc={result['acc']:.4f} | loss={result['loss']:.4f} "
                f"| train={train_time:.2f}s | infer/sample={result['inference_time_per_sample_ms']:.3f}ms"
            )
            if result["recall"] is not None:
                msg += f" | recall={result['recall']:.4f}"
            print(msg)

    if "naive" in first_histories and "regularized" in first_histories:
        graph_paths.append(save_variant_comparison(first_histories["naive"], first_histories["regularized"], model_type, output_dir))

    summary_rows = compute_summary(rows)
    graph_paths.append(save_repeats_summary_plot(summary_rows, output_dir, model_type))

    detailed_csv = os.path.join(output_dir, f"{model_type}_repeats_detailed.csv")
    summary_csv = os.path.join(output_dir, f"{model_type}_repeats_summary.csv")
    write_csv(detailed_csv, rows)
    write_csv(summary_csv, summary_rows)

    print("\nSummary (mean +- std):")
    for row in summary_rows:
        print(
            f"- {row['variant']}: "
            f"acc={row['acc_mean']:.4f} +- {row['acc_std']:.4f}, "
            f"loss={row['loss_mean']:.4f} +- {row['loss_std']:.4f}, "
            f"train={row['train_time_mean']:.2f} +- {row['train_time_std']:.2f}s, "
            f"infer={row['infer_ms_mean']:.3f} +- {row['infer_ms_std']:.3f}ms/sample, "
            f"params={int(row['params_mean'])}, size~{row['size_mb_mean']:.4f}MB"
        )

    print("\nFiles generated:")
    print(detailed_csv)
    print(summary_csv)
    for graph_path in graph_paths:
        print(graph_path)

    if show_graphs:
        open_graphs(graph_paths)


def run_experiment(model_type="cnn", variant=None, compare_cnn=False, epochs=35, batch_size=16, show_graphs=False):
    is_3d = resolve_is_3d(model_type)
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)
    input_shape = x_train.shape[1:]

    if compare_cnn and model_type != "cnn":
        raise ValueError("--compare-cnn can be used only with --model cnn")
=======
from models.mlp import build_mlp
from models.cnn import build_cnn, build_cnn_naive, build_cnn_regularized
from models.rnn import build_rnn
import argparse
import csv
import os
import random
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

def ensure_output_dir():
    os.makedirs("outputs", exist_ok=True)
    return "outputs"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def save_repeats_summary_plot(summary_rows, output_dir):
    labels = [row["variant"] for row in summary_rows]
    acc_mean = [row["acc_mean"] for row in summary_rows]
    acc_std = [row["acc_std"] for row in summary_rows]
    train_mean = [row["train_time_mean"] for row in summary_rows]
    train_std = [row["train_time_std"] for row in summary_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(labels, acc_mean, yerr=acc_std, capsize=5, color=["tab:orange", "tab:green"][:len(labels)])
    ax1.set_title("CNN repeats - test accuracy")
    ax1.set_ylabel("accuracy")
    ax1.grid(True, axis="y")

    ax2.bar(labels, train_mean, yerr=train_std, capsize=5, color=["tab:orange", "tab:green"][:len(labels)])
    ax2.set_title("CNN repeats - train time (s)")
    ax2.set_ylabel("seconds")
    ax2.grid(True, axis="y")

    fig.tight_layout()
    graph_path = os.path.join(output_dir, "cnn_repeats_summary.png")
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
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - start
    return history, train_time


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


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_summary(rows):
    summary = []
    variants = sorted({row["variant"] for row in rows})
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        acc = np.array([row["test_accuracy"] for row in variant_rows], dtype=float)
        loss = np.array([row["test_loss"] for row in variant_rows], dtype=float)
        train_time = np.array([row["train_time_s"] for row in variant_rows], dtype=float)
        infer_time = np.array([row["inference_time_per_sample_ms"] for row in variant_rows], dtype=float)
        params = np.array([row["n_parameters"] for row in variant_rows], dtype=float)
        size_mb = np.array([row["estimated_size_mb"] for row in variant_rows], dtype=float)

        summary.append(
            {
                "variant": variant,
                "acc_mean": float(acc.mean()),
                "acc_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
                "loss_mean": float(loss.mean()),
                "loss_std": float(loss.std(ddof=1) if len(loss) > 1 else 0.0),
                "train_time_mean": float(train_time.mean()),
                "train_time_std": float(train_time.std(ddof=1) if len(train_time) > 1 else 0.0),
                "infer_ms_mean": float(infer_time.mean()),
                "infer_ms_std": float(infer_time.std(ddof=1) if len(infer_time) > 1 else 0.0),
                "params_mean": float(params.mean()),
                "size_mb_mean": float(size_mb.mean()),
            }
        )
    return summary


def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    start = time.perf_counter()
    y_prob = model.predict(x_test, verbose=0).reshape(-1)
    infer_time = time.perf_counter() - start
    y_pred = (y_prob >= 0.5).astype(int)

    n_parameters = int(model.count_params())
    size_mb = (n_parameters * 4) / (1024 ** 2)

    return {
        "loss": float(loss),
        "acc": float(acc),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "n_parameters": n_parameters,
        "estimated_size_mb": float(size_mb),
        "inference_time_per_sample_ms": float((infer_time / len(x_test)) * 1000.0),
    }


def run_cnn_repeats(variants, repeats=5, base_seed=42, epochs=35, batch_size=16, show_graphs=False):
    output_dir = ensure_output_dir()
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=True)
    input_shape = x_train.shape[1:]

    rows = []
    graph_paths = []
    first_histories = {}

    for run_idx in range(repeats):
        seed = base_seed + run_idx
        seed_all(seed)
        print(f"Run {run_idx + 1}/{repeats} - seed={seed}")

        for variant in variants:
            model = pick_model("cnn", input_shape, variant)
            history, train_time = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
            result = evaluate_model(model, x_test, y_test)

            rows.append(
                {
                    "run": run_idx + 1,
                    "seed": seed,
                    "variant": variant,
                    "epochs_ran": int(len(history.history["loss"])),
                    "test_accuracy": result["acc"],
                    "test_loss": result["loss"],
                    "n_parameters": result["n_parameters"],
                    "estimated_size_mb": result["estimated_size_mb"],
                    "train_time_s": float(train_time),
                    "inference_time_per_sample_ms": result["inference_time_per_sample_ms"],
                }
            )

            if run_idx == 0:
                graph_paths.append(save_training_curves(history, "cnn", output_dir, tag=variant))
                graph_paths.append(save_confusion_matrix(y_test, result["y_pred"], "cnn", output_dir, tag=variant))
                graph_paths.append(save_score_distribution(y_test, result["y_prob"], "cnn", output_dir, tag=variant))
                first_histories[variant] = history

            print(
                f"  {variant}: acc={result['acc']:.4f} | loss={result['loss']:.4f} "
                f"| train={train_time:.2f}s | infer/sample={result['inference_time_per_sample_ms']:.3f}ms"
            )

    if "naive" in first_histories and "regularized" in first_histories:
        graph_paths.append(
            save_cnn_variant_comparison(first_histories["naive"], first_histories["regularized"], output_dir)
        )

    summary_rows = compute_summary(rows)
    graph_paths.append(save_repeats_summary_plot(summary_rows, output_dir))

    detailed_csv = os.path.join(output_dir, "cnn_repeats_detailed.csv")
    summary_csv = os.path.join(output_dir, "cnn_repeats_summary.csv")
    write_csv(detailed_csv, rows)
    write_csv(summary_csv, summary_rows)

    print("\nSummary (mean +- std):")
    for row in summary_rows:
        print(
            f"- {row['variant']}: "
            f"acc={row['acc_mean']:.4f} +- {row['acc_std']:.4f}, "
            f"loss={row['loss_mean']:.4f} +- {row['loss_std']:.4f}, "
            f"train={row['train_time_mean']:.2f} +- {row['train_time_std']:.2f}s, "
            f"infer={row['infer_ms_mean']:.3f} +- {row['infer_ms_std']:.3f}ms/sample, "
            f"params={int(row['params_mean'])}, size~{row['size_mb_mean']:.4f}MB"
        )

    print("\nFiles generated:")
    print(detailed_csv)
    print(summary_csv)
    for graph_path in graph_paths:
        print(graph_path)

    if show_graphs:
        open_graphs(graph_paths)

def run_experiment(model_type="mlp", cnn_variant="regularized", compare_cnn=False, epochs=35, batch_size=16, show_graphs=False):
    # Donnees 3D pour CNN/RNN, 2D pour MLP.
    is_3d = True if model_type in ["cnn", "rnn"] else False
    x_train, x_test, y_train, y_test = load_ecg_dataset(is_3d=is_3d)

    input_shape = x_train.shape[1:]

    if compare_cnn and model_type != "cnn":
        raise ValueError("--compare-cnn peut etre utilise seulement avec --model cnn")
>>>>>>> origin/cnn

    if compare_cnn:
        output_dir = ensure_output_dir()
        graph_paths = []

<<<<<<< HEAD
        model_naive = pick_model("cnn", input_shape, "naive")
=======
        model_naive = build_cnn_naive(input_shape)
>>>>>>> origin/cnn
        history_naive, train_time_naive = train_model(model_naive, x_train, y_train, epochs=epochs, batch_size=batch_size)
        result_naive = evaluate_model(model_naive, x_test, y_test)

        graph_paths.append(save_training_curves(history_naive, "cnn", output_dir, tag="naive"))
        graph_paths.append(save_confusion_matrix(y_test, result_naive["y_pred"], "cnn", output_dir, tag="naive"))
        graph_paths.append(save_score_distribution(y_test, result_naive["y_prob"], "cnn", output_dir, tag="naive"))

<<<<<<< HEAD
        model_reg = pick_model("cnn", input_shape, "regularized")
=======
        model_reg = build_cnn_regularized(input_shape)
>>>>>>> origin/cnn
        history_reg, train_time_reg = train_model(model_reg, x_train, y_train, epochs=epochs, batch_size=batch_size)
        result_reg = evaluate_model(model_reg, x_test, y_test)

        graph_paths.append(save_training_curves(history_reg, "cnn", output_dir, tag="regularized"))
        graph_paths.append(save_confusion_matrix(y_test, result_reg["y_pred"], "cnn", output_dir, tag="regularized"))
        graph_paths.append(save_score_distribution(y_test, result_reg["y_prob"], "cnn", output_dir, tag="regularized"))
<<<<<<< HEAD
        graph_paths.append(save_variant_comparison(history_naive, history_reg, "cnn", output_dir))
=======
        graph_paths.append(save_cnn_variant_comparison(history_naive, history_reg, output_dir))
>>>>>>> origin/cnn

        if show_graphs:
            open_graphs(graph_paths)

        print(
            "Result cnn naive - "
            f"Accuracy: {result_naive['acc']:.4f} | Loss: {result_naive['loss']:.4f} | "
<<<<<<< HEAD
            f"Train: {train_time_naive:.2f}s | Infer/sample: {result_naive['inference_time_per_sample_ms']:.3f}ms | "
=======
            f"Train: {train_time_naive:.2f}s | "
            f"Infer/sample: {result_naive['inference_time_per_sample_ms']:.3f}ms | "
>>>>>>> origin/cnn
            f"Params: {result_naive['n_parameters']} | Size~{result_naive['estimated_size_mb']:.4f}MB"
        )
        print(
            "Result cnn regularized - "
            f"Accuracy: {result_reg['acc']:.4f} | Loss: {result_reg['loss']:.4f} | "
<<<<<<< HEAD
            f"Train: {train_time_reg:.2f}s | Infer/sample: {result_reg['inference_time_per_sample_ms']:.3f}ms | "
=======
            f"Train: {train_time_reg:.2f}s | "
            f"Infer/sample: {result_reg['inference_time_per_sample_ms']:.3f}ms | "
>>>>>>> origin/cnn
            f"Params: {result_reg['n_parameters']} | Size~{result_reg['estimated_size_mb']:.4f}MB"
        )
        print(f"Graphs saved in {output_dir}")
        for graph_path in graph_paths:
            print(graph_path)
        return

<<<<<<< HEAD
    selected_variant = variant or default_variant_for_model(model_type)
    model = pick_model(model_type, input_shape, selected_variant)
=======
    model = pick_model(model_type, input_shape, cnn_variant)
>>>>>>> origin/cnn

    model.summary()
    history, train_time = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
    result = evaluate_model(model, x_test, y_test)
<<<<<<< HEAD

    output_dir = ensure_output_dir()
    tag = selected_variant
    graph_paths = [
        save_training_curves(history, model_type, output_dir, tag=tag),
        save_confusion_matrix(y_test, result["y_pred"], model_type, output_dir, tag=tag),
        save_score_distribution(y_test, result["y_prob"], model_type, output_dir, tag=tag),
    ]

    if model_type == "mlp" and selected_variant == "final":
        roc_path, roc_auc = save_mlp_final_diagnostic(result["y_prob"], y_test, output_dir)
        graph_paths.append(roc_path)
        print(f"MLP final ROC AUC: {roc_auc:.4f}")

    if show_graphs:
        open_graphs(graph_paths)

    msg = (
        f"Result {model_type} ({selected_variant}) - Accuracy: {result['acc']:.4f} | Loss: {result['loss']:.4f} | "
        f"Train: {train_time:.2f}s | Infer/sample: {result['inference_time_per_sample_ms']:.3f}ms | "
        f"Params: {result['n_parameters']} | Size~{result['estimated_size_mb']:.4f}MB"
    )
    if result["recall"] is not None:
        msg += f" | Recall: {result['recall']:.4f}"
    print(msg)

    print(f"Graphs saved in {output_dir}")
    for graph_path in graph_paths:
        print(graph_path)

=======

    output_dir = ensure_output_dir()
    tag = cnn_variant if model_type == "cnn" else ""
    graph_paths = []
    graph_paths.append(save_training_curves(history, model_type, output_dir, tag=tag))
    graph_paths.append(save_confusion_matrix(y_test, result["y_pred"], model_type, output_dir, tag=tag))
    graph_paths.append(save_score_distribution(y_test, result["y_prob"], model_type, output_dir, tag=tag))

    if show_graphs:
        open_graphs(graph_paths)

    if tag:
        print(
            f"Result {model_type} ({tag}) - Accuracy: {result['acc']:.4f} | Loss: {result['loss']:.4f} | "
            f"Train: {train_time:.2f}s | Infer/sample: {result['inference_time_per_sample_ms']:.3f}ms | "
            f"Params: {result['n_parameters']} | Size~{result['estimated_size_mb']:.4f}MB"
        )
    else:
        print(
            f"Result {model_type} - Accuracy: {result['acc']:.4f} | Loss: {result['loss']:.4f} | "
            f"Train: {train_time:.2f}s | Infer/sample: {result['inference_time_per_sample_ms']:.3f}ms | "
            f"Params: {result['n_parameters']} | Size~{result['estimated_size_mb']:.4f}MB"
        )
    print(f"Graphs saved in {output_dir}")
    for graph_path in graph_paths:
        print(graph_path)
>>>>>>> origin/cnn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnn", "rnn"], default="cnn")
<<<<<<< HEAD
    parser.add_argument("--variant", choices=["naive", "regularized", "final"], default=None)
    parser.add_argument("--cnn-variant", choices=["naive", "regularized"], default=None)
=======
    parser.add_argument("--cnn-variant", choices=["naive", "regularized"], default="regularized")
>>>>>>> origin/cnn
    parser.add_argument("--compare-cnn", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-graphs", action="store_true")
    args = parser.parse_args()

<<<<<<< HEAD
    if args.cnn_variant and args.model != "cnn":
        raise ValueError("--cnn-variant can be used only with --model cnn")

    selected_variant = args.variant
    if selected_variant is None and args.model == "cnn" and args.cnn_variant is not None:
        selected_variant = args.cnn_variant
    if selected_variant is None:
        selected_variant = default_variant_for_model(args.model)

    if args.repeats >= 2:
        if args.compare_cnn and args.model == "cnn":
            variants = ["naive", "regularized"]
        else:
            variants = [selected_variant]
        run_repeats(
            model_type=args.model,
=======
    if args.model == "cnn" and args.repeats >= 2:
        if args.compare_cnn:
            variants = ["naive", "regularized"]
        else:
            variants = [args.cnn_variant]
        run_cnn_repeats(
>>>>>>> origin/cnn
            variants=variants,
            repeats=args.repeats,
            base_seed=args.base_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            show_graphs=args.show_graphs,
        )
    else:
        run_experiment(
<<<<<<< HEAD
            model_type=args.model,
            variant=selected_variant,
=======
            args.model,
            cnn_variant=args.cnn_variant,
>>>>>>> origin/cnn
            compare_cnn=args.compare_cnn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            show_graphs=args.show_graphs,
<<<<<<< HEAD
        )
=======
        )
>>>>>>> origin/cnn
