import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List

import numpy as np
import tensorflow as tf


DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "ecg-private-net")
DEFAULT_THRESHOLD = float(os.getenv("ECG_CLASSIFICATION_THRESHOLD", "0.5"))
DEFAULT_METADATA_PATH = os.getenv("ECG_METADATA_PATH", "/models/preprocess.json")
DEFAULT_MODEL_PATHS = {
    "mlp": os.getenv("ECG_MODEL_MLP_PATH", "/models/mlp_model.keras"),
    "cnn": os.getenv("ECG_MODEL_CNN_PATH", "/models/cnn_model.keras"),
    "rnn": os.getenv("ECG_MODEL_RNN_PATH", "/models/rnn_model.keras"),
}


@dataclass
class RuntimeMetadata:
    mean: float
    std: float
    input_length: int
    threshold: float


class InferenceRuntime:
    def __init__(self) -> None:
        self._lock = Lock()
        self._initialized = False
        self._metadata: RuntimeMetadata | None = None
        self._models: Dict[str, tf.keras.Model] = {}

    @property
    def metadata(self) -> RuntimeMetadata | None:
        return self._metadata

    def available_models(self) -> List[str]:
        return sorted(self._models.keys())

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._metadata = self._load_metadata(DEFAULT_METADATA_PATH)
            for model_name, model_path in DEFAULT_MODEL_PATHS.items():
                p = Path(model_path)
                if p.exists():
                    self._models[model_name] = tf.keras.models.load_model(p)
            self._initialized = True

    def _load_model_if_available(self, model_name: str) -> bool:
        model_path = Path(DEFAULT_MODEL_PATHS[model_name])
        if not model_path.exists():
            return False

        with self._lock:
            if model_name in self._models:
                return True
            self._models[model_name] = tf.keras.models.load_model(model_path)
        return True

    def _load_metadata(self, metadata_path: str) -> RuntimeMetadata:
        p = Path(metadata_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Fichier metadata introuvable: {metadata_path}. Montez un fichier preprocess.json."
            )

        with p.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        mean = float(data.get("mean", 0.0))
        std = float(data.get("std", 1.0))
        if std == 0.0:
            std = 1.0

        input_length = int(data.get("input_length", 96))
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        return RuntimeMetadata(mean=mean, std=std, input_length=input_length, threshold=threshold)

    def _read_signal(self, file_bytes: bytes) -> np.ndarray:
        if self._metadata is None:
            raise RuntimeError("Runtime non initialise")

        text = file_bytes.decode("utf-8-sig", errors="ignore")
        non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not non_empty_lines:
            raise ValueError("Le fichier est vide")

        first_line = non_empty_lines[0]
        tokens = [token for token in re.split(r"[,;\s]+", first_line) if token]
        values = [float(token) for token in tokens]

        if len(values) == self._metadata.input_length + 1:
            values = values[1:]

        if len(values) != self._metadata.input_length:
            raise ValueError(
                f"Signal invalide: {len(values)} points recus, {self._metadata.input_length} attendus"
            )

        signal = np.asarray(values, dtype=np.float32)
        signal = (signal - self._metadata.mean) / self._metadata.std
        return signal

    def _format_for_model(self, signal: np.ndarray, model_name: str) -> np.ndarray:
        if model_name == "mlp":
            return signal.reshape(1, -1)
        return signal.reshape(1, -1, 1)

    def predict(self, model_name: str, file_bytes: bytes) -> Dict[str, object]:
        if not self._initialized:
            self.initialize()

        if model_name not in DEFAULT_MODEL_PATHS:
            raise ValueError("Modele inconnu. Choisissez: mlp, cnn, rnn")

        if model_name not in self._models and not self._load_model_if_available(model_name):
            raise FileNotFoundError(
                f"Modele '{model_name}' indisponible. Attendu: {DEFAULT_MODEL_PATHS[model_name]}"
            )

        signal = self._read_signal(file_bytes)
        x = self._format_for_model(signal, model_name)
        prob = float(self._models[model_name](x, training=False).numpy().reshape(-1)[0])

        threshold = self._metadata.threshold if self._metadata is not None else DEFAULT_THRESHOLD
        label = "malade" if prob >= threshold else "pas malade"

        return {
            "model": model_name,
            "probability_malade": prob,
            "threshold": threshold,
            "prediction": label,
            "docker_network": DOCKER_NETWORK_NAME,
        }

    def predict_all(self, file_bytes: bytes) -> Dict[str, object]:
        if not self._initialized:
            self.initialize()

        signal = self._read_signal(file_bytes)
        threshold = self._metadata.threshold if self._metadata is not None else DEFAULT_THRESHOLD

        results: Dict[str, object] = {}
        errors: Dict[str, str] = {}

        for model_name in ["mlp", "cnn", "rnn"]:
            if model_name not in self._models and not self._load_model_if_available(model_name):
                errors[model_name] = f"Modele indisponible: {DEFAULT_MODEL_PATHS[model_name]}"
                continue

            x = self._format_for_model(signal, model_name)
            prob = float(self._models[model_name](x, training=False).numpy().reshape(-1)[0])
            label = "malade" if prob >= threshold else "pas malade"
            results[model_name] = {
                "probability_malade": prob,
                "prediction": label,
            }

        return {
            "models": results,
            "errors": errors,
            "threshold": threshold,
            "docker_network": DOCKER_NETWORK_NAME,
        }


runtime = InferenceRuntime()
