import os

from flask import Flask, jsonify, request

from inference import DOCKER_NETWORK_NAME, runtime


app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "ecg-ia",
            "status": "ready",
            "message": "POST /predict ou /classify (single model), POST /predict-all (mlp+cnn+rnn)",
        }
    )


@app.get("/health")
def health():
    init_error = None
    try:
        runtime.initialize()
    except Exception as exc:
        init_error = str(exc)

    status = "ok" if init_error is None else "degraded"
    return jsonify(
        {
            "status": status,
            "docker_network": DOCKER_NETWORK_NAME,
            "available_models": runtime.available_models(),
            "metadata_loaded": runtime.metadata is not None,
            "init_error": init_error,
        }
    )


def _predict_impl():
    model_name = request.form.get("model", "cnn").strip().lower()
    incoming_file = (
        request.files.get("signal")
        or request.files.get("file")
        or request.files.get("picture")
    )

    if incoming_file is None:
        return jsonify({"error": "Aucun fichier recu. Utilisez le champ 'signal'."}), 400

    file_bytes = incoming_file.read()
    if not file_bytes:
        return jsonify({"error": "Le fichier envoye est vide."}), 400

    try:
        runtime.initialize()
        result = runtime.predict(model_name=model_name, file_bytes=file_bytes)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur interne: {exc}"}), 500


@app.post("/predict")
def predict():
    return _predict_impl()


@app.post("/classify")
def classify():
    return _predict_impl()


@app.post("/predict-all")
def predict_all():
    incoming_file = (
        request.files.get("signal")
        or request.files.get("file")
        or request.files.get("picture")
    )

    if incoming_file is None:
        return jsonify({"error": "Aucun fichier recu. Utilisez le champ 'signal'."}), 400

    file_bytes = incoming_file.read()
    if not file_bytes:
        return jsonify({"error": "Le fichier envoye est vide."}), 400

    try:
        runtime.initialize()
        result = runtime.predict_all(file_bytes=file_bytes)
        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Erreur interne: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
