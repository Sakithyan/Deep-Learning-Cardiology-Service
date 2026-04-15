# Production

Base production docker pour le service IA ECG.

## contenu
- docker-compose.yml
- ia_service/
- dockerisation/
- artifacts/

## lancer le service
```bash
cd production
docker compose up --build
```

API par defaut: http://localhost:5000

Routes:
- GET /
- GET /health
- POST /predict
- POST /classify

## test rapide
```bash
curl -X POST -F "model=cnn" -F "signal=@sample_ecg.txt" http://localhost:5000/predict
```

Les modeles et metadata doivent etre montes dans production/artifacts:
- mlp_model.keras
- cnn_model.keras
- rnn_model.keras
- preprocess.json
