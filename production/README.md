## prerequis (linux/wsl2)
```bash
cd production/dockerisation
chmod +x preload session-base.sh scripts/check-env.sh
./preload -d
./scripts/check-env.sh
```

## lancer le projet
```bash
cd production
docker compose up --build -d
docker compose ps
```


## site web
- URL du site: http://localhost:8081/


## API
- Front API: http://localhost:8081/api
- IA API: http://localhost:5000

Routes:
- POST /api/classify (1 modele)
- POST /api/classify-all

## test rapide
```bash
curl -X POST -F "signal=@votre_signal.txt" http://localhost:8081/api/classify-all
```

Les modeles et metadata doivent etre montes dans production/artifacts:
- mlp_model.keras
- cnn_model.keras
- rnn_model.keras
- preprocess.json

## arreter le projet
```bash
cd production
docker compose down
```
## contenu
- docker-compose.yml
- api/
- ia_service/
- dockerisation/
- artifacts/

## architecture
- conteneur 1: spring-front (port 8081)
- conteneur 2: ia-service (port 5000)
- reseau prive docker: ecg-private-net

## screenshot
![Capture du site](Capture-Site.png)
