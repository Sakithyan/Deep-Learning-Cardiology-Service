# Ressources production

## prerequis linux
- ubuntu recommande
- vmware ok / dualboot ok
- wsl2 ok aussi

## prerequis application
- curl
- docker
- maven
- java
- git
- pas d'IDE obligatoire

## commandes
```bash
cd production/dockerisation
chmod +x preload session-base.sh scripts/check-env.sh
./preload -d
# ou
./preload -e
# ou
./preload -f

./scripts/check-env.sh
```

## modes
- `-d` : garde l'existant, installe le manque
- `-e` : force docker engine officiel
- `-f` : docker.io legacy

## note
- bonne connexion reseau
- prevoir plusieurs Go libres
- reboot parfois necessaire apres install docker
