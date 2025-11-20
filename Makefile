# Installiert Python-Abhängigkeiten im Backend
backend-install:
	cd Backend && pip install -r requirements.txt

# Installiert Node-Abhängigkeiten für das Frontend
frontend-install:
	cd Frontend && npm install

# Startet das Backend (FastAPI mit uvicorn)
backend-start:
	cd Backend && uvicorn server:app --reload &

# Startet das Frontend (Angular)
frontend-start:
	cd Frontend && ng serve

# Bereinigt Angular Build-Files ggf.
frontend-clean:
	cd Frontend && rm -rf dist

# Alles mit einem Befehl: Erst installieren, dann beide Server starten
all: backend-install frontend-install backend-start frontend-start
