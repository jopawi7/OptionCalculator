# Identify Windows vs. Unix
ifeq ($(OS),Windows_NT)
    SHELL := cmd
    .SHELLFLAGS := /Q /C
    PY := python
    PIP := pip
    SEP := &
    CD_BACK := cd ..
    # PowerShell for more robust tasks if available
    POWERSHELL := powershell -NoProfile -ExecutionPolicy Bypass
    RM_DIST := $(POWERSHELL) Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist
else
    SHELL := /bin/sh
    PY := python3
    PIP := pip3
    SEP := &&
    CD_BACK := cd ..
    RM_DIST := rm -rf dist
endif

# Packages
BACKEND_DIR := Backend
FRONTEND_DIR := Frontend

.PHONY: backend-install frontend-install backend-start frontend-start frontend-clean all

# Install python dependencies in the backend
backend-install:
	@cd $(BACKEND_DIR) $(SEP) $(PIP) install -r requirements.txt

# install node dependencies in the frontend
frontend-install:
	@cd $(FRONTEND_DIR) $(SEP) npm install

# Start the backend (FastAPI with uvicorn)
# Note: Background processes are different under Windows/cmd; start separately for development
backend-start:
ifeq ($(OS),Windows_NT)
	@cd $(BACKEND_DIR) $(SEP) $(PY) -m uvicorn server:app --reload
else
	@cd $(BACKEND_DIR) $(SEP) uvicorn server:app --reload &
endif

# Starts the Angular Frontend
frontend-start:
	@cd $(FRONTEND_DIR) $(SEP) npx ng serve

# Cleans Angular build files
frontend-clean:
	@cd $(FRONTEND_DIR) $(SEP) $(RM_DIST)

# Build everything with make all
all: backend-install frontend-install backend-start frontend-start
