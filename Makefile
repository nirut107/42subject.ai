# =========================
# CONFIG
# =========================
PYTHON=python3
AI_DIR=ai
VENV=myvenv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python
UVICORN=$(VENV)/bin/uvicorn

PY_APP=api
PY_PORT=8000

WEB_DIR=web
NEXT_PORT=3000

# =========================
# PYTHON (AI)
# =========================

venv:
	cd $(AI_DIR) && $(PYTHON) -m venv myvenv

pip-install:
	$(PIP) install --upgrade pip
	$(PIP) install fastapi uvicorn google-api-python-client google-auth pdfplumber python-dotenv neo4j google-genai 
run-python:
	cd $(AI_DIR) && $(UVICORN) $(PY_APP):app --reload --host 0.0.0.0 --port $(PY_PORT)

# =========================
# NEXT.JS (WEB)
# =========================

run-next:
	cd $(WEB_DIR) && npm run dev -- -p $(NEXT_PORT)

# =========================
# ALL
# =========================


run-all:
	DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=180 AICHECK=0 make run-python & make run-next 

run-check:
	AICHECK=1 make run-python & make run-next 
# =========================
# CLEAN
# =========================

clean:
	rm -rf $(VENV)
