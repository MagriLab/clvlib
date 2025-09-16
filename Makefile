PY ?= python3
PIP ?= $(PY) -m pip
VENV ?= .venv

.PHONY: venv activate install dev build clean test lint format

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -U pip setuptools wheel

install: venv
	$(VENV)/bin/pip install -r requirements.txt

dev: venv
	$(VENV)/bin/pip install -r requirements.txt -r requirements-dev.txt

build:
	$(VENV)/bin/python -m build

test:
	$(VENV)/bin/python -m pytest -q

lint:
	$(VENV)/bin/ruff check clvlib

format:
	$(VENV)/bin/black clvlib

clean:
	rm -rf dist build *.egg-info
	rm -rf $(VENV)
