.DEFAULT_GOAL := help
.PHONY: help install up down logs seed test lint format

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
COMPOSE ?= docker compose

help: ## Lista os comandos disponíveis
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Cria o virtualenv e instala as dependências de desenvolvimento
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

up: ## Sobe API e banco com Docker (http://localhost:8000/docs)
	$(COMPOSE) up -d --build

down: ## Derruba os containers (mantém os volumes)
	$(COMPOSE) down

logs: ## Acompanha os logs da API
	$(COMPOSE) logs -f api

seed: ## Baixa o dataset, aplica as migrações e popula o banco
	$(COMPOSE) run --rm api sh -c "alembic upgrade head && sh scripts/download_data.sh && python scripts/seed_db.py"

test: ## Roda os testes com relatório de cobertura
	$(BIN)/python -m pytest --cov --cov-report=term

lint: ## Verifica estilo e formatação (ruff)
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Corrige estilo e formatação automaticamente
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
