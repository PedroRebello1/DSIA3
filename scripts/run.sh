#!/usr/bin/env sh
# Sobe a aplicação, com ou sem Docker (equivalente a "make up" e "make run").
# Uso: sh scripts/run.sh [docker|local]
#   docker (padrão)  sobe db, init (migrações, dataset, seed e treino) e api com o
#                    Docker Compose e espera a API responder
#   local            sem Docker: prepara o banco com scripts/init.sh e sobe o Uvicorn
#                    (rode antes: sh scripts/install.sh)
# Variáveis opcionais:
#   API_PORT       porta da API (padrão: 8000)
#   TIMEOUT        segundos de espera pela API no modo docker (padrão: 900)
#   VENV           virtualenv do modo local (padrão: .venv)
#   DATABASE_URL   banco do modo local (padrão: SQLite em steam.db, se não houver .env)
set -eu

MODE="${1:-docker}"
PORT="${API_PORT:-8000}"
VENV="${VENV:-.venv}"

wait_api() {
    url="http://localhost:$PORT/health"
    limit="${TIMEOUT:-900}"
    echo "Esperando a API em $url (até ${limit}s)"
    elapsed=0
    until curl -fsS "$url" >/dev/null 2>&1; do
        if [ "$elapsed" -ge "$limit" ]; then
            echo "Erro: a API não respondeu em ${limit}s. Veja: docker compose logs init api" >&2
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
}

case "$MODE" in
    docker)
        echo "Subindo db, init e api (a primeira vez leva alguns minutos: seed e treino)"
        docker compose up -d --build
        if command -v curl >/dev/null 2>&1; then
            wait_api
        fi
        echo "API no ar: http://localhost:$PORT/docs"
        echo "Logs: docker compose logs -f api | Parar: docker compose down"
        ;;
    local)
        # Linux/macOS usam bin/; o Windows (Git Bash) usa Scripts/.
        if [ -d "$VENV/bin" ]; then
            BIN="$(pwd)/$VENV/bin"
        elif [ -d "$VENV/Scripts" ]; then
            BIN="$(pwd)/$VENV/Scripts"
        else
            echo "Erro: virtualenv não encontrado em $VENV; rode antes: sh scripts/install.sh" >&2
            exit 1
        fi
        PATH="$BIN:$PATH"
        export PATH
        if [ -z "${DATABASE_URL:-}" ] && [ ! -f .env ]; then
            DATABASE_URL="sqlite:///steam.db"
            export DATABASE_URL
            echo "DATABASE_URL não definida; usando SQLite em steam.db"
        fi
        sh scripts/init.sh
        echo "Subindo a API em http://localhost:$PORT/docs (Ctrl+C para parar)"
        exec python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
        ;;
    *)
        echo "Uso: sh scripts/run.sh [docker|local]" >&2
        exit 2
        ;;
esac
