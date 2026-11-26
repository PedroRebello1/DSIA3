#!/usr/bin/env sh
# Smoke test da stack em execução (rode depois de "docker compose up -d --build").
# Espera a API responder, confere se o modelo híbrido foi carregado e pede uma recomendação.
#
# Variáveis opcionais: BASE_URL (padrão: http://localhost:$API_PORT), TIMEOUT em segundos.
set -eu

BASE_URL="${BASE_URL:-http://localhost:${API_PORT:-8000}}"
TIMEOUT="${TIMEOUT:-300}"

echo "Esperando $BASE_URL/health (até ${TIMEOUT}s)"
elapsed=0
until curl -fsS "$BASE_URL/health" >/dev/null 2>&1; do
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "Erro: a API não respondeu em ${TIMEOUT}s" >&2
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

check() {
    # check <descrição> <caminho> <trecho esperado na resposta>
    body=$(curl -fsS "$BASE_URL$2")
    echo "$1: $body"
    if ! printf '%s' "$body" | grep -q "$3"; then
        echo "Erro: esperado '$3' em $2" >&2
        exit 1
    fi
}

check "health" "/health" '"model_loaded":true'
check "modelo" "/model/info" '"strategy":"hybrid"'
check "recomendações" "/users/1/recommendations?k=3" '"reason":"'
check "similares" "/games/2/similar?k=3" '"similar":\[{'

echo "Smoke test OK"
