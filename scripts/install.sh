#!/usr/bin/env sh
# Cria o virtualenv e instala as dependências (equivalente a "make install").
# Uso: sh scripts/install.sh [--notebooks]
#   --notebooks   instala também as dependências dos notebooks (matplotlib, Jupyter)
# Variáveis opcionais: PYTHON (padrão: python3 ou python), VENV (padrão: .venv).
set -eu

VENV="${VENV:-.venv}"

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "" >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

if ! "$PYTHON" -c "import sys; sys.exit(sys.version_info < (3, 12))" >/dev/null 2>&1; then
    echo "Erro: é preciso Python 3.12 ou mais novo ($PYTHON)." >&2
    exit 1
fi

echo "Criando o virtualenv em $VENV"
"$PYTHON" -m venv "$VENV"

# Linux/macOS usam bin/; o Windows (Git Bash) usa Scripts/.
if [ -d "$VENV/bin" ]; then
    VENV_PYTHON="$VENV/bin/python"
    ACTIVATE=". $VENV/bin/activate"
else
    VENV_PYTHON="$VENV/Scripts/python"
    ACTIVATE="$VENV\\Scripts\\activate"
fi

echo "Instalando as dependências"
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r requirements-dev.txt
if [ "${1:-}" = "--notebooks" ]; then
    "$VENV_PYTHON" -m pip install -r notebooks/requirements.txt
fi

echo "Pronto. Ative o ambiente com: $ACTIVATE"
echo "Depois rode a aplicação com: sh scripts/run.sh local"
