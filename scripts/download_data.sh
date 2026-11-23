#!/usr/bin/env sh
# Baixa o dataset Steam (caserec) e descompacta em data/raw/steam/.
# Uso: scripts/download_data.sh [--force]
# Variáveis opcionais: DATA_DIR (padrão: data/raw), DATASET_URL, PYTHON (python3 ou python).
set -eu

URL="${DATASET_URL:-https://github.com/caserec/Datasets-for-Recommender-Systems/raw/master/Processed%20Datasets/Steam/steam.zip}"
DATA_DIR="${DATA_DIR:-data/raw}"
ZIP_PATH="$DATA_DIR/steam.zip"
FILES="game_purchase.dat game_play.dat item_info.dat user_info.dat"

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1 && python3 -c "" >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
fi

complete() {
    for f in $FILES; do
        [ -s "$DATA_DIR/steam/$f" ] || return 1
    done
}

if [ "${1:-}" != "--force" ] && complete; then
    echo "Dataset já presente em $DATA_DIR/steam (use --force para baixar de novo)."
    exit 0
fi

mkdir -p "$DATA_DIR"

echo "Baixando $URL"
if ! { command -v curl >/dev/null 2>&1 && curl -fsSL --retry 3 -o "$ZIP_PATH" "$URL"; }; then
    echo "curl indisponível ou falhou; usando Python para o download."
    "$PYTHON" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
        "$URL" "$ZIP_PATH"
fi

echo "Descompactando em $DATA_DIR/steam"
"$PYTHON" -m zipfile -e "$ZIP_PATH" "$DATA_DIR"

if ! complete; then
    echo "Erro: arquivos esperados não encontrados em $DATA_DIR/steam" >&2
    exit 1
fi
echo "Dataset pronto: $(ls "$DATA_DIR/steam" | tr '\n' ' ')"
