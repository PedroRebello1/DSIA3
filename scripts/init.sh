#!/usr/bin/env sh
# Prepara o ambiente do zero: migrações, dataset, seed e treino inicial do modelo.
# Todas as etapas são idempotentes; pode rodar quantas vezes quiser.
#
# Variáveis opcionais:
#   FORCE_TRAIN=1   retreina mesmo se o artefato do modelo já existir
#   TRAIN_ARGS      argumentos extras para "python -m recsys.train" (ex.: --no-search)
set -eu

MODEL="${ARTIFACTS_DIR:-artifacts}/${MODEL_FILENAME:-model.joblib}"

echo "[init] 1/4 Aplicando migrações do banco"
python -m alembic upgrade head

echo "[init] 2/4 Conferindo o dataset"
sh scripts/download_data.sh

echo "[init] 3/4 Populando o banco"
python scripts/seed_db.py

if [ -f "$MODEL" ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
    echo "[init] 4/4 Modelo já existe em $MODEL; treino pulado (FORCE_TRAIN=1 para retreinar)"
else
    echo "[init] 4/4 Treinando o modelo"
    # shellcheck disable=SC2086
    python -m recsys.train ${TRAIN_ARGS:-}
fi

echo "[init] Pronto"
