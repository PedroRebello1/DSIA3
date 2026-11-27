# Steam RecSys

API que recomenda jogos da Steam com base no que cada usuário comprou e em quanto tempo
jogou. O modelo é **híbrido**: filtragem colaborativa item a item, similaridade de conteúdo
dos títulos e popularidade, com pesos que dependem do tamanho do histórico do usuário.

- Cadastro de usuários e jogos, atualização de preferências (compra, horas jogadas e
  "não gostei") e recomendações personalizadas.
- Cada recomendação vem com o motivo: *"Porque você jogou Fallout 4"*.
- Usuários novos recebem os jogos mais populares; jogos cadastrados pela API já podem ser
  recomendados, sem retreino.
- Estratégia selecionável (`hybrid`, `cf`, `content`, `popular`) e retreino em segundo plano.
- Documentação interativa (Swagger UI) em `/docs`, com exemplos em todos os endpoints.

Stack: Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, PostgreSQL 16, pandas, scipy,
scikit-learn, Docker Compose e pytest.

## Sumário

- [Como rodar com Docker](#como-rodar-com-docker)
- [Como rodar sem Docker](#como-rodar-sem-docker)
- [Usando a API](#usando-a-api)
- [Configuração](#configuração)
- [Testes e qualidade](#testes-e-qualidade)
- [Modelo e avaliação](#modelo-e-avaliação)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Documentação](#documentação)
- [Equipe](#equipe)
- [Dataset e créditos](#dataset-e-créditos)

## Como rodar com Docker

**Pré-requisitos:** [Docker](https://docs.docker.com/get-docker/) com o plugin
Docker Compose v2 e acesso à internet (para baixar o dataset na primeira subida).

```bash
git clone https://github.com/PedroRebello1/DSIA3.git
cd DSIA3
docker compose up --build
```

Na primeira subida, o serviço `init` aplica as migrações, baixa o dataset, popula o banco e
treina o modelo; depois a API sobe em <http://localhost:8000>. Isso leva alguns minutos.
Nas subidas seguintes, o dataset, o banco e o modelo já estão nos volumes, e a API sobe em
segundos.

Abra a documentação em <http://localhost:8000/docs>.

Alternativa: `sh scripts/run.sh` faz o mesmo em segundo plano e só termina quando a API
estiver respondendo.

| Serviço | O que faz |
|---|---|
| `db` | PostgreSQL 16, com volume persistente e *healthcheck* |
| `init` | roda uma vez a cada subida: migrações → dataset → seed → treino (só se ainda não houver modelo) |
| `api` | Uvicorn na porta 8000; só sobe depois que o `init` termina com sucesso |

Com `make` instalado, os atalhos são:

| Comando | O que faz |
|---|---|
| `make up` | sobe tudo em segundo plano e espera a API (`sh scripts/run.sh docker`) |
| `make logs` | acompanha os logs da API |
| `make smoke` | confere se a stack está respondendo (`/health`, uma recomendação e `/similar`) |
| `make train` | retreina o modelo em um container (depois reinicie a API ou use `POST /model/retrain`) |
| `make down` | derruba os containers, mantendo os volumes |

Para recomeçar do zero (banco, dataset e modelo), use `docker compose down -v`.

## Como rodar sem Docker

**Pré-requisitos:** Python 3.12 ou mais novo e um shell POSIX (`sh`; no Windows, o Git Bash).
O banco pode ser um PostgreSQL local ou um arquivo SQLite.

```bash
sh scripts/install.sh       # cria o .venv e instala as dependências (make install)
sh scripts/run.sh local     # migrações, dataset, seed, treino e API (make run)
```

A API fica em <http://localhost:8000/docs>. Sem `DATABASE_URL` definida e sem `.env`, o
banco é um arquivo SQLite (`steam.db`). Para usar o PostgreSQL do compose, copie o
`.env.example` para `.env` (ele já aponta para `localhost:5432`) e suba só o banco com
`docker compose up -d db` antes do `run.sh`.

Os mesmos passos, sem os scripts:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

export DATABASE_URL=sqlite:///steam.db  # ou postgresql+psycopg://usuario:senha@localhost:5432/steam
sh scripts/init.sh                      # migrações, dataset, seed e treino
uvicorn app.main:app --reload
```

Se o download do dataset falhar (sem internet ou atrás de um proxy que intercepta HTTPS),
baixe o `steam.zip` do repositório caserec pelo navegador e aponte o script para ele:
`export DATASET_URL=file:///caminho/para/steam.zip` antes do `run.sh` ou do `init.sh`.

As etapas do `init.sh` também podem rodar separadas:

```bash
python -m alembic upgrade head          # cria as tabelas
sh scripts/download_data.sh             # baixa e descompacta o dataset em data/raw/steam/
python scripts/seed_db.py               # popula o banco (idempotente)
python -m recsys.train                  # treina e grava artifacts/model.joblib
```

## Usando a API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | status da API e do modelo carregado |
| POST | `/users` | cria um usuário, opcionalmente já com jogos |
| GET | `/users` · `/users/{id}` | lista usuários · dados e histórico de um usuário |
| PUT | `/users/{id}/preferences` | substitui todo o histórico |
| PATCH | `/users/{id}/preferences` | adiciona, atualiza ou remove jogos; marca "não gostei" |
| GET | `/users/{id}/recommendations` | recomendações (`k`, `strategy`, `exclude_owned`) |
| POST | `/games` | cadastra um jogo (`name` e `tags` opcionais) |
| GET | `/games` · `/games/{id}` | lista e busca (`search`) · detalhe de um jogo |
| GET | `/games/{id}/similar` | jogos parecidos |
| GET | `/model/info` | versão do modelo e métricas do último treino |
| POST | `/model/retrain` | retreina o modelo em segundo plano |

Exemplos com `curl` (os IDs de jogos são os do dataset: 2 = *Fallout 4*,
512 = *The Witcher 3 Wild Hunt*, 1 = *The Elder Scrolls V Skyrim*):

```bash
# Busca jogos pelo nome
curl "http://localhost:8000/games?search=fallout&limit=5"

# Cria um usuário com dois jogos no histórico (a resposta traz o "id" do usuário)
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"games": [{"game_id": 2, "hours": 80.5}, {"game_id": 512, "hours": 120}]}'

# Recomendações para o usuário criado (troque 12394 pelo id devolvido)
curl "http://localhost:8000/users/12394/recommendations?k=5"

# Marca um jogo como "não gostei": ele e jogos parecidos perdem espaço
curl -X PATCH http://localhost:8000/users/12394/preferences \
  -H "Content-Type: application/json" \
  -d '{"upsert": [{"game_id": 1, "disliked": true}]}'

# Usa só o modelo de conteúdo, incluindo jogos que o usuário já tem
curl "http://localhost:8000/users/12394/recommendations?k=5&strategy=content&exclude_owned=false"

# Cadastra um jogo novo e vê os jogos parecidos com Fallout 4
curl -X POST http://localhost:8000/games \
  -H "Content-Type: application/json" \
  -d '{"name": "Hades", "tags": ["roguelike", "indie"]}'
curl "http://localhost:8000/games/2/similar?k=5"

# Informações do modelo e retreino com os dados atuais do banco
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/model/retrain
```

No Windows, alguns terminais não enviam acentos em UTF-8 pelo `curl -d`; para textos com
acento, use o Swagger UI ou um arquivo (`curl --data-binary @corpo.json`).

Exemplo de resposta de `GET /users/{id}/recommendations?k=3` para o usuário criado acima:

```json
{
  "user_id": 12394,
  "strategy": "hybrid",
  "model_version": "2026-11-27T14:00:00+00:00",
  "recommendations": [
    {"game_id": 1506, "name": "The Witcher 3 Wild Hunt - New Finisher Animations", "score": 0.501, "reason": "Porque você jogou The Witcher 3 Wild Hunt"},
    {"game_id": 4, "name": "Fallout New Vegas", "score": 0.427, "reason": "Porque você jogou Fallout 4"},
    {"game_id": 1968, "name": "The Witcher 3 Wild Hunt - Expansion Pass", "score": 0.425, "reason": "Parecido com The Witcher 3 Wild Hunt, que você jogou"}
  ]
}
```

Erros seguem sempre o mesmo formato, por exemplo
`{"error": {"code": "not_found", "message": "Usuário 99999 não encontrado", "details": null}}`.

## Configuração

Variáveis de ambiente lidas pelo compose e pela API (veja o `.env.example`):

| Variável | Padrão | Uso |
|---|---|---|
| `POSTGRES_USER` · `POSTGRES_PASSWORD` · `POSTGRES_DB` | `steam` | credenciais do banco do compose |
| `POSTGRES_PORT` | `5432` | porta do banco na máquina |
| `API_PORT` | `8000` | porta da API na máquina |
| `DATABASE_URL` | PostgreSQL em `localhost:5432` | conexão da API e dos scripts fora do Docker |
| `ARTIFACTS_DIR` | `artifacts` | pasta do modelo treinado |
| `DEFAULT_K` | `10` | quantidade de recomendações quando `k` não é informado |
| `FORCE_TRAIN` | `0` | `1` faz o `init` retreinar mesmo se o modelo já existir |
| `TRAIN_ARGS` | vazio | argumentos extras do treino no `init` (ex.: `--no-search`) |
| `DATASET_URL` | repositório caserec | origem alternativa do `steam.zip` |

Exemplo: `FORCE_TRAIN=1 TRAIN_ARGS=--no-search docker compose up --build` retreina o modelo
sem a busca de hiperparâmetros.

## Testes e qualidade

```bash
make install    # cria o .venv e instala as dependências (sh scripts/install.sh)
make test       # pytest com relatório de cobertura (mínimo de 80%)
make lint       # ruff check + ruff format --check
```

Sem `make`: `pytest --cov` e `ruff check . && ruff format --check .`.

Os testes usam SQLite em memória e um dataset sintético, então não precisam do Docker
nem do dataset real. O CI (GitHub Actions) roda o lint, os testes com cobertura e um
*smoke test* que sobe o `docker compose` e chama a API.

## Modelo e avaliação

Avaliação offline com os 3.757 usuários que têm pelo menos 5 jogos (20% dos jogos de cada
um separados para teste):

| Modelo | NDCG@10 | Recall@10 | Hit rate@10 | Cobertura |
|---|---:|---:|---:|---:|
| M0 popularidade | 0,140 | 0,178 | 0,425 | 0,7% |
| M1 item-kNN | 0,385 | 0,392 | 0,680 | 37,9% |
| M2 SVD | 0,337 | 0,334 | 0,615 | 10,7% |
| M3 conteúdo | 0,181 | 0,214 | 0,448 | 45,1% |
| **M4 híbrido (servido)** | **0,398** | **0,403** | **0,694** | 38,5% |

Detalhes do modelo, do protocolo e do ajuste de hiperparâmetros em
[`docs/modelo.md`](docs/modelo.md). Para reproduzir:

```bash
pip install -r notebooks/requirements.txt   # para os notebooks
python -m recsys.tuning                     # ajuste de α e dos pesos do híbrido
python -m recsys.train --source files       # treino a partir dos arquivos do dataset
```

## Estrutura do repositório

```
app/                    API FastAPI
  api/routes/           rotas: users, preferences, recommendations, games, model, health
  schemas/              validação e exemplos (Pydantic)
  services/             regras de negócio e ponte com o modelo
  db/                   modelos SQLAlchemy, repositórios e migrações (Alembic)
recsys/                 modelos de recomendação
  data.py               leitura, limpeza e score de confiança
  models/               popularidade, item-kNN, SVD, conteúdo e híbrido
  evaluation.py         métricas e protocolo de avaliação offline
  train.py · tuning.py  treino do modelo servido e ajuste de hiperparâmetros
scripts/                instalação, execução, download do dataset, seed, init e smoke test
notebooks/              01_eda.ipynb (análise exploratória), 02_comparacao_modelos.ipynb
tests/                  unit/, integration/ e conftest.py (fixtures)
docs/                   modelo.md e decisoes.md
Dockerfile              imagem multi-stage, usuário não-root e healthcheck
docker-compose.yml      db, init e api
Makefile                atalhos de desenvolvimento
```

`data/` (dataset) e `artifacts/` (modelo treinado) são gerados localmente e não vão para o
Git.

## Documentação

- [`docs/modelo.md`](docs/modelo.md): dados, modelos, protocolo de avaliação, resultados e
  escolha do modelo servido.
- [`docs/decisoes.md`](docs/decisoes.md): camadas da API, banco, atualização online do
  modelo, retreino, erros e validações.
- Swagger UI em `/docs` e ReDoc em `/redoc`, com a API rodando.

## Equipe

- Pedro Rebello: dados e modelo
- Guilherme Mendes: API e banco de dados
- Gabriel Francisco: infraestrutura, qualidade e documentação

## Dataset e créditos

[Steam Video Games](https://www.kaggle.com/tamber/steam-video-games), publicado no Kaggle por
Tamber, na versão processada do repositório
[caserec/Datasets-for-Recommender-Systems](https://github.com/caserec/Datasets-for-Recommender-Systems/tree/master/Processed%20Datasets/Steam).
São cerca de 200 mil registros de compras e horas jogadas de 12.393 usuários em 5.155 jogos.
