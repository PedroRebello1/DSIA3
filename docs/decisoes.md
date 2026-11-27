# Decisões de arquitetura e design

Este documento explica como a API está organizada e por que. O modelo de recomendação
em si (dados, algoritmos e avaliação) está em [`modelo.md`](modelo.md).

## 1. Visão geral

```
cliente ── HTTP ──> FastAPI (app/)
                      routes ──> services ──> crud ──> PostgreSQL
                                     │
                                     └──> recsys (modelo em memória) <── artifacts/model.joblib
```

No Docker Compose, três serviços: `db` (PostgreSQL 16), `init` (migrações, dataset, seed e
treino, uma vez) e `api` (Uvicorn). A API só sobe depois que o `init` termina com sucesso.

## 2. Camadas

| Camada | Pasta | Responsabilidade |
|---|---|---|
| Rotas | `app/api/routes/` | HTTP: parâmetros, códigos de status e documentação. Não tocam SQL nem o modelo |
| Schemas | `app/schemas/` | Validação de entrada e formato de saída (Pydantic v2), com exemplos para o Swagger |
| Serviços | `app/services/` | Regras de negócio (404, 409), transações (`commit`) e ponte com o `recsys` |
| Repositórios | `app/db/crud.py` | Consultas SQLAlchemy; dão `flush`, mas quem chama decide o `commit` |
| Modelo | `recsys/` | Recomendadores; não conhecem a API nem o banco |

Por quê:

- cada camada é testada isoladamente (schemas sem banco, serviços sem HTTP, modelos sem
  API);
- o `recsys` é usado igual pela API, pelos notebooks e pela linha de comando;
- trocar o modelo não mexe nas rotas: a API só usa o contrato `Recommender`
  (`recsys/base.py`), e foi assim que a popularidade do início foi substituída pelo híbrido.

## 3. Banco de dados

| Tabela | Campos principais |
|---|---|
| `users` | `id`, `steam_real_id` (único; nulo para quem foi criado pela API), `created_at` |
| `games` | `id`, `name` (único), `tags` (JSON), `created_at` |
| `interactions` | `user_id`, `game_id`, `purchased`, `hours`, `disliked`, `updated_at`; chave (`user_id`, `game_id`) |
| `model_runs` | `id`, `trained_at`, `params` (JSON), `metrics` (JSON), `artifact_path` |

- **Uma linha por par (usuário, jogo)**: a chave composta torna o *upsert* do `PATCH`
  natural, e o "não gostei" fica na mesma linha (um jogo pode ser comprado e não gostado).
- **`tags` em JSON**, e não `ARRAY`: funciona no PostgreSQL e no SQLite dos testes.
- **Nome do jogo único sem diferenciar maiúsculas**: o serviço confere antes de inserir
  (409 com o ID do jogo existente). Se duas requisições chegam juntas, a restrição `UNIQUE`
  do banco barra a segunda, e o `IntegrityError` também vira 409.
- **IDs do dataset preservados**: usuários e jogos do seed mantêm os IDs dos arquivos, então
  um modelo treinado pelos arquivos ou pelo banco fala dos mesmos jogos. Depois do seed, as
  sequências do PostgreSQL são ajustadas para os novos cadastros continuarem do maior ID.
- **Migrações com Alembic** (`app/db/migrations/`): a URL vem das configurações, e o
  `render_as_batch` permite rodar as migrações também no SQLite.
- **Seed idempotente** (`scripts/seed_db.py`): usa a mesma limpeza do modelo
  (`recsys/data.py`), insere em lotes com `INSERT ... ON CONFLICT DO NOTHING` e pode rodar a
  cada subida do compose sem duplicar nada.

## 4. O modelo dentro da API

- **Carregamento na subida**: o `lifespan` do FastAPI carrega `artifacts/model.joblib`
  (gerado por `python -m recsys.train`) e guarda o serviço em `app.state.recommender`.
- **Sem artefato, a API sobe mesmo assim**: no primeiro pedido de recomendação, o serviço
  treina a popularidade com os dados do banco (alguns segundos). Nesse modo, só
  `strategy=popular` funciona; as demais respondem 503 (`model_unavailable`) com a dica de
  rodar `POST /model/retrain`. Jogos parecidos usam o modelo de conteúdo.
- **O modelo recebe o histórico, não o usuário**: a cada pedido, o serviço lê as interações
  do usuário no banco e monta `{game_id: score}` com o mesmo α usado no treino (guardado no
  artefato). Jogos marcados como "não gostei" vão em `disliked`, e os jogos do histórico vão
  em `exclude` (a menos que `exclude_owned=false`).
- **Estratégia selecionável**: o híbrido expõe seus componentes (item-kNN, conteúdo e
  popularidade), então `?strategy=cf|content|popular` usa as mesmas instâncias, sem memória
  extra.
- **Explicações**: o `reason` de cada recomendação vem do modelo e passa direto para a
  resposta.

## 5. Atualização online

| Mudança | Quando vale | Como |
|---|---|---|
| Preferências do usuário (`PUT`/`PATCH`) | no próximo pedido | o histórico é lido do banco a cada recomendação |
| Jogo novo (`POST /games`) | na hora | a rota chama `add_game` no modelo carregado; o conteúdo vetoriza o jogo |
| Vizinhos do item-kNN e popularidade | no próximo retreino | dependem de todas as interações |

O `add_game` altera o modelo em memória, então o serviço o executa sob um *lock*: dois
cadastros simultâneos não intercalam as alterações.

## 6. Retreino

`POST /model/retrain` responde **202** na hora e treina em uma *background task* do FastAPI:

1. lê do banco as interações (sem os "não gostei") e o catálogo;
2. roda `recsys.train.train` (com `?search=true`, inclui a busca de hiperparâmetros);
3. grava o artefato de forma atômica: arquivo temporário e depois `os.replace`, para que um
   reinício nunca leia um arquivo pela metade;
4. registra o treino em `model_runs`;
5. monta o novo serviço completo e o publica com **uma única atribuição** em
   `app.state.recommender`.

Por quê:

- **troca atômica sem travar a leitura**: pedidos em andamento continuam com a referência
  ao modelo antigo, e os seguintes já pegam o novo. Não há *lock* no caminho das
  recomendações;
- **um retreino por vez**: um *lock* e a flag `app.state.retraining` recusam um segundo
  pedido com 409. O `GET /model/info` mostra se há retreino em andamento;
- **falha não derruba nada**: se o treino der erro, o modelo anterior continua em uso e o
  erro vai para o log.

Limitação conhecida: o treino roda no mesmo processo da API, que sobe com um único worker
do Uvicorn. Com vários workers, cada um teria o seu modelo em memória; aí o caminho seria
uma fila de tarefas (Celery, RQ) e um aviso para os workers recarregarem o artefato.

## 7. Erros e validação

Todas as respostas de erro, inclusive as de validação do FastAPI e erros inesperados, usam
o mesmo envelope:

```json
{"error": {"code": "not_found", "message": "Usuário 99999 não encontrado", "details": null}}
```

| Status | `code` | Quando |
|---|---|---|
| 404 | `not_found` | usuário ou jogo inexistente, inclusive jogos citados no corpo da requisição |
| 405 | `method_not_allowed` | método HTTP não suportado na rota |
| 409 | `conflict` | nome de jogo ou `steam_real_id` repetido; retreino já em andamento |
| 422 | `validation_error` | dados inválidos; `details` lista o campo (ex.: `body.games.0.hours`) e o problema |
| 503 | `model_unavailable` | estratégia que depende do modelo treinado, sem modelo carregado |
| 500 | `internal_error` | erro inesperado (registrado no log, sem expor detalhes) |

Os serviços lançam exceções de domínio (`NotFoundError`, `ConflictError`,
`ModelUnavailableError`); os *handlers* em `app/core/errors.py` as convertem para HTTP.

Principais validações:

- IDs no caminho e no corpo maiores que zero; `k` entre 1 e 100;
- horas entre 0 e 100.000;
- nome do jogo com 1 a 255 caracteres, sem espaços nas pontas; até 20 tags de até 50
  caracteres, em minúsculas e sem repetição;
- o mesmo jogo não pode aparecer duas vezes na mesma lista, nem em `upsert` e `remove` ao
  mesmo tempo; o `PATCH` precisa de pelo menos uma alteração.

Duas escolhas de comportamento:

- **jogo inexistente no corpo é 404**, não 422: o formato está certo, mas o recurso citado
  não existe;
- **remover um jogo que não está no histórico não é erro**: o `PATCH` é idempotente.

## 8. Configuração

Variáveis de ambiente (ou arquivo `.env`), lidas com `pydantic-settings` em
`app/core/config.py`:

| Variável | Padrão | Uso |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://steam:steam@localhost:5432/steam` | conexão com o banco |
| `ARTIFACTS_DIR` | `artifacts` | pasta do modelo treinado |
| `MODEL_FILENAME` | `model.joblib` | nome do artefato |
| `DEFAULT_K` | `10` | quantidade de recomendações quando `k` não é informado |

## 9. Testes

- **Banco de teste**: SQLite em memória (`StaticPool`) e a dependência `get_db`
  substituída no `TestClient`; nenhum teste precisa de PostgreSQL.
- **Dataset sintético** (`tests/conftest.py`) com resultado conhecido: dois grupos de jogos
  (RPG e tiro) e um jogo sem interações, para testar CF, conteúdo e cold start.
- **Artefatos isolados**: cada teste usa uma pasta temporária, para o modelo treinado na
  máquina não interferir.
- **CI** (GitHub Actions): lint (`ruff`), testes com cobertura mínima de 80% e um *smoke
  test* que sobe o `docker compose` e chama `/health`, uma recomendação e `/similar`.
