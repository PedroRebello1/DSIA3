# Modelo de recomendação

## Resumo

- O modelo servido é um **híbrido (M4)** de três componentes: filtragem colaborativa item a
  item (item-kNN), similaridade de conteúdo dos títulos e popularidade. O peso de cada um
  depende de quantos jogos o usuário tem.
- Na avaliação offline com 3.757 usuários, ele é o melhor dos cinco modelos comparados:
  **NDCG@10 de 0,398**, contra 0,385 do melhor modelo isolado (item-kNN) e 0,140 da
  popularidade. Também lidera com históricos de 1 ou 2 jogos e cobre 38,5% do catálogo.
- Cada recomendação vem com o motivo ("Porque você jogou Fallout 4"). Mudanças de
  preferência e jogos novos valem na hora, sem retreino.

Este documento descreve os dados, os cinco modelos comparados, o protocolo de avaliação,
os resultados e a escolha do modelo servido. Os números são reproduzidos pelos notebooks
[`01_eda.ipynb`](../notebooks/01_eda.ipynb) (análise exploratória) e
[`02_comparacao_modelos.ipynb`](../notebooks/02_comparacao_modelos.ipynb) (comparação).

## 1. Dados

Dataset *Steam Video Games* (Kaggle), na versão processada do repositório
[caserec](https://github.com/caserec/Datasets-for-Recommender-Systems/tree/master/Processed%20Datasets/Steam):
compras (`game_purchase.dat`), horas jogadas (`game_play.dat`), catálogo (`item_info.dat`)
e mapeamento de IDs da Steam (`user_info.dat`).

**Limpeza** (`recsys/data.py`):

- 707 linhas de compra duplicadas são removidas; os 12 pares (usuário, jogo) repetidos nas
  horas são sessões separadas, então as horas são somadas;
- dois jogos repetem o nome de outro com um espaço no fim ("Duke Nukem Forever" e
  "Universe Sandbox"); cada par é unificado no menor ID;
- compras sem registro de horas ficam com 0 hora.

| Depois da limpeza | Valor |
|---|---|
| Interações (pares usuário–jogo) | 128.801 |
| Usuários | 12.393 |
| Jogos | 5.153 |
| Esparsidade da matriz usuários × jogos | 99,80% |
| Usuários com 1 jogo · mediana de jogos por usuário | 46,0% · 2 |
| Usuários com ≥ 5 jogos | 30,3% (3.757) |
| Compras sem horas jogadas | 45,3% |
| Horas jogadas (mediana · p99 · máximo) | 4,5 h · 908 h · 11.754 h |

O que isso implica: não há notas (feedback implícito), a matriz é muito esparsa, quase
metade dos usuários tem um único jogo (cold start pesado) e o único conteúdo disponível é
o título do jogo.

### Score de confiança

Compra e horas viram um score por par (usuário, jogo):

```
score = 1 + α · log1p(horas)
```

Comprar sem jogar vale 1; jogar aumenta o score de forma logarítmica, para que um jogo de
10.000 horas não domine o histórico. O modelo servido usa **α = 0,25** (seção 6).

Jogos marcados como **"não gostei"** pela API não entram no histórico positivo: nunca são
recomendados e penalizam jogos parecidos (seção 3, M4).

## 2. Contrato dos modelos

Todos os modelos seguem o contrato `Recommender` de `recsys/base.py`:

```python
fit(interactions, games)  # treino
recommend(history, k=10, exclude=None, disliked=None)  # -> list[Recommendation]
similar_games(game_id, k=10)  # -> list[Recommendation]
add_game(game_id, name, tags=None)  # jogo novo, sem retreino
save(path) / load(path)  # artefato joblib
```

`Recommendation` tem `game_id`, `score` e `reason`. O modelo recebe o **histórico**
(`{game_id: score}`), não o ID do usuário: usuários criados ou alterados pela API recebem
recomendações na hora, sem retreino. Os modelos não excluem o histórico por conta própria;
quem chama passa em `exclude` o que não deve aparecer (a API passa os jogos que o usuário
já tem, a menos que `exclude_owned=false`).

## 3. Modelos

| # | Modelo | Arquivo | Papel |
|---|---|---|---|
| M0 | Popularidade | `recsys/models/popularity.py` | baseline e cold start |
| M1 | Item-kNN | `recsys/models/item_knn.py` | componente colaborativo do híbrido |
| M2 | SVD | `recsys/models/svd.py` | comparação com fatoração de matrizes |
| M3 | Conteúdo | `recsys/models/content.py` | jogos novos e franquias |
| M4 | **Híbrido** | `recsys/models/hybrid.py` | **modelo servido pela API** |

**M0, popularidade.** Ordena os jogos pela quantidade de usuários distintos. Ignora o
histórico (só exclui o que foi pedido). Motivo: "Popular entre jogadores".

**M1, item-kNN.** Cada jogo é a sua coluna na matriz de scores; a similaridade entre jogos
é o cosseno entre colunas. Só os **20 vizinhos** mais parecidos de cada jogo são guardados,
em matriz esparsa (5 mil jogos × 20 vizinhos). O score de um candidato `j` é
`Σ score(i) · sim(i, j)` sobre os jogos `i` do histórico. O motivo aponta o jogo do
histórico que mais contribuiu: "Porque você jogou Fallout 4". Em `similar_games`, o motivo
é "Quem jogou Fallout 4 também jogou".

**M2, SVD.** `TruncatedSVD` com 64 fatores sobre a matriz de scores. Cada jogo vira um
vetor latente; o histórico é projetado no mesmo espaço (*fold-in*), então também não precisa
de retreino para usuários novos.

**M3, conteúdo.** Cada jogo vira um vetor com três blocos: TF-IDF das palavras do título
(unigramas e bigramas, captura franquias como "Fallout" e "Total War"), TF-IDF de n-gramas
de caracteres de 3 a 5 (tolera variações como "Half-Life" e "Half Life 2") e as `tags`
opcionais dos jogos cadastrados pela API (com *hashing*, então uma tag nunca vista vale na
hora). Os vetores têm norma 1, e o score é a similaridade de cosseno com os jogos do
histórico, ponderada pelo score de cada um. Motivo: "Parecido com Fallout 4, que você
jogou". `add_game` vetoriza o jogo novo na hora.

**M4, híbrido.** Para cada pedido:

1. separa o histórico positivo (sem os jogos marcados como "não gostei");
2. escolhe os pesos (CF, conteúdo, popularidade) pela faixa de tamanho desse histórico:

   | Histórico | CF (M1) | Conteúdo (M3) | Popularidade (M0) |
   |---|---:|---:|---:|
   | 0 jogos | 0 | 0 | 1 |
   | 1–2 jogos | 0,3 | 0,3 | 0,4 |
   | 3 ou mais | 0,7 | 0,2 | 0,1 |

3. pede a cada componente com peso maior que zero um conjunto de candidatos
   (`max(100, 5·k)`), já sem os jogos bloqueados;
4. normaliza os scores de cada componente para [0, 1] (min-max) e soma com os pesos;
5. **penaliza "não gostei"**: para cada candidato, mede a similaridade (CF e conteúdo,
   normalizadas; vale a maior) com os jogos marcados e subtrai `0,5 × similaridade`;
6. descarta scores ≤ 0, ordena e devolve os `k` primeiros. O motivo de cada jogo vem do
   componente que mais contribuiu para o score dele.

Em `similar_games`, o híbrido faz a média (0,5 / 0,5) entre o item-kNN e o conteúdo.
`add_game` repassa o jogo a todos os componentes: o conteúdo já o torna recomendável e
o colaborativo passa a considerá-lo no próximo treino.

## 4. Protocolo de avaliação

Implementado em `recsys/evaluation.py`:

- avaliam-se os usuários com pelo menos 5 jogos (3.757). Para cada um, 20% dos jogos
  (no mínimo 1), sorteados com a semente 42, vão para o teste. Os usuários com menos jogos
  ficam inteiros no treino;
- cada modelo treina no conjunto de treino, recebe o histórico de treino do usuário e
  recomenda 10 jogos que ele ainda não tem;
- métricas médias @10: **Precision**, **Recall**, **NDCG**, **MAP** e **Hit rate**;
- **cobertura**: fração do catálogo (5.153 jogos) que aparece em pelo menos uma lista;
- **novidade**: autoinformação média `-log2(popularidade)` dos jogos recomendados, em que a
  popularidade é a fração de usuários de treino que têm o jogo (maior = menos óbvio).

As métricas têm testes com casos calculados à mão (`tests/unit/test_evaluation.py`).

## 5. Resultados da avaliação offline

### Comparação final (parâmetros ajustados, α = 0,25 para todos)

| Modelo | Precision | Recall | NDCG | MAP | Hit rate | Cobertura | Novidade | ms/usuário |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 popularidade | 0,0626 | 0,1779 | 0,1399 | 0,0849 | 0,4248 | 0,7% | 3,70 | 0,02 |
| M1 item-kNN (k = 20) | 0,1369 | 0,3922 | 0,3847 | 0,3088 | 0,6803 | 37,9% | 6,27 | 0,7 |
| M2 SVD (64 fatores) | 0,1168 | 0,3335 | 0,3374 | 0,2707 | 0,6149 | 10,7% | 6,01 | 1,4 |
| M3 conteúdo | 0,0724 | 0,2142 | 0,1808 | 0,1248 | 0,4480 | 45,1% | 8,89 | 4,8 |
| **M4 híbrido** | **0,1421** | **0,4027** | **0,3975** | **0,3186** | **0,6944** | 38,5% | 6,24 | 5,7 |

- O **item-kNN** é o melhor modelo isolado, com folga sobre o SVD: com 99,8% de zeros e
  23% dos jogos com um único usuário, vizinhanças esparsas funcionam melhor que 64 fatores
  latentes, e o SVD concentra as recomendações em poucos jogos (cobertura de 10,7%).
- O **conteúdo** sozinho acerta pouco (só há títulos), mas é o que mais cobre o catálogo e
  o que mais recomenda jogos pouco conhecidos (maior novidade).
- A **popularidade** cobre só 38 jogos (0,7% do catálogo): todos recebem quase a mesma lista.
- O **híbrido** é o melhor em todas as métricas de acurácia e mantém a cobertura do
  item-kNN. O custo é o tempo por pedido (cerca de 6 ms), que segue baixo para a API.

### Por tamanho do histórico

NDCG@10 por quantidade de jogos do usuário no treino (notebook 02, seção 3):

| Jogos no treino | Usuários | M4 híbrido | M1 item-kNN | M3 conteúdo | M0 popularidade |
|---|---:|---:|---:|---:|---:|
| 4–7 | 1.568 | **0,4278** | 0,4211 | 0,1940 | 0,1236 |
| 8–15 | 924 | **0,3848** | 0,3737 | 0,1636 | 0,1510 |
| 16–40 | 744 | **0,3330** | 0,3147 | 0,1551 | 0,1378 |
| 41+ | 521 | **0,4205** | 0,3944 | 0,2083 | 0,1723 |

O híbrido vence em todas as faixas, e a vantagem sobre o item-kNN cresce com o histórico.

### Cold start simulado

O protocolo não inclui usuários com 1 ou 2 jogos (não sobra o que testar). Para medir esse
caso, os modelos recebem só 1 ou 2 jogos sorteados do histórico de cada usuário avaliado
(amostra de 1.500 usuários; notebook 02, seção 4):

| NDCG@10 com | M4 híbrido | M1 item-kNN | M3 conteúdo | M0 popularidade |
|---|---:|---:|---:|---:|
| 1 jogo | **0,1508** | 0,1450 | 0,0683 | 0,1107 |
| 2 jogos | **0,1879** | 0,1745 | 0,0904 | 0,1121 |

Com histórico curto, o híbrido também fica à frente de todos os modelos isolados. Sem
nenhum jogo, ele recomenda só por popularidade.

## 6. Ajuste de hiperparâmetros (`python -m recsys.tuning`)

| Parâmetro | Valores testados | Escolhido | Por quê |
|---|---|---|---|
| Vizinhos do item-kNN | 20 / 50 / 100 | **20** | maior NDCG (0,357, contra 0,348 e 0,344) |
| α do score de confiança | 0 / 0,25 / 0,5 / 1 / 2 | **0,25** | maior NDCG (0,3825); as horas ajudam pouco (α = 0 dá 0,3810) e α alto piora |
| Pesos, histórico ≥ 3 jogos | 6 combinações | **0,7 / 0,2 / 0,1** | a proposta inicial já era a melhor (NDCG 0,3743) |
| Pesos, histórico 1–2 jogos | 6 combinações | **0,3 / 0,3 / 0,4** | NDCG 0,1840, contra 0,1633 da proposta inicial (0,4 / 0,4 / 0,2) |

Pesos na ordem CF / conteúdo / popularidade. Com pouco histórico, a popularidade ganha
importância. O SVD não entrou no ajuste: com 64 fatores, ele já ficou bem atrás do item-kNN,
que é o componente colaborativo do híbrido.

<details>
<summary>Tabelas completas do ajuste</summary>

Amostra fixa de 1.500 usuários avaliados. O script roda as etapas em sequência, cada uma
partindo do melhor valor da anterior: α na grade 0,5 / 1 / 2, depois os pesos de cada faixa
de histórico. Por isso as tabelas de pesos abaixo usam α = 0,5. Os valores α = 0 e 0,25
foram testados depois, com os pesos já escolhidos; essa rodada está no notebook 02
(seção 5) e levou ao α = 0,25 do modelo servido.

**k vizinhos do item-kNN** (busca de `recsys/train.py`, α = 1): 20 → NDCG 0,357;
50 → 0,348; 100 → 0,344. Escolhido: **20**.

**α do score de confiança** `1 + α·log1p(horas)`:

| α | Precision | Recall | NDCG | MAP | Hit rate | Cobertura |
|---|---:|---:|---:|---:|---:|---:|
| 0 (só compra) | 0,1341 | 0,3776 | 0,3810 | 0,3077 | 0,6553 | 26,3% |
| **0,25** | **0,1355** | **0,3875** | **0,3825** | 0,3061 | **0,6713** | 24,5% |
| 0,5 | 0,1336 | 0,3819 | 0,3743 | 0,2969 | 0,6707 | 26,3% |
| 1 | 0,1246 | 0,3627 | 0,3467 | 0,2709 | 0,6480 | 29,3% |
| 2 | 0,1109 | 0,3355 | 0,3094 | 0,2359 | 0,6160 | 33,0% |

As horas jogadas ajudam pouco: α = 0 quase empata com o melhor. Valores altos de α deixam
que poucos jogos muito jogados dominem o histórico e pioram a acurácia. Escolhido: **0,25**.

**Pesos CF / conteúdo / popularidade, histórico ≥ 3 jogos** (α = 0,5):

| Pesos | NDCG | MAP | Hit rate | Cobertura |
|---|---:|---:|---:|---:|
| **0,7 / 0,2 / 0,1** (proposta inicial) | **0,3743** | **0,2969** | **0,6707** | 26,3% |
| 0,8 / 0,2 / 0 | 0,3731 | 0,2967 | 0,6687 | 26,3% |
| 0,8 / 0,1 / 0,1 | 0,3711 | 0,2956 | 0,6687 | 25,9% |
| 0,9 / 0,1 / 0 | 0,3704 | 0,2950 | 0,6673 | 26,2% |
| 0,6 / 0,3 / 0,1 | 0,3700 | 0,2912 | 0,6647 | 27,5% |
| 1 / 0 / 0 (só CF) | 0,3646 | 0,2900 | 0,6607 | 26,0% |

**Pesos para histórico de 1–2 jogos.** O protocolo padrão só tem usuários com 5 ou mais
jogos, então aqui o modelo treina com todos os dados de treino, mas recebe apenas 2 jogos
do histórico de cada usuário avaliado (α = 0,5):

| Pesos | NDCG | MAP | Hit rate | Cobertura |
|---|---:|---:|---:|---:|
| 0,4 / 0,4 / 0,2 (proposta inicial) | 0,1633 | 0,1107 | 0,4273 | 50,1% |
| 0,6 / 0,2 / 0,2 | 0,1758 | 0,1204 | 0,4433 | 47,5% |
| **0,3 / 0,3 / 0,4** | **0,1840** | **0,1240** | 0,4813 | 45,9% |
| 0,3 / 0,2 / 0,5 | 0,1784 | 0,1129 | 0,5027 | 42,9% |
| 0,2 / 0,2 / 0,6 | 0,1722 | 0,1043 | 0,5120 | 38,7% |
| 1 / 0 / 0 (só CF) | 0,1679 | 0,1167 | 0,4173 | 47,6% |

Com pouco histórico, a popularidade ganha importância; acima de 0,4, o hit rate ainda sobe,
mas NDCG, MAP e cobertura caem. Escolhido: **0,3 / 0,3 / 0,4**.

</details>

## 7. Modelo servido

O **M4 híbrido** é o modelo da API porque:

1. é o melhor em todas as métricas de acurácia, tanto no protocolo padrão quanto com
   históricos curtos;
2. mantém a cobertura do item-kNN (38,5% do catálogo);
3. resolve o cold start (popularidade) e os jogos novos (conteúdo, via `add_game`);
4. explica cada recomendação, e todos os componentes pontuam direto do histórico, então
   mudanças de preferência valem no pedido seguinte, sem retreino.

| Parâmetro | Valor |
|---|---|
| α (score de confiança) | 0,25 |
| Vizinhos do item-kNN | 20 |
| Pesos, histórico 0 | 0 / 0 / 1 (só popularidade) |
| Pesos, histórico 1–2 | 0,3 / 0,3 / 0,4 |
| Pesos, histórico ≥ 3 | 0,7 / 0,2 / 0,1 |
| Penalidade "não gostei" | 0,5 (sem dados de feedback negativo para avaliar offline) |
| Candidatos por componente | `max(100, 5·k)` |

## 8. Treino e atualização

`python -m recsys.train` (no Docker, o serviço `init` roda na primeira subida):

1. lê as interações do banco (`--source db`, padrão) ou dos arquivos (`--source files`);
2. busca o número de vizinhos do item-kNN (20 / 50 / 100) em 1.000 usuários
   (`--no-search` pula a busca e usa os parâmetros da seção 7);
3. avalia os parâmetros escolhidos no protocolo da seção 4 (`--eval-users` limita a amostra);
4. retreina com **todos** os dados e grava `artifacts/model.joblib` de forma atômica
   (arquivo temporário + troca), com parâmetros, métricas e data do treino;
5. registra o treino na tabela `model_runs` (o `GET /model/info` mostra o último).

Na API, o artefato é carregado na subida. Depois disso:

- **preferências** alteradas valem no próximo pedido de recomendação (o modelo recebe o
  histórico atual do banco);
- **jogos novos** (`POST /games`) entram no modelo de conteúdo na hora;
- `POST /model/retrain` treina um modelo novo em segundo plano, com os dados atuais do
  banco, e o troca pelo antigo de uma vez quando termina.

## 9. Limitações

- O único conteúdo do dataset é o título. Jogos de franquias diferentes, mas do mesmo
  gênero, só ficam parecidos pelo CF ou pelas tags dos jogos cadastrados pela API.
- As horas jogadas ajudam pouco na acurácia (α = 0 quase empata com o melhor valor).
- A penalidade de "não gostei" não foi avaliada offline: o dataset não tem feedback
  negativo.
- A avaliação é offline e por amostragem aleatória do histórico, não temporal: o dataset
  não tem datas.
- Ficaram de fora o ALS implícito (biblioteca `implicit`) e o re-ranqueamento por
  diversidade (MMR), que eram opcionais.

## 10. Como reproduzir

```bash
sh scripts/download_data.sh             # dataset em data/raw/steam/
python -m recsys.tuning                 # tabelas da seção 6 (alguns minutos)
python -m recsys.train --source files   # treino, avaliação e artefato
python -m recsys.evaluation             # M0 a M3 com os parâmetros iniciais (α = 1)
```

A comparação final (seção 5) e as análises por histórico e de cold start estão no notebook
`notebooks/02_comparacao_modelos.ipynb` (dependências em `notebooks/requirements.txt`).
