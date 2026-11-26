# Modelo de recomendação

> Descrição completa do modelo em elaboração. Esta versão registra os resultados da
> avaliação offline e do ajuste de hiperparâmetros.

## Resultados da avaliação offline

**Protocolo** (`recsys/evaluation.py`): usuários com pelo menos 5 jogos (3.757); para cada
um, 20% dos jogos sorteados (semente 42) vão para o teste. Os modelos treinam com o
restante, recebem o histórico de treino de cada usuário e recomendam 10 jogos que ele ainda
não tem. Métricas médias @10; cobertura = fração do catálogo (5.153 jogos) que aparece em
alguma lista; novidade = autoinformação média `-log2(popularidade)` dos itens recomendados.

### Comparação final (parâmetros ajustados, α = 0,25 para todos)

| Modelo | Precision | Recall | NDCG | MAP | Hit rate | Cobertura | Novidade | ms/usuário |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 popularidade | 0,0626 | 0,1779 | 0,1399 | 0,0849 | 0,4248 | 0,7% | 3,70 | 0,02 |
| M1 item-kNN (k = 20) | 0,1369 | 0,3922 | 0,3847 | 0,3088 | 0,6803 | 37,9% | 6,27 | 0,7 |
| M2 SVD (64 fatores) | 0,1168 | 0,3335 | 0,3374 | 0,2707 | 0,6149 | 10,7% | 6,01 | 1,4 |
| M3 conteúdo | 0,0724 | 0,2142 | 0,1808 | 0,1248 | 0,4480 | 45,1% | 8,89 | 4,8 |
| **M4 híbrido** | **0,1421** | **0,4027** | **0,3975** | **0,3186** | **0,6944** | 38,5% | 6,24 | 5,7 |

O híbrido (M4) é o modelo servido pela API: é o melhor em todas as métricas de acurácia,
mantém a cobertura do item-kNN e resolve o cold start (popularidade) e os jogos novos
(conteúdo).

### Ajuste de hiperparâmetros (`python -m recsys.tuning`)

Amostra fixa de 1.500 usuários avaliados; cada etapa parte do melhor valor da anterior.

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
| **0,7 / 0,2 / 0,1** (planejamento) | **0,3743** | **0,2969** | **0,6707** | 26,3% |
| 0,8 / 0,2 / 0 | 0,3731 | 0,2967 | 0,6687 | 26,3% |
| 0,8 / 0,1 / 0,1 | 0,3711 | 0,2956 | 0,6687 | 25,9% |
| 0,9 / 0,1 / 0 | 0,3704 | 0,2950 | 0,6673 | 26,2% |
| 0,6 / 0,3 / 0,1 | 0,3700 | 0,2912 | 0,6647 | 27,5% |
| 1 / 0 / 0 (só CF) | 0,3646 | 0,2900 | 0,6607 | 26,0% |

**Pesos para histórico de 1–2 jogos.** O protocolo padrão só tem usuários com 5 ou mais
jogos, então aqui o modelo treina com todos os dados, mas recebe apenas 2 jogos do histórico
de cada usuário avaliado:

| Pesos | NDCG | MAP | Hit rate | Cobertura |
|---|---:|---:|---:|---:|
| 0,4 / 0,4 / 0,2 (planejamento) | 0,1633 | 0,1107 | 0,4273 | 50,1% |
| 0,6 / 0,2 / 0,2 | 0,1758 | 0,1204 | 0,4433 | 47,5% |
| **0,3 / 0,3 / 0,4** | **0,1840** | **0,1240** | 0,4813 | 45,9% |
| 0,3 / 0,2 / 0,5 | 0,1784 | 0,1129 | 0,5027 | 42,9% |
| 0,2 / 0,2 / 0,6 | 0,1722 | 0,1043 | 0,5120 | 38,7% |
| 1 / 0 / 0 (só CF) | 0,1679 | 0,1167 | 0,4173 | 47,6% |

Com pouco histórico, a popularidade ganha importância; acima de 0,4, o hit rate ainda sobe,
mas NDCG, MAP e cobertura caem. Escolhido: **0,3 / 0,3 / 0,4**.

### Parâmetros do modelo servido

| Parâmetro | Valor |
|---|---|
| α (score de confiança) | 0,25 |
| Vizinhos do item-kNN | 20 |
| Pesos, histórico 0 | 0 / 0 / 1 (só popularidade) |
| Pesos, histórico 1–2 | 0,3 / 0,3 / 0,4 |
| Pesos, histórico ≥ 3 | 0,7 / 0,2 / 0,1 |
| Penalidade "não gostei" | 0,5 (sem dados de feedback negativo para avaliar offline) |
