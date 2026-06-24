# ML Canvas — Sistema de Recomendação E-commerce

> Tech Challenge Fase 2 (FIAP pós-tech ML Engineering) — Grupo 4.
> Modelo: **Machine Learning Canvas** de [Louis Dorard](https://www.louisdorard.com/machine-learning-canvas)
> (10 blocos). Documenta o problema de ML de ponta a ponta — do valor de negócio
> ao monitoramento — e amarra cada bloco ao código já existente no repositório.
>
> **Legenda de maturidade:** ✅ implementado (Etapas 1–3) · 🔜 planejado (Etapa 4) ·
> 💡 hipótese de produto (fora do escopo acadêmico).

---

## 0. Objetivo / Proposta de valor (bloco central)

Recomendar a cada usuário uma lista ordenada de **produtos relevantes** com base no
seu **comportamento de navegação** (views, adições ao carrinho e compras), de forma a
movê-lo pelo funil de e-commerce (`view → add_to_cart → purchase`) e **aumentar
engajamento e conversão**.

A recomendação é um problema de **ranking top-k com feedback implícito**: não há notas
explícitas (ratings); a relevância é inferida das interações observadas.

---

## Blocos de aprendizado (como o modelo é construído)

### 1. Fontes de dados ✅

| Fonte | Detalhe |
|---|---|
| **Dataset sintético de interações** | Gerado de forma reprodutível (seed fixa) inspirado no [RetailRocket E-commerce](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — escolhido por focar em navegação real, não em ratings. Código: [`src/recsys/data/generator.py`](../src/recsys/data/generator.py) |
| **Schema** | `user_id`, `item_id`, `interaction_type` (`view`/`add_to_cart`/`purchase`), `timestamp` — ver [`src/recsys/data/schema.py`](../src/recsys/data/schema.py) |
| **Variante enriquecida** | Adiciona `category`, `user_gender` e sazonalidade semanal — [`src/recsys/data/generator_enriched.py`](../src/recsys/data/generator_enriched.py) |
| **Versionamento** | Dados versionados com **DVC** (`dvc.yaml`, remote local); cada execução é rastreável dado↔modelo |

> Volume default: 2.000 usuários · 500 itens · 50.000 interações · janela de 90 dias.
> A distribuição segue uma lei de potência (Zipf, `PopularityBiasedStrategy`) para imitar
> a cauda longa de catálogos reais.

### 2. Coleta de dados ✅ / 💡

- **Hoje (acadêmico):** dataset gerado on-demand pelo stage `generate` do pipeline DVC
  ([`pipeline/generate.py`](../src/recsys/pipeline/generate.py)), garantindo
  reprodutibilidade bit-a-bit. Substituível pelo RetailRocket/MovieLens reais alterando
  só os stages `generate`/`preprocess`.
- **Produção (hipótese):** eventos de navegação seriam capturados por instrumentação do
  front-end (clickstream) e ingeridos em um data lake, com a mesma estrutura de schema.

### 3. Features / Engenharia de atributos ✅ → 🔜

| Estágio | Feature | Código |
|---|---|---|
| ✅ Etapa 3 | **Popularidade por item** (contagem de interações, ordenada) | [`features/build_features.py`](../src/recsys/features/build_features.py) |
| ✅ Etapa 3 | **Itens vistos por usuário** (para não recomendar o já visto) | [`models/baseline.py`](../src/recsys/models/baseline.py) |
| ✅ Etapa 3 | **Codificação de IDs** em índices contíguos (pré-requisito de embeddings) | [`preprocessing/encoder.py`](../src/recsys/preprocessing/encoder.py) |
| 🔜 Etapa 4 | **Embeddings** de usuário e item; possíveis features de `category`, `user_gender` e sinais temporais | rede neural PyTorch |

### 4. Construção dos modelos ✅ → 🔜

- **Baseline (Etapa 3):** `PopularityRecommender` — recomenda os itens globalmente mais
  populares ainda não vistos pelo usuário. Implementa a interface abstrata
  [`Recommender`](../src/recsys/models/base.py) (`fit` / `recommend`).
- **Modelo-alvo (Etapa 4):** rede neural **MLP / embedding-based** em PyTorch, que
  implementará a **mesma interface** `Recommender` e entrará no pipeline **sem alterar**
  os stages `train`/`evaluate` (Open/Closed Principle).
- **Re-treino:** `dvc repro` reexecuta apenas os stages afetados quando código, dados ou
  `params.yaml` mudam (hash de dependências). Hiperparâmetros centralizados em
  [`params.yaml`](../params.yaml).
- **Rastreamento:** cada execução registra parâmetros, métricas e artefatos no **MLflow**
  ([`tracking.py`](../src/recsys/tracking.py)); na Etapa 4, o melhor modelo é promovido a
  **Production** via Model Registry.

---

## Blocos de predição (como o modelo é usado)

### 5. Tarefa de predição ✅

- **Tipo:** ranking / recomendação top-k com feedback implícito.
- **Entidade de predição:** um usuário (`user_id`).
- **Saída:** lista ordenada de até `k` `item_id`s que o usuário ainda **não** interagiu
  (`k = 10` por default, configurável em `params.yaml`).
- **Entrada em inferência:** histórico de interações do usuário + estado do modelo treinado.

### 6. Decisões / Uso das predições ✅ → 💡

- **Hoje:** as recomendações alimentam a avaliação offline e o `metrics.json` versionado.
- **Produto (hipótese):** popular blocos do tipo *"Recomendados para você"* / *"Quem viu
  isto também viu"* na home, página de produto e carrinho, priorizando itens com maior
  probabilidade de avançar no funil.

### 7. Realização das predições ✅ → 💡

- **Hoje:** **batch/offline** — o stage `train` serializa o modelo
  (`models/baseline.pkl`) e o stage `evaluate` gera recomendações para os usuários do
  conjunto de teste. Tudo containerizado (`docker compose up`).
- **Produto (hipótese):** servir top-k via API/serviço containerizado (caminho natural
  para o deploy bônus em Kubernetes), com recomendações pré-computadas em batch e/ou
  geradas sob demanda.

### 8. Avaliação offline ✅

Quatro métricas de ranking (≥ 4, requisito do enunciado), em
[`evaluation/metrics.py`](../src/recsys/evaluation/metrics.py):

| Métrica | O que mede |
|---|---|
| **Precision@k** | Fração das top-k recomendações que são relevantes |
| **Recall@k** | Fração dos itens relevantes recuperados nas top-k posições |
| **NDCG@k** | Qualidade do ranking com desconto logarítmico por posição |
| **MAP@k** | Precisão média ao longo das posições relevantes |

- **Split temporal sem vazamento** ([`preprocessing/splitter.py`](../src/recsys/preprocessing/splitter.py)):
  as interações mais recentes (20%) viram teste — respeita a natureza de série temporal.
- Métricas escritas em `metrics/metrics.json` (DVC metrics) e logadas no MLflow; o CI
  comenta o `dvc metrics diff` a cada PR.

### 9. Avaliação ao vivo e monitoramento 🔜 / 💡

- **Pós-deploy (Etapa 4 bônus):** monitorar métricas de negócio reais — CTR das
  recomendações, taxa de conversão (`view → purchase`), cobertura de catálogo e diversidade.
- **Saúde do modelo:** detectar **drift** na distribuição de interações (mudança de
  popularidade, sazonalidade) e degradação das métricas offline em re-treinos periódicos.
- **Gatilho de re-treino:** queda sustentada de métricas ou volume relevante de dados
  novos dispara novo `dvc repro` + comparação no MLflow antes de promover a Production.

---

## Resumo do estado atual (Etapa 3)

| Bloco | Estado |
|---|---|
| Objetivo, Tarefa de predição, Avaliação offline | ✅ Definidos e implementados |
| Fontes de dados, Features, Construção de modelos | ✅ Baseline de popularidade no pipeline DVC + MLflow |
| Realização de predições, Decisões | ✅ Batch offline · 💡 produto hipotético |
| Avaliação ao vivo / monitoramento | 🔜 Etapa 4 (depende do deploy) |

Ver também: [docs/etapa-03-resumo.md](etapa-03-resumo.md) e [README](../README.md).
