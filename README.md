# Sistema de Recomendação E-commerce — Tech Challenge Fase 2 (Grupo 4)

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.55-13ADC7?logo=dvc&logoColor=white)](https://dvc.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Poetry](https://img.shields.io/badge/Poetry-1.8-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/badge/linted-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![CI](https://github.com/Janoti/mlet-tech-challenge-fase2/actions/workflows/ci.yml/badge.svg)](https://github.com/Janoti/mlet-tech-challenge-fase2/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Sistema de recomendação de produtos para e-commerce baseado no **comportamento de
navegação dos usuários**. O modelo central é uma rede neural (MLP ou embedding-based)
em PyTorch, com pipeline completo containerizado em Docker, dados versionados com DVC,
experimentos rastreados no MLflow e código seguindo padrões profissionais de Clean Code.

> **Status atual: Etapas 1, 2 e 3 concluídas.** As demais etapas serão implementadas
> incrementalmente — ver [§ 5 Roadmap](#5-roadmap).

## Arquitetura planejada

```mermaid
flowchart TD
    A[DVC stage: generate<br/>Strategy: PopularityBiasedStrategy] --> B[data/raw/<br/>interactions.parquet]
    B --> C[DVC stage: preprocess<br/>Etapa 3]
    C --> D[data/interim/]
    D --> E[DVC stage: feature_eng<br/>Etapa 3]
    E --> F[data/processed/]
    F --> G[DVC stage: train<br/>baseline Etapa 3 → MLP Etapa 4]
    F --> H[Baselines sklearn<br/>Etapa 4]
    G --> I[(MLflow tracking<br/>params, metrics, artifacts)]
    H --> I
    I --> J[DVC stage: evaluate<br/>Etapa 3]
    J --> K[MLflow Model Registry<br/>Staging → Production — Etapa 4]
    K --> L[Docker image<br/>multi-stage — Etapa 3]
    L --> M[Deploy<br/>Etapa 4 bônus — K8s]

    style A fill:#16A34A,color:#fff
    style B fill:#16A34A,color:#fff
    style C fill:#16A34A,color:#fff
    style D fill:#16A34A,color:#fff
    style E fill:#16A34A,color:#fff
    style F fill:#16A34A,color:#fff
    style G fill:#16A34A,color:#fff
    style I fill:#0194E2,color:#fff
    style J fill:#16A34A,color:#fff
    style K fill:#0194E2,color:#fff
    style L fill:#2496ED,color:#fff
    style M fill:#326CE5,color:#fff
```

> Legenda: nós em verde já estão implementados (Etapas 1 a 3). Demais estão como
> placeholder e serão habilitados nas próximas etapas.

## Design pattern aplicado (Strategy)

```mermaid
classDiagram
    class InteractionStrategy {
        <<abstract>>
        +sample_pairs(config, rng) tuple
    }
    class UniformInteractionStrategy {
        +sample_pairs(config, rng) tuple
    }
    class PopularityBiasedStrategy {
        -item_skew: float
        -user_skew: float
        +sample_pairs(config, rng) tuple
    }
    class DatasetGenerator {
        -strategy: InteractionStrategy
        +generate(config) DataFrame
    }
    class GenerationConfig {
        <<frozen dataclass>>
        +num_users: int
        +num_items: int
        +num_interactions: int
        +seed: int
        +time_window_days: int
    }

    InteractionStrategy <|-- UniformInteractionStrategy
    InteractionStrategy <|-- PopularityBiasedStrategy
    DatasetGenerator o--> InteractionStrategy : "depende da abstração (DIP)"
    DatasetGenerator ..> GenerationConfig : "usa"
```

## Status atual

- [x] Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`, `docs/`
- [x] Design pattern **Strategy** aplicado no gerador de dataset
- [x] Design pattern **Factory Method** aplicado na criação de geradores (`DatasetGeneratorFactory`)
- [x] Pacote `recsys` com type hints + docstrings Google style em toda função pública
- [x] `pyproject.toml` com **Poetry** (deps prod/dev separadas)
- [x] **Ruff** configurado (lint + format) — regras `E, W, F, I, B, C90, N, UP, SIM, D, ANN`
- [x] **Pre-commit hooks** (ruff + higiene de arquivos + bloqueio de arquivos >500 kb)
- [x] **GitHub Actions CI** (jobs `lint` e `test` em paralelo, push/PR contra `main`)
- [x] Script reprodutível de geração de dataset sintético (seed fixa, schema inspirado no **RetailRocket**)
- [x] Suíte de testes cobrindo schema, reprodutibilidade, Strategy pattern e validação
- [x] Documentação inicial: `README.md` + `docs/etapa-01-resumo.md`
- [x] Lock file `poetry.lock` commitado (Etapa 2)
- [x] `Pydantic Settings` + `.env` real (Etapa 2)
- [x] Script `scripts/validate_env.py` (Etapa 2)
- [x] `Dockerfile` multi-stage + `docker-compose.yml` (Etapa 3)
- [x] `dvc.yaml` com pipeline `generate → preprocess → feature_eng → train → evaluate` (Etapa 3)
- [x] MLflow tracking (params, métricas, artefatos) com servidor SQLite (Etapa 3)
- [x] Pipeline reprodutível via `dvc repro` + remote DVC local (Etapa 3)
- [x] CI extra: pipeline em miniatura no PR + scan Trivy da imagem (Etapa 3)
- [ ] MLflow Model Registry com promoção a Production (Etapa 4)
- [ ] Rede neural PyTorch (MLP / embedding) + comparação com baselines sklearn (Etapa 4)
- [ ] Model Card + vídeo STAR de 5 minutos (Etapa 4)
- [ ] Deploy bônus em nuvem (Kubernetes — alinhado com as aulas gravadas)

## Quick Start

> **Pré-requisitos:** Python 3.11+ e [Poetry](https://python-poetry.org/docs/#installation).

```bash
# 1. Clone o repositório
git clone https://github.com/Janoti/mlet-tech-challenge-fase2.git
cd mlet-tech-challenge-fase2

# 2. Instale as dependências (cria .venv automaticamente)
poetry install

# 3. Gere o dataset sintético reproduzível (50k interações, seed=42)
poetry run python scripts/generate_dataset.py

# 4. Rode os testes
poetry run pytest -v

# 5. Rode o linter
poetry run ruff check .
```

### Pipeline reprodutível (Etapa 3)

```bash
# Roda a pipeline completa (generate → preprocess → feature_eng → train → evaluate)
make repro          # = poetry run dvc repro
make metrics        # mostra P@K, R@K, NDCG, MAP

# Tudo containerizado: servidor MLflow (UI em http://localhost:5000) + treino
make compose-up     # = docker compose up --build
```

Ver detalhes em [docs/etapa-03-resumo.md](docs/etapa-03-resumo.md).

Saída esperada do passo 3:

```
... | INFO    | Configuração: GenerationConfig(num_users=2000, num_items=500, num_interactions=50000, seed=42, time_window_days=90)
... | INFO    | Dataset gerado: 50000 linhas em /…/data/raw/interactions.parquet
... | INFO    | Distribuição de tipos:
view            42531
add_to_cart      6004
purchase         1465
```

## Modos de execução

| Modo | Comando | Quando usar |
|---|---|---|
| **Gerar dataset (default)** | `poetry run python scripts/generate_dataset.py` | Recria o dataset sintético com seed 42 |
| **Gerar dataset customizado** | `NUM_INTERACTIONS=100000 poetry run python scripts/generate_dataset.py` | Trocar tamanho via env var (ver `.env.example`) |
| **Gerar dataset enriquecido** | `poetry run python scripts/generate_dataset_enriched.py` | Base com sazonalidade, categorias e gênero |
| **Validar ambiente** | `poetry run python scripts/validate_env.py` | Checar settings, pacotes e diretórios |
| **Rodar testes** | `poetry run pytest -v` | Validar schema, reprodutibilidade e Strategy |
| **Rodar testes com cobertura** | `poetry run pytest --cov=recsys --cov-report=term-missing` | Análise de cobertura |
| **Lint** | `poetry run ruff check .` | Verificar qualidade do código |
| **Auto-format** | `poetry run ruff format .` | Corrigir formatação |
| **Pre-commit em todos os arquivos** | `poetry run pre-commit run --all-files` | Validar tudo antes de commitar |
| **Instalar pre-commit hooks localmente** | `poetry run pre-commit install` | Ativar checks automáticos no `git commit` |
| **CI local completo** | `make check` | Lint + format + testes antes de abrir PR |
| **Pipeline completo** | `make all` | Setup → validate → check → data → eda |

## 1. Objetivo

- Construir um sistema de recomendação a partir de **interações user-item**
  (visualização, adição ao carrinho, compra).
- Treinar uma **rede neural** (MLP ou embedding-based) em PyTorch e compará-la
  com baselines sklearn em ≥ 4 métricas.
- Garantir **reprodutibilidade ponta-a-ponta**: seeds fixos, dataset versionado
  com DVC, pipeline `dvc repro` funcional, imagem Docker otimizada.
- Rastrear experimentos no **MLflow** (parâmetros, métricas, artefatos) e
  promover o melhor modelo a **Production** via Model Registry.

## 2. Pipeline e estrutura

As Etapas 1 e 2 entregam a fundação. O fluxo completo (a ser construído nas próximas
etapas) será orquestrado pelo DVC:

```text
data/raw/  →  preprocess  →  feature_eng  →  train (MLP)  →  evaluate
                                       ↘  train (baselines sklearn)  ↗
```

Cada estágio do DVC consumirá artefatos versionados do estágio anterior. Hoje,
o nó-fonte (`data/raw/`) é populado via `scripts/generate_dataset.py` (base original)
ou `scripts/generate_dataset_enriched.py` (base com sazonalidade, categorias e gênero).

## 3. Dataset

### 3.1 Base: RetailRocket E-commerce

O script `scripts/generate_dataset.py` gera **dados sintéticos** inspirados no
[**RetailRocket E-commerce dataset**](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset),
o mais aderente ao enunciado entre os três sugeridos (Instacart, RetailRocket,
MovieLens) por focar em **comportamento de navegação real** (não em ratings).

Mapeamento de schema (gerado ↔ RetailRocket original):

| Coluna gerada | Tipo | RetailRocket original | Equivalência |
|---|---|---|---|
| `user_id` | `int32` | `visitorid` | Identificador do usuário |
| `item_id` | `int32` | `itemid` | Identificador do produto |
| `interaction_type` | `str` | `event` | `view` / `add_to_cart` / `purchase` |
| `timestamp` | `datetime64[s]` | `timestamp` | Momento da interação (UTC) |

### 3.2 Características do dataset gerado

| Característica | Valor default | Configurável via |
|---|---|---|
| Número de usuários | 2.000 | `NUM_USERS` |
| Número de itens | 500 | `NUM_ITEMS` |
| Número de interações | 50.000 | `NUM_INTERACTIONS` |
| Janela temporal | 90 dias | (futuramente exposto) |
| Distribuição de tipos | 85 % view / 12 % add_to_cart / 3 % purchase | (constante) |
| Estratégia de amostragem | `PopularityBiasedStrategy` (Zipf) | injeção do construtor |
| Seed | 42 | `RANDOM_SEED` |

### 3.3 Dataset enriquecido (Etapa 2)

O script `scripts/generate_dataset_enriched.py` gera uma versão alternativa mais realista,
com três dimensões adicionais:

| Dimensão | Detalhe |
|---|---|
| **Sazonalidade semanal** | Fins de semana têm 2.5× mais tráfego e funil com mais purchase (6% vs 2%) |
| **Categoria de produto** | 5 categorias: eletronicos, moda, casa, esportes, beleza |
| **Gênero do usuário** | M / F / NB com proporções realistas (48% / 47% / 5%) |

Saída: `data/raw/interactions_enriched.parquet` com colunas `user_id`, `item_id`, `category`, `user_gender`, `interaction_type`, `timestamp`.

### 3.4 Por que sintético em vez de baixar o RetailRocket?

- **Reprodutibilidade total** — duas execuções com a mesma seed produzem
  exatamente o mesmo `parquet` (verificado em [test_generator.py:65](tests/data/test_generator.py#L65)).
- **Velocidade de avaliação** — o avaliador não precisa baixar ~300 MB do
  Kaggle: gera o dataset em segundos.
- **Foco didático** — concentra a complexidade no modelo, não no ETL do
  dataset real.
- **Substituível na Etapa 3** — o dataset será versionado pelo DVC. Trocar
  por RetailRocket real (ou MovieLens) é alterar apenas o stage `preprocess`,
  sem mexer no resto do pipeline.

## 4. Estrutura do repositório

```text
mlet-tech-challenge-fase2/
├── .github/workflows/
│   └── ci.yml                       # CI: lint (ruff) + test (pytest) em paralelo
├── data/
│   ├── raw/                         # Dataset bruto (versionado via DVC na Etapa 3)
│   ├── interim/                     # Artefatos intermediários
│   └── processed/                   # Splits prontos para treino
├── docs/
│   ├── etapa-01-resumo.md           # Resumo detalhado da Etapa 1
│   └── etapa-02-resumo.md           # Resumo detalhado da Etapa 2
├── models/                          # Artefatos de modelo (MLflow Registry, Etapa 4)
├── notebooks/
│   └── 01_eda.ipynb                 # EDA do dataset enriquecido (9 seções)
├── scripts/
│   ├── generate_dataset.py          # Entrypoint CLI do gerador (base original)
│   ├── generate_dataset_enriched.py # Entrypoint CLI do gerador enriquecido
│   └── validate_env.py              # Valida ambiente: settings, pacotes, dirs
├── src/recsys/
│   ├── __init__.py
│   ├── config.py                    # Pydantic Settings — fonte única de config
│   ├── data/
│   │   ├── factory.py               # Factory Method — cria geradores por modo
│   │   ├── generator.py             # Strategy pattern + DatasetGenerator
│   │   ├── generator_enriched.py    # Gerador com sazonalidade, categoria, gênero
│   │   └── schema.py                # InteractionType (StrEnum) + constantes
│   ├── models/                      # PyTorch + baselines (Etapas 3-4)
│   ├── preprocessing/               # Estratégias de pré-processamento (Etapa 3)
│   └── utils/
│       └── seed.py                  # set_global_seed centralizado
├── tests/
│   ├── config/
│   │   └── test_config.py           # Settings: defaults, normalização, validação, env override
│   ├── data/
│   │   ├── test_factory.py          # Factory Method — criação por modo, estratégia, erro
│   │   ├── test_generator.py        # Schema · Reprodutibilidade · Strategy · Validação
│   │   └── test_generator_enriched.py # Schema · Reprodutibilidade · Validação (enriquecido)
│   └── utils/
│       ├── test_logging_utils.py
│       └── test_seed.py
├── .dockerignore                    # Pronto para Etapa 3 (Docker → K8s)
├── .env.example                     # Template de variáveis de ambiente
├── .gitignore
├── .pre-commit-config.yaml          # Hooks locais (ruff + higiene)
├── .python-version                  # 3.11
├── Makefile                         # Atalhos: make check, make data, make test, ...
├── pyproject.toml                   # Poetry + ruff + pytest (única fonte da verdade)
└── README.md
```

## 5. Roadmap

| Etapa | Foco | Entregáveis | Status |
|---|---|---|---|
| **1** | Clean Code e Estrutura | Estrutura, design patterns, linting, CI, gerador de dataset | ✅ Concluída |
| **2** | Ambiente e Dependências | `poetry.lock`, `Pydantic Settings`, `.env`, `validate_env.py` | ✅ Concluída |
| **3** | Containerização e Versionamento | `Dockerfile` multi-stage, `docker-compose`, `dvc init`, `dvc.yaml` (≥ 3 stages), MLflow tracking | ⏳ |
| **4** | Rede Neural, Registry e Entrega | MLP PyTorch, baselines sklearn (≥ 4 métricas), Model Registry → Production, Model Card, vídeo STAR | ⏳ |
| **Bônus** | Deploy em nuvem | Kubernetes (alinhado com as aulas gravadas) — URL pública acessível | ⏳ |

Detalhes em [docs/etapa-01-resumo.md](docs/etapa-01-resumo.md) e [docs/etapa-02-resumo.md](docs/etapa-02-resumo.md).

## 6. Ambiente e instalação

### 6.1 Pré-requisitos

- Python 3.11 (ver `.python-version`)
- Poetry 1.8+
- Git
- Docker Desktop (necessário a partir da Etapa 3)

### 6.2 Requisitos de sistema

**Sistemas operacionais testados:**

| SO | Versão | Status |
|---|---|---|
| Ubuntu | 22.04 LTS | ✅ Testado (CI GitHub Actions) |
| macOS | 14 Sonoma (Apple Silicon) | ✅ Testado |
| Windows | 11 | ✅ Testado |

**GPU vs CPU:**

| Etapa | GPU necessária? | Observação |
|---|---|---|
| 1 — Geração de dados | Não | Roda em qualquer CPU |
| 2 — EDA e ambiente | Não | Roda em qualquer CPU |
| 3 — Treino (PyTorch) | Recomendada | Funciona em CPU, mas lento |
| 4 — Avaliação do modelo | Recomendada | Idem |

**Instalação do PyTorch por hardware:**

```bash
# CPU apenas (Etapas 1-2 ou máquina sem GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU com CUDA 12.1 (recomendado para Etapa 3+)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# GPU com CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> **Nota:** `poetry install` baixa a versão CPU por padrão (comportamento do PyPI).
> Para treinar na GPU na Etapa 3, será necessário reinstalar o torch com o índice correto
> para a versão de CUDA do seu hardware.

### 6.3 Instalação

```bash
git clone https://github.com/Janoti/mlet-tech-challenge-fase2.git
cd mlet-tech-challenge-fase2

# Instala todas as deps (prod + dev) num .venv local
poetry install

# Ativa o hook de pre-commit (executa ruff antes de cada commit)
poetry run pre-commit install
```

### 6.4 Variáveis de ambiente

Copie `.env.example` para `.env` e ajuste se necessário:

```bash
cp .env.example .env
```

Variáveis disponíveis (resumo — ver `.env.example` para a lista completa):

| Variável | Default | Descrição |
|---|---|---|
| `RANDOM_SEED` | `42` | Semente global de reprodutibilidade |
| `NUM_USERS` | `2000` | Usuários distintos no dataset sintético |
| `NUM_ITEMS` | `500` | Itens distintos no catálogo simulado |
| `NUM_INTERACTIONS` | `50000` | Total de interações (≥ 10 000, requisito) |
| `DATA_RAW_DIR` | `data/raw` | Diretório de saída do gerador |
| `MLFLOW_TRACKING_URI` | `./mlruns` | Onde o MLflow grava as runs (Etapa 3) |
| `MLFLOW_EXPERIMENT_NAME` | `recsys-ecommerce` | Nome do experimento padrão |
| `LOG_LEVEL` | `INFO` | Verbosidade dos logs |

## 7. Geração dos dados

### 7.1 Dataset original

```bash
poetry run python scripts/generate_dataset.py
```

Gera `data/raw/interactions.parquet` com 50.000 linhas, seed=42 e viés de popularidade
(`PopularityBiasedStrategy`). Colunas: `user_id`, `item_id`, `interaction_type`, `timestamp`.

### 7.2 Dataset enriquecido

```bash
poetry run python scripts/generate_dataset_enriched.py
```

Gera `data/raw/interactions_enriched.parquet` com sazonalidade semanal, categorias de produto
e gênero do usuário. Colunas extras: `category`, `user_gender`.

### 7.3 Customização via env vars

```bash
NUM_INTERACTIONS=100000 NUM_USERS=5000 RANDOM_SEED=7 \
  poetry run python scripts/generate_dataset.py
```

### 7.4 Estratégia alternativa (uso programático)

Para gerar um dataset com distribuição uniforme (baseline ingênua, sem cauda longa):

```python
from recsys.data.generator import DatasetGenerator, GenerationConfig, UniformInteractionStrategy

config = GenerationConfig(num_users=2000, num_items=500, num_interactions=50_000, seed=42)
generator = DatasetGenerator(strategy=UniformInteractionStrategy())
df = generator.generate(config)
df.to_parquet("data/raw/interactions_uniform.parquet", index=False)
```

Adicionar uma nova estratégia (ex.: `TemporalDriftStrategy`) é apenas implementar
a interface `InteractionStrategy.sample_pairs(...)` — o `DatasetGenerator` não precisa
ser alterado (OCP).

## 8. Qualidade de código

### 8.1 Lint e formatação (Ruff)

```bash
poetry run ruff check .                  # apenas verifica
poetry run ruff check . --fix            # auto-fix do que for seguro
poetry run ruff format .                 # formata
poetry run ruff format . --check         # checa formatação (modo CI)
```

Regras ativas: `E, W, F, I, B, C90 (max-complexity=8), N, UP, SIM, D (Google), ANN`.

### 8.2 Testes (pytest)

```bash
poetry run pytest                                          # resumido + cobertura
poetry run pytest -v                                       # verbose
poetry run pytest tests/data/test_generator.py -v          # arquivo específico
poetry run pytest tests/data/test_generator.py::TestReproducibility -v
```

Suítes de teste — **59 testes, 99% de cobertura**:

| Arquivo | Classe | Cobre |
|---|---|---|
| `tests/config/test_config.py` | `TestDefaults` | Valores default do Settings |
| | `TestLogLevelNormalization` | Normalização para maiúsculas |
| | `TestValidation` | Rejeição de valores inválidos |
| | `TestEnvOverride` | Variáveis de ambiente sobrepõem defaults |
| `tests/data/test_factory.py` | `TestDatasetGeneratorFactory` | Criação por modo, estratégia customizada, modo inválido |
| `tests/data/test_generator.py` | `TestSchema` | Colunas, contagem, ranges de IDs, tipos válidos |
| | `TestReproducibility` | Mesma seed ⇒ DataFrame idêntico |
| | `TestStrategyPattern` | Zipf concentra mais que uniforme |
| | `TestConfigValidation` | Rejeita inputs inválidos |
| `tests/data/test_generator_enriched.py` | `TestSchema` | Schema enriquecido: category, user_gender, timestamp ordenado |
| | `TestReproducibility` | Reprodutibilidade bit-a-bit |
| | `TestConfigValidation` | Rejeita skew ≤ 1.0 e interações < 10k |

### 8.3 Pre-commit (local)

```bash
poetry run pre-commit install            # ativa hooks no .git/hooks/
poetry run pre-commit run --all-files    # roda manualmente em tudo
```

Hooks ativos: ruff (lint+format), trailing-whitespace, end-of-file-fixer, check-yaml,
check-toml, check-added-large-files (≤ 500 kb), check-merge-conflict, detect-private-key.

### 8.4 Padrão de logs

O projeto **não usa `print()`** — toda saída diagnóstica passa pelo módulo
[`recsys.utils.logging_utils`](src/recsys/utils/logging_utils.py), idêntico em
estilo ao adotado na Fase 1 do grupo.

**Formato canônico:**

```
2026-05-10 18:49:41,339 | INFO | recsys.scripts.generate_dataset | dataset_written | path=data/raw/interactions.parquet rows=50000
```

Componentes: `timestamp | level | logger_name | event | key1=v1 key2=v2 …`.

**Como usar em código novo:**

```python
from recsys.utils.logging_utils import get_logger, log_kv, setup_logging

setup_logging()                      # apenas nos entrypoints (scripts/CLI)
logger = get_logger(__name__)        # em qualquer módulo

logger.info("training_started")
log_kv(logger, "epoch_finished", epoch=3, loss=0.42, lr=1e-3)
```

`setup_logging()` é **idempotente** — pode ser chamada várias vezes sem
duplicar handlers (importante em testes e notebooks).

**Controlar verbosidade:**

```bash
LOG_LEVEL=DEBUG poetry run python scripts/generate_dataset.py
LOG_LEVEL=WARNING poetry run pytest -v
```

Valores aceitos: `DEBUG`, `INFO` (default), `WARNING`, `ERROR`, `CRITICAL`.
Valor inválido cai silenciosamente em `INFO` — log nunca deve quebrar a app.

### 8.5 CI (GitHub Actions)

Definido em [.github/workflows/ci.yml](.github/workflows/ci.yml). Roda em todo push
e PR contra `main`:

- **Job `lint`** — `ruff check` + `ruff format --check` (segundos).
- **Job `test`** — `poetry install` + `pytest --cov=recsys`.

Cache do `.venv` reduz execuções subsequentes de ~3 min para ~30 s após o `poetry.lock`
ser commitado na Etapa 2.

## 9. Convenções

### 9.1 Branches e commits

- **Branch principal:** `main` (protegida — alterações via PR).
- **Branches de trabalho:** `feat/<descrição>`, `fix/<descrição>`, `chore/<descrição>`,
  `docs/<descrição>`, `refactor/<descrição>`, `test/<descrição>`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/).
  Exemplos:
  - `feat(etapa-01): bootstrap estrutura, clean code e design patterns`
  - `fix(generator): valida num_interactions >= 10_000 no __post_init__`
  - `docs(readme): adiciona mermaid da arquitetura planejada`

### 9.2 Estilo de código (Clean Code)

- **Funções ≤ 20 linhas** (regra do enunciado).
- **Complexidade ciclomática ≤ 8** (configurado no ruff `mccabe.max-complexity`).
- **Type hints obrigatórias** em todas as funções públicas.
- **Docstrings Google style** com `Args:`, `Returns:`, `Raises:`.
- **SOLID**: `S` (uma responsabilidade por classe), `O` (extensível via Strategy),
  `L` (subclasses intercambiáveis), `I` (interfaces pequenas), `D` (depender de abstrações).
- **Imutabilidade** onde fizer sentido (ex.: `GenerationConfig` é `@dataclass(frozen=True)`).
- **Constantes nomeadas** em vez de magic numbers/strings.

## 10. Documentação

- [docs/etapa-01-resumo.md](docs/etapa-01-resumo.md) — resumo da Etapa 1: Clean Code, Strategy pattern, CI, gerador de dataset.
- [docs/etapa-02-resumo.md](docs/etapa-02-resumo.md) — resumo da Etapa 2: Poetry, Pydantic Settings, .env, validate_env, EDA.

Documentação adicional será criada conforme as etapas avançam:

- `docs/etapa-03-resumo.md` (Docker + DVC + MLflow)
- `docs/etapa-04-resumo.md` (PyTorch + Registry + Model Card)
- `docs/model_card.md` (Model Card final, Etapa 4)

## 11. Próximos passos imediatos

1. Iniciar Etapa 3 — `Dockerfile` multi-stage, `docker-compose.yml` com MLflow server, `dvc init` + `dvc.yaml` com ≥ 3 stages (`preprocess → feature_eng → train`).
2. Configurar MLflow tracking nos scripts de treino (Etapa 3).
3. Implementar modelo PyTorch (MLP / embedding) e baselines sklearn (Etapa 4).

## 12. Contato

Grupo 4 — Tech Challenge FIAP pós-tech (Machine Learning Engineering).
