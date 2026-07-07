# Etapa 1 — Resumo

> Tech Challenge Fase 2 (FIAP pós-tech ML Engineering). Foco da Etapa 1:
> **Clean Code + estrutura + design patterns + linting**.

## O que foi feito

| Entrega Etapa 1 (enunciado) | Implementação |
|---|---|
| Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/` | criada |
| Naming conventions + SOLID | aplicado em todos os módulos |
| ≥ 1 design pattern (Factory/Strategy/Template Method) | **Strategy** em [src/recsys/data/generator.py](../src/recsys/data/generator.py) |
| Type hints + docstrings Google style | em todas as funções públicas |
| `ruff` sem erros + pre-commit hooks | [.pre-commit-config.yaml](../.pre-commit-config.yaml) |
| Script reprodutível de geração de dataset (seed fixa) | [scripts/generate_dataset.py](../scripts/generate_dataset.py) |
| CI (boa prática extra, não exigida) | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |

## Clean Code aplicado (princípios do livro Robert C. Martin)

- **Funções ≤ 20 linhas** (regra do enunciado) e **complexidade ciclomática ≤ 8** (configurado no ruff `mccabe.max-complexity`).
- **Naming reveladores de intenção**: `set_global_seed`, `PopularityBiasedStrategy`, `InteractionType.PURCHASE` em vez de strings mágicas.
- **SOLID**:
  - **S** — `DatasetGenerator` orquestra; cada `Strategy` sabe só amostrar pares; `set_global_seed` cuida só de seed.
  - **O** — adicionar uma nova estratégia (ex.: `TemporalDriftStrategy`) **não exige modificar** o `DatasetGenerator`.
  - **L** — toda subclasse de `InteractionStrategy` é intercambiável.
  - **I** — interface pequena (1 método: `sample_pairs`).
  - **D** — `DatasetGenerator` depende da abstração, não da implementação concreta (injeção via construtor).
- **Falhar cedo**: validações no `__post_init__` do `GenerationConfig` e no `set_global_seed`.
- **Constantes nomeadas** (`COLUMN_USER_ID`, `_INTERACTION_TYPE_PROBS`) em vez de magic numbers/strings.
- **Imutabilidade**: `GenerationConfig` é `@dataclass(frozen=True)` — não pode ser alterado após criação.

## Estrutura do repositório

```text
.
├── .github/workflows/ci.yml         # CI: ruff + pytest em cada push/PR
├── configs/                         # Configurações por etapa (YAML, Etapa 2+)
├── data/                            # Versionado por DVC (Etapa 3), não Git
│   ├── raw/                         # Dados originais imutáveis
│   ├── interim/                     # Intermediários
│   └── processed/                   # Prontos para treino
├── docs/etapa-01-resumo.md          # Este documento
├── models/                          # Artefatos (MLflow Registry, Etapa 4)
├── notebooks/                       # Exploração (não vai pra runtime)
├── scripts/
│   └── generate_dataset.py          # Entrypoint CLI do gerador
├── src/recsys/                      # Pacote principal
│   ├── data/
│   │   ├── generator.py             # Strategy pattern + DatasetGenerator
│   │   └── schema.py                # InteractionType (StrEnum) + constantes
│   ├── models/                      # PyTorch + baselines (Etapa 3-4)
│   ├── preprocessing/               # Estratégias de pré-proc (Etapa 3)
│   └── utils/seed.py                # set_global_seed centralizado
├── tests/data/test_generator.py     # 4 suítes: schema, repro, strategy, validação
├── .dockerignore                    # Pronto para Etapa 3 (Docker → K8s)
├── .env.example                     # Template de variáveis de ambiente
├── .gitignore
├── .pre-commit-config.yaml          # Hooks locais: ruff + higiene
├── .python-version                  # 3.11
├── pyproject.toml                   # Poetry + ruff + pytest (única fonte)
└── README.md
```

## Pipelines

Há **três pipelines** no projeto, em camadas diferentes:

1. **Pre-commit (local)** — `ruff check`, `ruff format`, trailing whitespace,
   EOL, bloqueio de arquivos >500kb, detecção de chaves privadas. Roda no
   próprio dev antes do commit.
2. **CI (GitHub Actions)** — `.github/workflows/ci.yml`. Jobs **lint** e
   **test** em paralelo, em cada push/PR contra `main`. Cache do `.venv`
   acelera execuções futuras.
3. **DVC pipeline** — `preprocess → feature_eng → train → evaluate`. Será
   implementado na **Etapa 3** (`dvc.yaml`).

## Dataset — baseado no RetailRocket

O script `generate_dataset.py` gera dados **sintéticos** inspirados no
[**RetailRocket E-commerce dataset**](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset),
o mais aderente ao enunciado entre os sugeridos (foco em *comportamento de
navegação*):

| Coluna gerada | RetailRocket original |
|---|---|
| `user_id` (int) | `visitorid` |
| `item_id` (int) | `itemid` |
| `interaction_type` (`view` / `add_to_cart` / `purchase`) | `event` (`view` / `addtocart` / `transaction`) |
| `timestamp` | `timestamp` |

**Diferenças intencionais:**
- 50.000 interações default (configurável via `NUM_INTERACTIONS`), acima do
  mínimo de 10.000 do enunciado. RetailRocket real tem ~2.7M.
- Viés de popularidade modelado com distribuição de **Zipf** (lei de
  potência) — reflete a cauda longa típica de e-commerce.
- Probabilidades do funil: 85% view / 12% add_to_cart / 3% purchase.
- **Seed fixa = 42** (configurável). Reprodutibilidade bit-a-bit verificada
  em teste ([test_generator.py](../tests/data/test_generator.py)).

> Por que sintético em vez de baixar o RetailRocket? **Reprodutibilidade**
> e **velocidade de iteração**. Para o avaliador rodar o pipeline sem
> precisar baixar 300MB do Kaggle, o script gera o dataset em segundos.
> Na Etapa 3, o dataset será versionado via DVC — então pode ser trocado
> pelo RetailRocket real sem alterar o restante do pipeline.

## Próximas etapas

- **Etapa 2** — `poetry install` limpo, lock file commitado, `.env` real,
  Pydantic Settings, script `validate_env.py`.
- **Etapa 3** — Dockerfile multi-stage (base para **K8s** no deploy bônus),
  `docker-compose` com MLflow server, `dvc init` + `dvc.yaml` com ≥ 3 stages.
- **Etapa 4** — MLP/embedding PyTorch, Model Registry (Staging → Production),
  Model Card, vídeo STAR (5 min), deploy em nuvem (bônus, possivelmente K8s).
