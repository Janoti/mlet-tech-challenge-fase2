# Etapa 3 — Resumo

> Tech Challenge Fase 2 (FIAP pós-tech ML Engineering). Foco da Etapa 3:
> **Containerização (Docker) + versionamento de dados/pipeline (DVC) + rastreamento
> de experimentos (MLflow), integrados em um pipeline reprodutível.**

## Tese central: reprodutibilidade ponta a ponta

Toda a arquitetura da Etapa 3 materializa uma única ideia:

> **Código (Git) + Ambiente (Poetry) + Dados (DVC) + Resultado (MLflow) = Reprodutibilidade.**

Qualquer pessoa clona o repositório, instala o ambiente travado pelo `poetry.lock`,
roda `dvc repro` e obtém **exatamente** os mesmos dados, modelo e métricas — ou sobe
tudo containerizado com `docker compose up`.

## O que foi feito

| Entrega Etapa 3 (enunciado) | Implementação |
|---|---|
| Dockerfile multi-stage (builder + runtime) | [Dockerfile](../Dockerfile) — `python:3.11-slim`, build isola deps em `/install`, runtime enxuto e **usuário não-root** |
| `docker-compose.yml` (treino + MLflow server) | [docker-compose.yml](../docker-compose.yml) — servidor MLflow (SQLite) + serviço de treino que roda a pipeline e registra no servidor |
| `dvc init`, dataset versionado, remote | DVC inicializado, remote **local** (`./.dvcstore`), `dvc push/pull` funcionais |
| Pipeline DVC com ≥ 3 stages | `dvc.yaml` com **5 stages**: `generate → preprocess → feature_eng → train → evaluate` |
| MLflow tracking (params, métricas, artefatos) | Cada execução registra params, 4 métricas de ranking e tag de versão dos dados |
| Pipeline em miniatura no CI (extra) | [pipeline.yml](../.github/workflows/pipeline.yml) — `dvc repro` reduzido a cada PR |
| Scan de segurança da imagem (extra) | [security.yml](../.github/workflows/security.yml) — Trivy falha em CVE CRITICAL/HIGH |
| Diff de métricas no PR (extra) | comentário automático com `dvc metrics diff` |

## 1. Pipeline DVC (`dvc.yaml`) — 5 stages

```text
params.yaml ─► generate ─► preprocess ─► feature_eng ─► train ─► evaluate ─► metrics.json
              (raw)        (train/test)   (popularidade)  (baseline.pkl)   (P@K,R@K,NDCG,MAP)
```

- **`params.yaml` é a fonte única** de hiperparâmetros: lido pelo código, registrado
  no MLflow e usado pelo DVC no cálculo de hash de cada stage. Mudar um parâmetro
  invalida apenas os stages afetados — `dvc repro` reexecuta só o necessário.
- Cada stage é um **console script** fino (`recsys-generate`, `recsys-preprocess`, …)
  declarado no `pyproject.toml` e chamado pelo `dvc.yaml`. A lógica de domínio fica em
  módulos focados (`preprocessing/`, `features/`, `models/`, `evaluation/`), cada um com
  responsabilidade única e sem dependência de DVC/MLflow.
- O modelo da Etapa 3 é um **baseline de popularidade** (`PopularityRecommender`), que
  implementa a interface abstrata `Recommender`. Na Etapa 4 a rede neural PyTorch
  implementará a mesma interface e entrará na pipeline sem alterar `train`/`evaluate`.

```bash
make repro      # dvc repro — roda os 5 stages
make metrics    # dvc metrics show — P@K, R@K, NDCG, MAP
dvc push        # envia dados/modelo versionados ao remote local
```

## 2. Docker (multi-stage, imagem enxuta)

- **Estágio `build`**: instala as dependências de runtime (geradas do `poetry.lock` em
  `requirements.txt`) em um prefixo isolado e instala o próprio pacote.
- **Estágio `runtime`**: copia apenas o necessário, cria um **usuário não-root** e roda
  `dvc repro`. A imagem da Etapa 3 **não inclui o PyTorch** (só necessário na Etapa 4),
  ficando significativamente menor.
- `.dockerignore` mantém dados, modelos, notebooks e segredos fora do contexto de build.

```bash
make docker-build   # docker build -t recsys:0.3.0 .
make compose-up     # MLflow server + treino, tudo containerizado
```

## 3. MLflow tracking

- O `docker-compose.yml` sobe um **servidor MLflow** com backend **SQLite**
  (`sqlite:///mlflow.db`) e artefatos em volume persistente. O backend de banco é o que
  habilita o **Model Registry** (Etapa 4) — file store não suportaria.
- O serviço de treino aponta `MLFLOW_TRACKING_URI` para o servidor e registra, a cada
  execução: os parâmetros do `params.yaml`, as 4 métricas de ranking e uma tag com a
  versão dos dados (rastreabilidade dado↔modelo).
- Localmente, sem o compose, o tracking cai em `./mlruns` (`make mlflow-ui`).

## 4. Qualidade e automação

- Suíte de testes cobrindo métricas (valores conferidos), split temporal, encoder,
  baseline e um **smoke test** que roda a pipeline em dados reduzidos.
- CI com três frentes: lint+testes, **pipeline em miniatura** (`dvc repro` a cada PR) e
  **scan de segurança** da imagem com Trivy.
- Commits pequenos e semânticos; cada um mantém `make check` verde.

## Como reproduzir do zero

```bash
poetry install            # ambiente travado pelo poetry.lock
make repro                # roda a pipeline completa (dvc repro)
make metrics              # mostra as métricas
# ou, containerizado:
make compose-up           # MLflow server (UI em http://localhost:5000) + treino
```
