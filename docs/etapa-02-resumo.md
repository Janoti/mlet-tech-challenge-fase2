# Etapa 2 — Resumo

> Tech Challenge Fase 2 (FIAP pós-tech ML Engineering). Foco da Etapa 2:
> **Ambiente reprodutível, gerenciamento de dependências e configuração externalizada**.

## O que foi feito

| Entrega Etapa 2 (enunciado) | Implementação |
|---|---|
| `poetry.lock` commitado | Gerado e commitado — garante ambiente idêntico em qualquer máquina |
| `Pydantic Settings` + `.env` | [src/recsys/config.py](../src/recsys/config.py) + `.env` (copiado de `.env.example`) |
| Script de validação do ambiente | [scripts/validate_env.py](../scripts/validate_env.py) |
| Dependências prod/dev separadas | `pyproject.toml` — prod: torch, sklearn, mlflow, dvc / dev: ruff, pytest, ipykernel |
| EDA exploratório (bônus, boa prática) | [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) |
| Dataset enriquecido (bônus) | [src/recsys/data/generator_enriched.py](../src/recsys/data/generator_enriched.py) |

---

## 1. Gerenciamento de dependências (Poetry)

### Por que Poetry?

Duas ferramentas dominam o ecossistema Python moderno para gerenciamento de pacotes: **Poetry** e **uv**. O projeto usa Poetry porque:
- A FIAP indica Poetry explicitamente nas aulas (Aula 02 — Gerenciamento de Pacotes).
- A equipe já usa Poetry desde a Fase 1 do Tech Challenge.
- `pyproject.toml` centraliza metadados, deps, linter e pytest em arquivo único.

### O papel do `poetry.lock`

O lockfile registra **versões exatas** de todas as dependências (diretas e transitivas). Sem ele, dois `poetry install` em momentos diferentes podem instalar versões diferentes de um mesmo pacote — silenciosamente quebrando reprodutibilidade.

```text
Sem lockfile:   numpy>=2.1  →  hoje instala 2.1.3, amanhã pode instalar 2.2.0
Com lockfile:   numpy 2.1.3  →  sempre 2.1.3, em toda máquina, em todo CI
```

O lockfile **deve ser commitado** no Git (é o único arquivo grande que deve ser). Nunca adicioná-lo ao `.gitignore`.

### Separação prod vs dev

```toml
# pyproject.toml

[tool.poetry.dependencies]          # vão para a imagem Docker de runtime
torch = "^2.4"
scikit-learn = "^1.5"
mlflow = "^2.16"
dvc = "^3.0"

[tool.poetry.group.dev.dependencies]  # NÃO entram na imagem Docker
ruff = "^0.7"
pytest = "^8.3"
matplotlib = "^3.9"
ipykernel = "^6.29"
```

`poetry install --only main` instala apenas dependências de produção — usado no `Dockerfile` (Etapa 3).

---

## 2. Configuração externalizada (Pydantic Settings)

### O problema que resolve

Hard-codar valores no código é um anti-pattern clássico ("magic strings/numbers"). Se a URI do MLflow muda entre dev e prod, alterar o código exige um novo commit — o oposto de reprodutibilidade.

### Implementação

```python
# src/recsys/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    random_seed: int = Field(default=42, ge=0)
    mlflow_tracking_uri: str = Field(default="./mlruns")
    log_level: str = Field(default="INFO")
    # ...

settings = Settings()   # singleton — importar onde precisar
```

O Pydantic Settings:
1. Lê variáveis do `.env` (ou do ambiente do sistema operacional).
2. Valida tipos e restrições (`ge=0`, `gt=0`, `@field_validator`) na inicialização.
3. Falha imediatamente com mensagem clara se um valor inválido for fornecido ("fail fast").

### O arquivo `.env`

```bash
# .env  (não entra no Git — veja .gitignore)
RANDOM_SEED=42
NUM_USERS=2000
NUM_ITEMS=500
NUM_INTERACTIONS=50000
MLFLOW_TRACKING_URI=./mlruns
LOG_LEVEL=INFO
```

O `.env.example` é o template commitado no repositório. Cada dev copia para `.env` e ajusta localmente. Nunca commitar o `.env` real — pode conter tokens ou URIs de servidores internos.

---

## 3. Validação do ambiente (`validate_env.py`)

O script verifica três camadas antes de qualquer execução:

```bash
poetry run python scripts/validate_env.py
```

```
=== Validação do Ambiente ===

[ Configurações (.env) ]
  [OK] RANDOM_SEED=42
  [OK] NUM_USERS=2000
  ...

[ Dependências ]
  [OK] torch 2.4.x
  [OK] sklearn 1.5.x
  [OK] mlflow 2.16.x
  ...

[ Diretórios ]
  [OK] data/raw/
  [OK] data/interim/
  ...
```

Retorna código de saída `0` (sucesso) ou `1` (falha) — integrável ao CI/CD.

---

## 4. Dataset enriquecido (bônus)

Para tornar a base mais realista e aumentar o poder do EDA, foi criado um segundo gerador **sem modificar o original** (princípio OCP — Open/Closed):

| Dimensão adicionada | Detalhe |
|---|---|
| **Sazonalidade semanal** | Fins de semana têm 2.5× mais tráfego e funil com +3× mais purchase (6% vs 2%) |
| **Categoria de produto** | 5 categorias: eletronicos, moda, casa, esportes, beleza (fixas por item e seed) |
| **Gênero do usuário** | M/F/NB com proporções realistas (48%/47%/5%), fixos por usuário e seed |

```python
# src/recsys/data/generator_enriched.py
config = EnrichedGenerationConfig(num_users=2000, num_items=500, num_interactions=50_000)
df = EnrichedDatasetGenerator().generate(config)
# → user_id, item_id, category, user_gender, interaction_type, timestamp
```

Saída: `data/raw/interactions_enriched.parquet`.

---

## 5. EDA exploratório

O notebook [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) analisa o dataset enriquecido em 9 seções seguindo o estilo adotado na Fase 1:

| Seção | O que analisa |
|---|---|
| 0. Business Understanding | Contexto e objetivos de negócio |
| 1. Carregamento | Schema, tipos, memória |
| 2. Qualidade | Nulos, duplicatas, ranges |
| 3. Funil | view → add_to_cart → purchase |
| 3.1. Categoria | Distribuição por segmento |
| 4. Popularidade | Cauda longa (Zipf) — implicações para cold start |
| 5. Atividade | Distribuição de interações por usuário |
| 5.1. Gênero | Fairness — distribuição de compras por gênero |
| 6. Esparsidade | Densidade da matriz user-item |
| 7. Temporal | Evolução no tempo (janela de 90 dias) |
| 7.1. Sazonalidade | Pico de tráfego nos fins de semana |
| 8. Data Readiness | Checklist pré-modelagem |
| 9. Conclusão | Decisões de feature engineering para Etapa 3 |

---

## Alinhamento com as aulas

| Aula | Conceito-chave | Onde está no código |
|---|---|---|
| Aula 01 — Reprodutibilidade | Código + Dados + Ambiente = Resultado | `set_global_seed` + `default_rng(seed)` + `.env` + `validate_env.py` |
| Aula 02 — Poetry | `pyproject.toml` como única fonte da verdade; lockfile commitado | `pyproject.toml` + `poetry.lock` |
| Aula 03 — Controle de versões | Lockfile evita dependency hell; deps prod/dev separadas | `poetry.lock` + grupos `[dev]` isolados |

---

## Próxima etapa

- **Etapa 3** — `Dockerfile` multi-stage (base para K8s), `docker-compose.yml` com MLflow server, `dvc init` + `dvc.yaml` com estágios `preprocess → feature_eng → train → evaluate`.
