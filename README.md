# mlet-tech-challenge-fase2

Sistema de recomendação de produtos para e-commerce — Tech Challenge Fase 2 da
pós-tech FIAP (Machine Learning Engineering).

> **Status:** Etapa 1 (Clean Code e Estrutura). Demais etapas ainda não foram
> implementadas — ver [Roadmap](#roadmap).

---

## Visão geral do problema

Uma empresa de e-commerce precisa recomendar produtos com base no comportamento
de navegação dos usuários. O modelo central será uma rede neural (MLP ou
embedding-based) treinada com PyTorch, com pipeline completo containerizado em
Docker, dados versionados com DVC e experimentos rastreados no MLflow.

## Estrutura do projeto

A estrutura segue convenções de projetos Python modernos e aplica o princípio
de **separação de responsabilidades** (S do SOLID):

```text
.
├── configs/                # Arquivos de configuração (YAML) por etapa
├── data/                   # Dados (versionados via DVC, não Git)
│   ├── raw/                # Dados originais, imutáveis
│   ├── interim/            # Dados intermediários (limpos, mas não finais)
│   └── processed/          # Dados prontos para treino
├── models/                 # Artefatos de modelo (gerenciados pelo MLflow)
├── notebooks/              # Notebooks de exploração (NÃO entram em produção)
├── scripts/                # Scripts executáveis (entrypoints CLI)
├── src/recsys/             # Código-fonte do pacote
│   ├── data/               # Geração, schema e carregamento de dados
│   ├── models/             # Modelos (PyTorch + baselines sklearn)
│   ├── preprocessing/      # Estratégias de pré-processamento
│   └── utils/              # Utilitários (seed, logging, etc.)
└── tests/                  # Testes unitários (espelha src/)
```

## Etapa 1 — Clean Code e Estrutura ✅

Foco: projeto limpo com padrões de engenharia desde o início.

- [x] Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/`.
- [x] Naming conventions e princípios SOLID aplicados desde a primeira linha.
- [x] Design pattern **Strategy** implementado no gerador de dataset
      (`src/recsys/data/generator.py`) — permite trocar a estratégia de geração
      de interações user-item sem modificar o código cliente (OCP).
- [x] Type hints em todas as funções públicas + docstrings Google style.
- [x] `ruff` configurado e passando, com `pre-commit` hooks.
- [x] Script reprodutível de geração de dataset sintético com **seed fixa**
      (`scripts/generate_dataset.py`).

### Gerando o dataset sintético

> **Pré-requisito:** Python 3.11. As dependências serão formalizadas na Etapa 2
> (Poetry). Por enquanto, basta `pip install numpy pandas pydantic
> pydantic-settings`.

```bash
# Gera dataset reproduzível em data/raw/interactions.parquet
python scripts/generate_dataset.py

# Customizando via env vars (ver .env.example para a lista completa):
NUM_INTERACTIONS=100000 python scripts/generate_dataset.py
```

O dataset gerado contém colunas `user_id`, `item_id`, `interaction_type`
(`view` / `add_to_cart` / `purchase`) e `timestamp`, simulando comportamento de
navegação realista com **viés de popularidade** (poucos itens populares
concentram a maior parte das interações — distribuição de cauda longa, típica
de e-commerce).

## Roadmap

- **Etapa 1** — Clean Code e estrutura (esta entrega).
- **Etapa 2** — Poetry + lock file + Pydantic Settings + validação de ambiente.
- **Etapa 3** — Dockerfile multi-stage + DVC pipeline (≥ 3 stages) + MLflow.
- **Etapa 4** — MLP/embedding PyTorch + Model Registry + Model Card + vídeo.

## Convenções

- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
- **Estilo de código**: `ruff` (formatação e lint) — ver `pyproject.toml`.
- **Funções**: máximo 20 linhas (regra do enunciado).
- **Type hints**: obrigatórias em toda função pública.
