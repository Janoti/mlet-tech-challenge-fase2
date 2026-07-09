# =============================================================================
# Tech Challenge Fase 2 — Makefile
# -----------------------------------------------------------------------------
# Atalhos para todo o ciclo de desenvolvimento: setup, lint, testes, geração
# de dados, EDA e checks de reprodutibilidade.
#
# Uso:
#     make            # mostra ajuda (alvo default)
#     make help       # idem
#     make setup      # primeira vez: cria .env e instala dependências
#     make check      # CI local: lint + format + testes
#     make all        # pipeline completo (setup + check + data + eda)
#
# Convenção:
#     - Todo alvo é .PHONY (não gera arquivo de mesmo nome).
#     - Saída colorida pra facilitar leitura.
#     - Cada alvo imprime um cabeçalho explicando o que faz.
# =============================================================================

SHELL := /bin/bash

# ---------- Cores ANSI (fallback vazio em terminais sem suporte) -------------
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
BLUE   := \033[34m
CYAN   := \033[36m
RESET  := \033[0m

# ---------- Variáveis configuráveis ------------------------------------------
PYTHON         ?= python3.12
POETRY         ?= poetry
PARQUET_BASIC  := data/raw/interactions.parquet
PARQUET_ENRICH := data/raw/interactions_enriched.parquet
NOTEBOOK       := notebooks/01_eda.ipynb

# ---------- Deploy Kubernetes local (k3d + Helm) -----------------------------
K8S_CLUSTER    ?= fase2
K8S_NS         ?= recsys
K8S_IMAGE      ?= recsys-api
K8S_TAG        ?= local
K8S_BASE_HOST  ?= 127.0.0.1.nip.io
K8S_CHART      := deploy/helm/recsys

# Helper interno: imprime cabeçalho de seção.
define _section
	@printf "\n$(BOLD)$(CYAN)══════════════════════════════════════════════════════════════════$(RESET)\n"
	@printf "$(BOLD)$(CYAN)▶ $(1)$(RESET)\n"
	@printf "$(BOLD)$(CYAN)══════════════════════════════════════════════════════════════════$(RESET)\n"
endef

define _ok
	@printf "$(GREEN)✓ $(1)$(RESET)\n"
endef

define _warn
	@printf "$(YELLOW)⚠ $(1)$(RESET)\n"
endef


# =============================================================================
# Default: help
# =============================================================================
.DEFAULT_GOAL := help

.PHONY: help
help:  ## Mostra esta ajuda (alvo default).
	@printf "\n$(BOLD)Tech Challenge Fase 2 — Makefile$(RESET)\n"
	@printf "$(BLUE)Sistema de recomendação para e-commerce (FIAP MLE Grupo 4)$(RESET)\n\n"
	@printf "$(BOLD)Uso:$(RESET) make $(YELLOW)<alvo>$(RESET)\n\n"
	@printf "$(BOLD)Alvos disponíveis:$(RESET)\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}'
	@printf "\n$(BOLD)Fluxos comuns:$(RESET)\n"
	@printf "  $(GREEN)Primeira vez:$(RESET)   make setup && make validate\n"
	@printf "  $(GREEN)CI local:$(RESET)       make check\n"
	@printf "  $(GREEN)Pipeline total:$(RESET) make all\n"
	@printf "  $(GREEN)Reprodutibilidade:$(RESET) make repro-check\n\n"


# =============================================================================
# Setup — primeira vez
# =============================================================================
.PHONY: setup
setup: env install  ## Setup completo: cria .env e instala dependências.
	$(call _ok,Setup concluído. Próximo passo: 'make validate')

.PHONY: env
env:  ## Cria .env a partir de .env.example (se não existir).
	$(call _section,Configurando arquivo .env)
	@if [ -f .env ]; then \
		printf "$(YELLOW)→ .env já existe; nada a fazer.$(RESET)\n"; \
	else \
		cp .env.example .env; \
		printf "$(GREEN)→ .env criado a partir de .env.example$(RESET)\n"; \
		printf "$(BLUE)  Edite manualmente se precisar mudar defaults.$(RESET)\n"; \
	fi

.PHONY: install
install:  ## Instala dependências via Poetry (lockfile).
	$(call _section,Instalando dependências com Poetry)
	@printf "$(BLUE)→ poetry install (lê o poetry.lock — versões pinadas)$(RESET)\n"
	@$(POETRY) install
	$(call _ok,Dependências instaladas)

.PHONY: install-prod
install-prod:  ## Instala apenas deps de produção (sem grupo dev). Útil pro Dockerfile.
	$(call _section,Instalando apenas dependências de produção)
	@$(POETRY) install --only main
	$(call _ok,Deps de produção instaladas (sem ruff/pytest/jupyter))


# =============================================================================
# Validação — ambiente
# =============================================================================
.PHONY: validate
validate:  ## Valida ambiente: .env, deps críticas e diretórios.
	$(call _section,Validando ambiente)
	@printf "$(BLUE)→ Executa scripts/validate_env.py$(RESET)\n"
	@printf "$(BLUE)  Checa: settings do Pydantic, libs (torch/sklearn/mlflow/dvc), diretórios.$(RESET)\n\n"
	@$(POETRY) run python scripts/validate_env.py


# =============================================================================
# Qualidade de código — lint + format
# =============================================================================
.PHONY: lint
lint:  ## Roda ruff check (sem auto-fix).
	$(call _section,Lint — ruff check)
	@printf "$(BLUE)→ Verifica regras: pep8, imports, complexidade, type hints, docstrings$(RESET)\n\n"
	@$(POETRY) run ruff check .
	$(call _ok,Lint passou)

.PHONY: lint-fix
lint-fix:  ## Roda ruff check com auto-fix.
	$(call _section,Lint — ruff check --fix)
	@$(POETRY) run ruff check . --fix
	$(call _ok,Lint corrigido)

.PHONY: format-check
format-check:  ## Verifica formatação sem alterar arquivos.
	$(call _section,Format check — ruff format --check)
	@printf "$(BLUE)→ Apenas reporta arquivos fora do padrão (não altera).$(RESET)\n\n"
	@$(POETRY) run ruff format --check .
	$(call _ok,Formatação OK)

.PHONY: format
format:  ## Aplica formatação (escreve nos arquivos).
	$(call _section,Format — ruff format)
	@$(POETRY) run ruff format .
	$(call _ok,Formatação aplicada)

.PHONY: pre-commit
pre-commit:  ## Roda todos os hooks de pre-commit em todos os arquivos.
	$(call _section,Pre-commit — todos os hooks em todos os arquivos)
	@printf "$(BLUE)→ Roda os hooks definidos em .pre-commit-config.yaml$(RESET)\n\n"
	@$(POETRY) run pre-commit run --all-files

.PHONY: pre-commit-install
pre-commit-install:  ## Instala o hook de pre-commit no .git/hooks (1x).
	$(call _section,Instalando hook de pre-commit no Git)
	@$(POETRY) run pre-commit install
	$(call _ok,Hook instalado — rodará automaticamente em 'git commit')


# =============================================================================
# Testes
# =============================================================================
.PHONY: test
test:  ## Roda pytest com cobertura (config em pyproject.toml).
	$(call _section,Testes — pytest)
	@printf "$(BLUE)→ Cobertura mínima esperada: ~99%% (último commit: a6fab87)$(RESET)\n\n"
	@$(POETRY) run pytest

.PHONY: test-fast
test-fast:  ## Roda pytest sem cobertura (mais rápido pra TDD).
	$(call _section,Testes — pytest sem cobertura)
	@$(POETRY) run pytest --no-cov

.PHONY: test-cov-html
test-cov-html:  ## Gera relatório HTML de cobertura em htmlcov/.
	$(call _section,Testes — cobertura HTML)
	@$(POETRY) run pytest --cov-report=html
	$(call _ok,Relatório gerado em htmlcov/index.html)
	@printf "$(BLUE)  Abrir: xdg-open htmlcov/index.html (Linux) | start htmlcov/index.html (Windows)$(RESET)\n"

.PHONY: smoke
smoke:  ## Smoke test: importa todos os módulos principais do pacote.
	$(call _section,Smoke test — imports do pacote recsys)
	@$(POETRY) run python -c "\
from recsys.config import settings; \
from recsys.data.generator import DatasetGenerator, GenerationConfig, PopularityBiasedStrategy; \
from recsys.data.generator_enriched import EnrichedDatasetGenerator, EnrichedGenerationConfig; \
from recsys.data.factory import DatasetGeneratorFactory; \
from recsys.utils.logging_utils import get_logger; \
from recsys.utils.seed import set_global_seed; \
print('  [OK] recsys.config           — settings.random_seed =', settings.random_seed); \
print('  [OK] recsys.data.generator           (Strategy pattern)'); \
print('  [OK] recsys.data.generator_enriched  (sazonalidade + categoria + gênero)'); \
print('  [OK] recsys.data.factory             (Factory Method pattern)'); \
print('  [OK] recsys.utils.logging_utils'); \
print('  [OK] recsys.utils.seed');"
	$(call _ok,Todos os imports funcionam)


# =============================================================================
# Geração de dados
# =============================================================================
.PHONY: data
data: data-basic data-enriched  ## Gera AMBOS os datasets (basic + enriched).
	$(call _ok,Datasets gerados em data/raw/)

.PHONY: data-basic
data-basic:  ## Gera dataset original (Etapa 1 — usa Strategy pattern).
	$(call _section,Gerando dataset básico — generator.py)
	@printf "$(BLUE)→ Saída esperada: $(PARQUET_BASIC)$(RESET)\n"
	@printf "$(BLUE)→ Design pattern em destaque: Strategy (PopularityBiasedStrategy)$(RESET)\n\n"
	@$(POETRY) run python scripts/generate_dataset.py
	$(call _ok,$(PARQUET_BASIC) gerado)

.PHONY: data-enriched
data-enriched:  ## Gera dataset enriquecido (Etapa 2 — sazonalidade + categoria + gênero).
	$(call _section,Gerando dataset enriquecido — generator_enriched.py)
	@printf "$(BLUE)→ Saída esperada: $(PARQUET_ENRICH)$(RESET)\n"
	@printf "$(BLUE)→ Colunas extras: category, user_gender, sazonalidade semanal$(RESET)\n\n"
	@$(POETRY) run python scripts/generate_dataset_enriched.py
	$(call _ok,$(PARQUET_ENRICH) gerado)

.PHONY: data-info
data-info:  ## Mostra schema e amostras dos parquets gerados.
	$(call _section,Inspecionando datasets em data/raw/)
	@$(POETRY) run python -c "\
import pandas as pd; \
from pathlib import Path; \
for p in sorted(Path('data/raw').glob('*.parquet')): \
    df = pd.read_parquet(p); \
    print(f'\n=== {p.name} ==='); \
    print(f'  Linhas: {len(df):,}'); \
    print(f'  Colunas: {list(df.columns)}'); \
    print(f'  Dtypes:'); \
    print(df.dtypes.to_string()); \
    print(f'  Head:'); \
    print(df.head(3).to_string());"


# =============================================================================
# Notebook EDA
# =============================================================================
.PHONY: eda
eda:  ## Executa o notebook EDA ponta-a-ponta via nbconvert (não abre UI).
	$(call _section,Executando notebook EDA — $(NOTEBOOK))
	@printf "$(BLUE)→ Roda todas as células e descarta a saída.$(RESET)\n"
	@printf "$(BLUE)→ Falha se alguma célula der erro.$(RESET)\n\n"
	@$(POETRY) run jupyter nbconvert --to notebook --execute $(NOTEBOOK) \
		--output _executed.ipynb --output-dir /tmp
	@rm -f /tmp/_executed.ipynb
	$(call _ok,Notebook executou sem erro)

.PHONY: eda-open
eda-open:  ## Abre o notebook no Jupyter Lab (UI interativa).
	$(call _section,Abrindo Jupyter Lab)
	@$(POETRY) run jupyter lab $(NOTEBOOK)


# =============================================================================
# Reprodutibilidade — checa que mesma seed gera mesmo hash
# =============================================================================
.PHONY: repro-check
repro-check:  ## Gera datasets 2x e compara MD5 (deve ser idêntico — seed fixa).
	$(call _section,Verificando reprodutibilidade dos datasets)
	@printf "$(BLUE)→ Gera os parquets, captura hash MD5.$(RESET)\n"
	@printf "$(BLUE)→ Regera tudo e compara — hashes devem bater (seed=42 fixa).$(RESET)\n\n"
	@$(MAKE) -s data
	@HASH1=$$(md5sum data/raw/*.parquet | sort); \
	printf "\n$(YELLOW)Hash da rodada 1:$(RESET)\n$$HASH1\n\n"; \
	rm -f data/raw/*.parquet; \
	$(MAKE) -s data; \
	HASH2=$$(md5sum data/raw/*.parquet | sort); \
	printf "$(YELLOW)Hash da rodada 2:$(RESET)\n$$HASH2\n\n"; \
	if [ "$$HASH1" = "$$HASH2" ]; then \
		printf "$(GREEN)✓ Hashes idênticos — reprodutibilidade confirmada.$(RESET)\n"; \
	else \
		printf "$(RED)✗ Hashes DIFERENTES — algo está quebrando reprodutibilidade!$(RESET)\n"; \
		exit 1; \
	fi


# =============================================================================
# Pipeline DVC + MLflow + Docker (Etapa 3)
# =============================================================================
.PHONY: repro
repro:  ## Roda a pipeline DVC completa (generate→evaluate).
	$(call _section,Pipeline DVC — dvc repro)
	@$(POETRY) run dvc repro

.PHONY: metrics
metrics:  ## Mostra as métricas rastreadas pelo DVC.
	$(call _section,Métricas — dvc metrics show)
	@$(POETRY) run dvc metrics show

.PHONY: requirements
requirements:  ## Regenera requirements.txt do lock (deps de runtime: main + pipeline).
	$(call _section,Exportando requirements.txt (main,pipeline))
	@$(POETRY) export -f requirements.txt --only main,pipeline --without-hashes -o requirements.txt
	$(call _ok,requirements.txt atualizado)

.PHONY: requirements-serving
requirements-serving:  ## Exporta requirements com deps de serving (API).
	$(call _section,Exportando requirements-serving.txt (main,serving,dl))
	@$(POETRY) export --only main,serving,dl --without-hashes -f requirements.txt -o requirements-serving.txt
	$(call _ok,requirements-serving.txt atualizado)

.PHONY: docker-build
docker-build:  ## Builda a imagem Docker multi-stage (recsys:0.3.0).
	$(call _section,Docker build — recsys:0.3.0)
	@docker build -t recsys:0.3.0 .

.PHONY: compose-up
compose-up:  ## Sobe MLflow server + serviço de treino via docker compose.
	$(call _section,docker compose up (MLflow server + treino))
	@docker compose up --build

.PHONY: mlflow-ui
mlflow-ui:  ## Abre a UI do MLflow local (./mlruns) em http://localhost:5000.
	$(call _section,MLflow UI — ./mlruns)
	@$(POETRY) run mlflow ui --backend-store-uri ./mlruns


# =============================================================================
# Pipelines agregados (CI local)
# =============================================================================
.PHONY: check
check: lint format-check test  ## CI local: lint + format-check + test. Use antes de cada PR.
	$(call _ok,Tudo passou — pode commitar / abrir PR)

.PHONY: all
all: setup validate check data eda  ## Pipeline completo: setup → validate → check → data → eda.
	$(call _section,Pipeline completo terminou)
	$(call _ok,Tudo verde — projeto pronto)


# =============================================================================
# Deploy local em Kubernetes (k3d + Helm + ingress-nginx + nip.io)
# Guia completo: docs/deploy-k8s.md
# =============================================================================
.PHONY: k8s
k8s: k8s-tools k8s-cluster k8s-image k8s-ingress k8s-deploy k8s-urls  ## Deploy completo: instala ferramentas (se faltarem) → cluster → imagem → ingress → chart.
	$(call _ok,Deploy k8s no ar — veja as URLs acima)

.PHONY: k8s-tools
k8s-tools:  ## Garante k3d, kubectl e helm instalados (baixa+instala os que faltarem; Linux x86_64).
	$(call _section,Ferramentas k8s (k3d / kubectl / helm))
	@if command -v k3d >/dev/null 2>&1; then printf "$(GREEN)✓ k3d já instalado$(RESET)\n"; \
	else printf "$(YELLOW)⚠ k3d ausente — instalando...$(RESET)\n"; \
		curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash; fi
	@if command -v kubectl >/dev/null 2>&1; then printf "$(GREEN)✓ kubectl já instalado$(RESET)\n"; \
	else printf "$(YELLOW)⚠ kubectl ausente — instalando (requer sudo)...$(RESET)\n"; \
		curl -sLO "https://dl.k8s.io/release/$$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
		&& sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm -f kubectl; fi
	@if command -v helm >/dev/null 2>&1; then printf "$(GREEN)✓ helm já instalado$(RESET)\n"; \
	else printf "$(YELLOW)⚠ helm ausente — instalando...$(RESET)\n"; \
		curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash; fi

.PHONY: k8s-cluster
k8s-cluster:  ## Cria o cluster k3d (idempotente; mapeia 80/443 -> loadbalancer, desabilita traefik).
	$(call _section,Cluster k3d '$(K8S_CLUSTER)')
	@if k3d cluster list 2>/dev/null | grep -qw "$(K8S_CLUSTER)"; then \
		printf "$(GREEN)✓ cluster '$(K8S_CLUSTER)' já existe$(RESET)\n"; \
	else \
		k3d cluster create $(K8S_CLUSTER) \
			--servers 1 --agents 2 \
			--port "80:80@loadbalancer" --port "443:443@loadbalancer" \
			--k3s-arg "--disable=traefik@server:0"; \
	fi

.PHONY: k8s-image
k8s-image:  ## Builda a imagem serving-local (modelo bakeado) e importa para o cluster k3d.
	$(call _section,Imagem $(K8S_IMAGE):$(K8S_TAG) (serving-local) -> cluster)
	@docker build --target serving-local -t $(K8S_IMAGE):$(K8S_TAG) .
	@k3d image import $(K8S_IMAGE):$(K8S_TAG) -c $(K8S_CLUSTER)

.PHONY: k8s-ingress
k8s-ingress:  ## Instala o ingress-nginx no cluster (idempotente) e aguarda ficar pronto.
	$(call _section,ingress-nginx)
	@if helm status ingress-nginx -n ingress-nginx >/dev/null 2>&1; then \
		printf "$(GREEN)✓ ingress-nginx já instalado$(RESET)\n"; \
	else \
		helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx >/dev/null; \
		helm repo update >/dev/null; \
		helm install ingress-nginx ingress-nginx/ingress-nginx \
			--namespace ingress-nginx --create-namespace \
			--set controller.publishService.enabled=true; \
	fi
	@kubectl -n ingress-nginx rollout status deployment ingress-nginx-controller --timeout=180s

.PHONY: k8s-deploy
k8s-deploy:  ## Instala/atualiza o chart recsys via Helm e aguarda o rollout da API.
	$(call _section,Helm upgrade --install recsys (ns: $(K8S_NS)))
	@helm upgrade --install recsys $(K8S_CHART) \
		--namespace $(K8S_NS) --create-namespace \
		--set image.repository=$(K8S_IMAGE) \
		--set image.tag=$(K8S_TAG) \
		--set ingress.baseHost=$(K8S_BASE_HOST)
	@kubectl -n $(K8S_NS) rollout status deployment/recsys-api --timeout=180s

.PHONY: k8s-urls
k8s-urls:  ## Mostra as URLs de acesso (API/Swagger, Prometheus, Grafana).
	$(call _section,URLs (nip.io -> $(K8S_BASE_HOST)))
	@printf "  $(GREEN)API — Swagger:$(RESET)   http://api.$(K8S_BASE_HOST)/docs\n"
	@printf "  $(GREEN)API — health:$(RESET)    http://api.$(K8S_BASE_HOST)/health\n"
	@printf "  $(GREEN)API — métricas:$(RESET)  http://api.$(K8S_BASE_HOST)/metrics\n"
	@printf "  $(GREEN)Prometheus:$(RESET)      http://prometheus.$(K8S_BASE_HOST)\n"
	@printf "  $(GREEN)Grafana (admin/admin):$(RESET) http://grafana.$(K8S_BASE_HOST)\n"

.PHONY: k8s-down
k8s-down:  ## Remove o chart, o ingress-nginx e deleta o cluster k3d.
	$(call _section,Derrubando deploy k8s)
	@helm uninstall recsys -n $(K8S_NS) 2>/dev/null || true
	@helm uninstall ingress-nginx -n ingress-nginx 2>/dev/null || true
	@k3d cluster delete $(K8S_CLUSTER) 2>/dev/null || true
	$(call _ok,cluster '$(K8S_CLUSTER)' removido)


# =============================================================================
# Limpeza
# =============================================================================
.PHONY: clean
clean:  ## Remove caches (__pycache__, .pytest_cache, .ruff_cache, htmlcov, .coverage).
	$(call _section,Limpando caches)
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	$(call _ok,Caches removidos)

.PHONY: clean-data
clean-data:  ## Remove parquets gerados em data/raw/.
	$(call _section,Removendo datasets gerados)
	@rm -f data/raw/*.parquet
	$(call _ok,data/raw/*.parquet removidos)

.PHONY: clean-venv
clean-venv:  ## Remove o venv do Poetry (força reinstalação).
	$(call _section,Removendo venv do Poetry)
	@$(POETRY) env remove --all || true
	$(call _ok,Venv removido — rode 'make install' pra reinstalar)

.PHONY: clean-all
clean-all: clean clean-data clean-venv  ## Limpa TUDO: caches + parquets + venv. Volta ao zero.
	$(call _warn,Estado equivalente a clone fresco. Rode 'make setup' pra recomeçar.)


# =============================================================================
# Reprodutibilidade extrema — clone fresco simulado
# =============================================================================
.PHONY: fresh-install
fresh-install: clean-all setup validate test  ## Simula avaliador clonando do zero (clean-all → setup → validate → test).
	$(call _section,Instalação fresca completa)
	$(call _ok,Projeto sobreviveu a um 'clone fresco' simulado — reprodutibilidade garantida)
