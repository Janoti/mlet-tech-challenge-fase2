# syntax=docker/dockerfile:1
# ---------- Stage 1: build (instala dependências num prefixo isolado) -------
FROM python:3.11-slim AS build
WORKDIR /build
# Camada estável primeiro (cache): dependências antes do código.
# requirements.txt é gerado do poetry.lock com `make requirements`
# (poetry export --only main,pipeline) — runtime da pipeline SEM torch/CUDA.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# Instala o próprio pacote (console scripts recsys-*) usando poetry-core.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --prefix=/install .

# ---------- Stage 2: runtime (imagem enxuta, usuário não-root) --------------
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN addgroup --system mlgroup && adduser --system --ingroup mlgroup mluser
COPY --from=build /install /usr/local
COPY src ./src
COPY dvc.yaml params.yaml ./
# .dvc/config (sem cache — excluído no .dockerignore) torna a imagem um repo DVC
# autocontido: `dvc repro` roda a pipeline inteira dentro do container.
COPY .dvc ./.dvc
RUN mkdir -p data/raw data/interim data/processed models metrics \
    && chown -R mluser:mlgroup /app
USER mluser
# Pipeline reprodutível dentro do container.
CMD ["dvc", "repro"]

# ---------- Stage 3: serving (API FastAPI + torch) --------------------------
FROM python:3.11-slim AS serving
WORKDIR /app
RUN addgroup --system mlgroup && adduser --system --ingroup mlgroup mluser
COPY requirements-serving.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serving.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --prefix=/install . && cp -r /install/* /usr/local/
COPY params.yaml ./
RUN mkdir -p models && chown -R mluser:mlgroup /app
USER mluser
EXPOSE 8000
CMD ["uvicorn", "recsys.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Stage 4: serving-local (modelos bakeados p/ deploy imutável) ----
# Estende `serving` e embute os modelos treinados; a API os carrega do disco
# (MODEL_SOURCE=local), sem depender do MLflow no boot. Usado no deploy k8s.
FROM serving AS serving-local
COPY --chown=mluser:mlgroup models/embedding.pkl models/baseline.pkl ./models/
ENV MODEL_SOURCE=local
ENV MODEL_VERSION=local
