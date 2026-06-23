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
RUN mkdir -p data/raw data/interim data/processed models metrics \
    && chown -R mluser:mlgroup /app
USER mluser
# Pipeline reprodutível dentro do container.
CMD ["dvc", "repro"]
