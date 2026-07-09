# Deploy local em Kubernetes (k3d + Helm + ingress-nginx + nip.io)

Sobe a API de recomendação e a observabilidade (Prometheus + Grafana) num cluster
Kubernetes local (k3d), expostos por Ingress com DNS via **nip.io** — sem editar
`/etc/hosts`. É o deploy do **bônus** do desafio: container acessível via URL.

O modelo é **bakeado na imagem** (`MODEL_SOURCE=local`): a API carrega
`embedding.pkl` + `baseline.pkl` do disco, sem depender do MLflow no boot.

## Atalho: `make k8s`

O jeito rápido — um alvo que instala as ferramentas ausentes, cria o cluster,
builda+importa a imagem, sobe o ingress-nginx e aplica o chart, imprimindo as URLs:

```bash
make k8s        # deploy completo (idempotente)
make k8s-urls   # só reimprime as URLs
make k8s-down   # derruba tudo (chart + ingress + cluster)
```

Alvos individuais (`make help` lista todos): `k8s-tools`, `k8s-cluster`,
`k8s-image`, `k8s-ingress`, `k8s-deploy`. Variáveis configuráveis: `K8S_CLUSTER`,
`K8S_NS` (default `recsys`), `K8S_IMAGE`, `K8S_TAG`, `K8S_BASE_HOST`.

O passo a passo manual abaixo faz exatamente o que o `make k8s` automatiza.

## Pré-requisitos

- Docker, `k3d` (v5+), `kubectl`, `helm` (v3+/v4). `make k8s-tools` instala os que faltarem.
- Portas 80/443 livres no host (o loadbalancer do k3d as mapeia).
- Espaço em disco: a imagem de serving tem ~3.5GB; com pouco espaço livre o k3s
  entra em `DiskPressure` e não agenda pods (libere com `docker system prune`).

## Passo a passo

### 1. Construir a imagem de serving com o modelo bakeado

```bash
# (opcional) regenerar os modelos: poetry run dvc repro
docker build --target serving-local -t recsys-api:local .
```

### 2. Criar o cluster

```bash
k3d cluster create fase2 \
  --servers 1 \
  --agents 2 \
  --port "80:80@loadbalancer" \
  --port "443:443@loadbalancer" \
  --k3s-arg "--disable=traefik@server:0"
```

### 3. Importar a imagem para o cluster

Como a imagem é local (sem registry), é preciso importá-la para os nodes:

```bash
k3d image import recsys-api:local -c fase2
```

### 4. Instalar o ingress-nginx

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.publishService.enabled=true

kubectl -n ingress-nginx rollout status deployment ingress-nginx-controller
```

### 5. Instalar o chart do recsys

```bash
helm install recsys deploy/helm/recsys --namespace recsys --create-namespace

# aguardar os pods ficarem prontos
kubectl -n recsys rollout status deployment/recsys-api
```

## URLs (nip.io resolve para 127.0.0.1)

Com o loadbalancer mapeando a porta 80 do host, `*.127.0.0.1.nip.io` resolve para
`127.0.0.1` e o Ingress roteia por host:

| Serviço | URL |
|---|---|
| API — Swagger | http://api.127.0.0.1.nip.io/docs |
| API — health | http://api.127.0.0.1.nip.io/health |
| API — métricas | http://api.127.0.0.1.nip.io/metrics |
| Prometheus | http://prometheus.127.0.0.1.nip.io |
| Grafana (admin/admin) | http://grafana.127.0.0.1.nip.io |

## Como testar

```bash
# health
curl -s http://api.127.0.0.1.nip.io/health

# recomendação para um usuário conhecido (source=embedding)
curl -s "http://api.127.0.0.1.nip.io/recommendations/1?k=5"

# usuário desconhecido -> fallback de popularidade (source=fallback)
curl -s "http://api.127.0.0.1.nip.io/recommendations/999999?k=5"

# métricas Prometheus
curl -s http://api.127.0.0.1.nip.io/metrics | grep recsys_
```

No Grafana, o dashboard **"Recsys API"** já vem provisionado (RPS, latência p95,
taxa de fallback, modelo carregado). No Prometheus, confira o target `recsys-api` em
Status → Targets.

## Configuração (Helm values)

Override com `--set` ou um arquivo `-f`:

```bash
helm upgrade recsys deploy/helm/recsys -n recsys \
  --set image.tag=local \
  --set ingress.baseHost=127.0.0.1.nip.io \
  --set grafana.enabled=true \
  --set prometheus.enabled=true
```

Principais valores (ver [`deploy/helm/recsys/values.yaml`](../deploy/helm/recsys/values.yaml)):
`image.repository/tag`, `ingress.baseHost`, `api.replicas`, `api.modelSource`
(`local` | `registry`), `prometheus.enabled`, `grafana.enabled`.

## Limpeza

```bash
# atalho: make k8s-down
helm uninstall recsys -n recsys
helm uninstall ingress-nginx -n ingress-nginx
k3d cluster delete fase2
```

## Notas

- Um IP diferente de 127.0.0.1? Ajuste `ingress.baseHost` (ex.: `192.168.0.10.nip.io`).
- `image.pullPolicy` é `IfNotPresent` — combina com a imagem importada via `k3d image import`.
- `MODEL_SOURCE=registry` faria a API carregar do MLflow Registry; exige um MLflow
  acessível com versão em `Production` (fora do escopo deste deploy local).
