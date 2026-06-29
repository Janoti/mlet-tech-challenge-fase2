# Etapa 4 — Modelo de Recomendação com Embeddings (MLP + PyTorch)

> Tech Challenge Fase 2 (FIAP pós-tech ML Engineering). Foco da Etapa 4:
> **Treinar um modelo MLP/embedding com PyTorch para recomendação personalizada,
> integrado ao pipeline DVC e rastreado no MLflow.**

---

## Tese central: personalização via aprendizado de representações

O baseline de popularidade (Etapa 3) recomendava os mesmos itens populares para
todo mundo. O modelo de embeddings aprende **representações densas** de usuários
e itens — vetores que capturam preferências individuais — e passa a recomendar
itens relevantes para cada usuário especificamente.

> **Usuários com gostos parecidos → vetores próximos no espaço → recomendações parecidas.**

---

## 1. Por que esse modelo?

### Alternativas consideradas

| Abordagem | Descrição | Por que descartada |
|---|---|---|
| Baseline de popularidade | Recomenda os itens globalmente mais clicados | Não é personalizado; ignora preferências individuais |
| One-hot encoding + NN | Input binário de 2500 dims (2000 users + 500 items) | Matematicamente equivalente ao embedding, porém 20× mais lento e memória |
| Matrix Factorization (SVD/NMF) | Decompõe a matriz user-item em fatores latentes | Só relações lineares; não adiciona camadas não-lineares facilmente |
| **MLP com Embeddings (escolhido)** | Lookup de vetores densos + MLP com ReLU | Ponto de equilíbrio: personalizável, não-linear, treina em minutos no CPU |
| Two-Tower / Transformers | Redes separadas por user/item; atenção sobre sequência | Requerem muito mais dados e infraestrutura; fora do escopo do dataset sintético |

### Por que MLP com Embeddings é o escolhido certo aqui

1. **Exatamente o que o enunciado pede** — *"MLP/embedding model com PyTorch"*
2. **Personalizável** — cada usuário e item tem seu próprio vetor aprendido
3. **Não-linear** — a camada ReLU captura interações que Matrix Factorization não captura
4. **Eficiente** — treino em ~2 minutos no CPU com 50k interações
5. **Extensível** — a mesma interface `Recommender` do baseline; entra no pipeline sem alterar os stages de avaliação
6. **Arquitetura consagrada** — variação do NCF (Neural Collaborative Filtering, He et al. 2017)

---

## 2. Dado de entrada: gerador enriquecido com afinidade

### Por que mudamos o gerador de dados

O gerador básico (`PopularityBiasedStrategy`) sorteia itens com distribuição Zipf
**igual para todos os usuários**. Não existe sinal de personalização — qualquer
usuário tem a mesma chance de interagir com qualquer item popular.

Nesse cenário, o embedding aprende as mesmas preferências que o baseline de
popularidade e as métricas ficam idênticas.

### Solução: afinidade usuário→categoria

O `EnrichedDatasetGenerator` ([generator_enriched.py](../src/recsys/data/generator_enriched.py))
foi estendido com dois parâmetros:

| Parâmetro | Valor padrão | Significado |
|---|---|---|
| `n_pref_categories` | 2 | Quantas das 5 categorias cada usuário prefere |
| `affinity_strength` | 3.0 | Peso extra dos itens nas categorias preferidas |

**Como funciona:**

```
Usuário 42 → categorias preferidas: ["eletrônicos", "esportes"]

Probabilidade de interagir com item de eletrônicos  = 3× maior que o normal
Probabilidade de interagir com item de moda         = peso normal (1×)
```

Matematicamente, para cada usuário um vetor de pesos é construído sobre os 500 itens:

```python
weights = where(item_categoria in preferidas, affinity_strength, 1.0)
weights = weights / weights.sum()  # normaliza para distribuição de probabilidade
item_id = rng.choice(all_items, p=weights)
```

Isso cria o **sinal de personalização** que o embedding precisa para superar o
baseline: usuários com preferências distintas interagem com conjuntos distintos
de itens.

### Schema do dado gerado

```
user_id | item_id | category    | user_gender | interaction_type | timestamp
   7    |   115   | beleza      |      F      |      view        | 2026-02-23
  845   |     0   | casa        |      M      |      view        | 2026-02-23
   21   |   142   | casa        |      F      |      purchase    | 2026-02-24
```

**50.000 interações · 2.000 usuários · 500 itens · 5 categorias · janela de 90 dias**

O modelo de embedding usa apenas `user_id` e `item_id`. As colunas `category` e
`user_gender` passam pelo pipeline e ficam disponíveis para análises de fairness
e features futuras.

---

## 3. Pré-processamento: `IdEncoder`

IDs arbitrários (`user_id=1042`) não são índices válidos para uma tabela de
embedding. O `IdEncoder` ([encoder.py](../src/recsys/preprocessing/encoder.py))
mapeia IDs para índices contíguos `0..n-1`:

```
IDs originais:  [7, 21, 845, 1966]  →  Índices: [0, 1, 2, 3]
IDs originais:  [0, 115, 142]       →  Índices: [0, 1, 2]
```

O encoder é **ajustado no dado de treino** e aplicado tanto no treino quanto na
inferência. A operação inversa (`inverse_transform`) converte índices de volta
para IDs originais na hora de retornar recomendações.

**Dois encoders independentes** são mantidos: um para usuários e outro para itens.

---

## 4. Arquitetura: `MLPScorer`

Definida em [`src/recsys/models/embedding.py`](../src/recsys/models/embedding.py).

```
user_id ──► IdEncoder ──► idx ──► Embedding(n_users, 64) ──► u ∈ ℝ⁶⁴
                                                                        │
                                                              concat([u, i]) ∈ ℝ¹²⁸
                                                                        │
item_id ──► IdEncoder ──► idx ──► Embedding(n_items, 64) ──► i ∈ ℝ⁶⁴  │
                                                                        ▼
                                                             Linear(128 → 128)
                                                                        │
                                                                      ReLU
                                                                        │
                                                             Linear(128 → 1)
                                                                        │
                                                                  logit (score)
```

### Componentes

**`nn.Embedding(n_users, emb_dim)`**
Tabela de pesos de forma `(n_users, 64)`. Cada linha é o vetor de representação
de um usuário, iniciado com Xavier uniform e atualizado por backpropagation.
O `nn.Embedding` é um lookup eficiente: ao invés de multiplicar uma matriz
`(2000, 2000)` por um vetor one-hot, simplesmente acessa uma linha diretamente.

**`concat([u, i])`**
Concatena os vetores de usuário e item em um único vetor de 128 dimensões.
Alternativa ao produto interno puro (dot product): a concatenação permite ao MLP
aprender **combinações assimétricas** — "usuário A gosta de itens do tipo X"
não é necessariamente igual a "item X é preferido por usuários do tipo A".

**`Linear(128 → 128) + ReLU`**
Camada oculta com ativação não-linear. A ReLU (`max(0, x)`) é o que diferencia
o modelo de uma regressão linear: ela permite aprender relações como
"usuário que comprou eletrônicos E acessórios tende a comprar periféricos" —
uma interação que não é captável por combinação linear.

**`Linear(128 → 1)`**
Produz um único logit (score bruto sem sigmoid). O sigmoid é aplicado pela loss
durante o treino e explicitamente na inferência.

### Hiperparâmetros (em `params.yaml`)

```yaml
embedding:
  emb_dim: 64        # dimensão dos vetores de usuário e item
  hidden_dim: 128    # neurônios na camada oculta
  lr: 0.001          # taxa de aprendizado do Adam
  epochs: 30         # épocas de treino
  batch_size: 1024   # tamanho do mini-batch
  neg_samples: 8     # amostras negativas por interação positiva
```

---

## 5. Treinamento

### Negative sampling

O dataset de interações tem apenas sinais **positivos** (o que o usuário clicou
ou comprou). Para treinar um classificador binário, precisamos de exemplos
negativos — itens que o usuário **não** quer.

Para cada interação positiva no treino geramos 8 pares negativos com itens
aleatórios do catálogo:

```
Positivo (label=1):  (user=42, item=115)
Negativos (label=0): (user=42, item=33)   ← aleatório
                     (user=42, item=201)
                     (user=42, item=87)
                     ... (8 no total)
```

Isso expande os 40.000 pares originais para **360.000 pares de treino**
(40k × 9).

**Por que amostragem aleatória é suficiente aqui?**
Com 500 itens e afinidade por categoria, a probabilidade de amostrar
acidentalmente um item "positivo" como negativo é baixa (~1%). Em datasets
maiores, negative sampling mais sofisticado (BPR, in-batch) traria ganhos.

### Função de perda: `BCEWithLogitsLoss`

Binary Cross-Entropy com sigmoid embutida:

```
loss = -[ y × log(σ(score)) + (1−y) × log(1 − σ(score)) ]
```

- Par positivo (`y=1`): penaliza score baixo → modelo aprende a dar score alto para itens relevantes
- Par negativo (`y=0`): penaliza score alto → modelo aprende a dar score baixo para itens irrelevantes

O `WithLogits` (sigmoid interna) é numericamente mais estável do que aplicar
`sigmoid` antes e depois `BCELoss` — evita underflow em scores muito negativos.

### Otimizador: Adam

Adam (`lr=0.001`) com gradientes acumulados por mini-batch de 1024 pares.
A cada passo:

1. Calcula o gradiente da loss em relação a todos os pesos
2. Atualiza os pesos na direção que reduz a loss
3. O gradiente flui de volta pelo MLP **e pelos embeddings** — os vetores de
   usuários e itens são ajustados a cada batch

Curva de convergência típica (30 épocas):

```
Época  1: loss ≈ 0.58  (modelo aleatório)
Época  5: loss ≈ 0.45
Época 15: loss ≈ 0.38
Época 30: loss ≈ 0.33  (convergindo)
```

---

## 6. Inferência: recomendação top-k

Para gerar recomendações para um usuário:

```python
# 1. Codifica o user_id para índice
user_idx = user_encoder.transform([user_id])

# 2. Calcula score contra TODOS os itens simultaneamente (vetorizado)
u_tensor = tensor([user_idx] * n_items)   # repete o usuário 500×
i_tensor = arange(n_items)                # todos os itens
scores   = sigmoid(mlp(concat(emb_u[u_tensor], emb_i[i_tensor])))

# 3. Ordena por score decrescente
ranked_indices = argsort(-scores)

# 4. Filtra itens já vistos no treino
item_ids = encoder.inverse_transform(ranked_indices)
result   = [iid for iid in item_ids if iid not in seen[user_id]][:k]
```

A etapa 2 é inteiramente vetorizada no PyTorch — não há loop sobre os 500 itens.
Em produção com catálogos de milhões de itens, essa abordagem seria substituída
por busca aproximada de vizinhos (FAISS, ScaNN), mas para 500 itens é eficiente.

---

## 7. Pipeline DVC

Os dois novos stages se encaixam no pipeline existente sem modificar os stages anteriores:

```
params.yaml
    │
    ▼
generate ──► preprocess ──► feature_eng ──► train ──► evaluate
                │                                       (baseline.pkl → metrics.json)
                │
                └──► train_embedding ──► evaluate_embedding
                      (embedding.pkl)     (metrics_embedding.json)
```

| Stage | Comando | Entrada | Saída |
|---|---|---|---|
| `train_embedding` | `recsys-train-embedding` | `train.parquet` + params | `models/embedding.pkl` |
| `evaluate_embedding` | `recsys-evaluate-embedding` | `embedding.pkl` + `test.parquet` | `metrics/metrics_embedding.json` |

O DVC rastreia mudanças em `params.yaml` (seção `embedding`) e em
`src/recsys/models/embedding.py` — qualquer alteração invalida os dois stages e
força re-execução. Stages não afetados (generate, preprocess, etc.) são
reutilizados do cache.

Cada execução de treino é registrada no MLflow com:
- Todos os parâmetros do `params.yaml` (generate + embedding)
- Loss por época (`train_loss` step-by-step)
- Métricas de capacidade (`train_n_users`, `train_n_items`)
- Tag com versão dos dados DVC

---

## 8. Métricas e resultados

### Baseline vs. Embedding (k=10)

| Métrica | Baseline (popularidade) | Embedding (MLP) | Ganho |
|---|---|---|---|
| Precision@10 | 0.0097 | **0.0123** | +27% |
| Recall@10 | 0.0210 | **0.0255** | +21% |
| NDCG@10 | 0.0154 | **0.0215** | +40% |
| MAP@10 | 0.0066 | **0.0104** | +58% |

O embedding supera o baseline em **todas as métricas**, com ganho mais
expressivo no MAP@10 (+58%) — que penaliza mais fortemente recomendações
relevantes em posições baixas. Isso indica que o embedding não só acerta mais
itens, mas os coloca em posições mais altas no ranking.

### Por que os valores absolutos são baixos

O problema ficou genuinamente mais difícil após adicionar a afinidade por
categoria:

- **Antes** (dado sem afinidade): todos os usuários interagiam com os mesmos
  itens populares. O baseline acertava porque seus top-10 globais apareciam
  no teste de qualquer usuário. Precision@10 ≈ 0.066.

- **Agora** (dado com afinidade): cada usuário interage com itens de 2
  categorias específicas. Para acertar, o modelo precisa identificar não só
  os itens populares, mas os populares **dentro da categoria certa do usuário**.
  Precision@10 ≈ 0.012 (problema 5× mais difícil).

Em dados reais, com histórico mais longo por usuário, sinais de compra
(mais fortes que view) e mais interações por item, as métricas absolutas
ficam tipicamente na faixa 0.05–0.20 para Precision@10.

### Efeito dos hiperparâmetros

| Configuração | Precision@10 | NDCG@10 | MAP@10 |
|---|---|---|---|
| v1: `emb_dim=32, epochs=10, neg=4` | 0.0103 | 0.0165 | 0.0075 |
| v2: `emb_dim=64, epochs=30, neg=8` | **0.0123** | **0.0215** | **0.0104** |

Dobrar a capacidade do embedding e triplicar as épocas de treino resultou em
+27% de Precision@10 e +39% de NDCG@10.

---

## 9. Decisões técnicas

### Interface `Recommender` (Open/Closed Principle)

O `EmbeddingRecommender` implementa a mesma interface abstrata `Recommender`
([base.py](../src/recsys/models/base.py)) que o `PopularityRecommender`.
Os stages `evaluate` e `evaluate_embedding` carregam qualquer `Recommender`
via `pickle.load` — adicionar um terceiro modelo não exige alteração nos stages
de avaliação.

### `BCEWithLogitsLoss` vs `BCELoss`

A sigmoid está **dentro** da loss (`WithLogits`) em vez de ser aplicada no
`forward()` do modelo. Razão: para scores muito negativos, `sigmoid(x)` pode
retornar exatamente `0.0` em float32 e `log(0)` causa `NaN`. A implementação
fused evita esse problema com aritmética numericamente estável.

### Dispositivo CPU

O treino roda em CPU (`device="cpu"`) por padrão — suficiente para o dataset
de 50k interações. O parâmetro `device` é exposto no construtor do
`EmbeddingRecommender` para permitir uso de GPU sem alterar o código:

```python
model = EmbeddingRecommender(device="cuda").fit(train_df)
```

### Inicialização Xavier

Os pesos dos embeddings são inicializados com `nn.init.xavier_uniform_` em vez
do padrão do PyTorch (uniforme em `[-1/√n, 1/√n]`). Xavier mantém a variância
dos gradientes estável ao longo das camadas, acelerando a convergência inicial.

### Cold-start

Usuários não presentes no dado de treino causariam `KeyError` no `IdEncoder`.
O método `recommend()` captura essa exceção e retorna lista vazia — comportamento
explícito e seguro, sem fallback silencioso para popularidade.

---

## 10. Como reproduzir

```bash
# 1. Instalar dependências (inclui PyTorch)
poetry install --with dl

# 2. Rodar o pipeline completo do zero
poetry run dvc repro

# 3. Rodar só os stages do embedding (dados já gerados)
poetry run dvc repro train_embedding evaluate_embedding

# 4. Comparar métricas baseline vs. embedding
poetry run dvc metrics show

# 5. Ver experimentos no MLflow
poetry run mlflow ui  # acessa http://localhost:5000
```

### Ajustar hiperparâmetros

Edite a seção `embedding` em [`params.yaml`](../params.yaml) e re-execute:

```bash
poetry run dvc repro train_embedding evaluate_embedding
```

O DVC detecta a mudança nos params e re-executa automaticamente apenas os
stages afetados.
