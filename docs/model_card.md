# Model Card — EmbeddingRecommender (MLP)

> Modelo de recomendação personalizada para e-commerce. Documento vivo:
> reflete a versão promovida a **Production** no MLflow Model Registry.

---

## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| Nome no Registry | `EmbeddingRecommender` |
| Estágio | `Production` (promovido via gate de qualidade) |
| Arquitetura | MLP com embeddings de usuário e item (estilo Neural Collaborative Filtering) |
| Framework | PyTorch |
| Entrada | `user_id`, `item_id` (mapeados por `IdEncoder` para índices contíguos) |
| Saída | Score de relevância por item → top-k não vistos |
| Treinamento | Negative sampling + `BCEWithLogitsLoss` + Adam, com **early stopping por NDCG de validação** |
| Reprodutibilidade | Seed global fixa (Python, NumPy, PyTorch); dataset determinístico |

Hiperparâmetros em [`params.yaml`](../params.yaml) (seção `embedding`): `emb_dim=96`,
`hidden_dim=192`, `lr=0.001`, `epochs=50` (teto), `patience=10`, `val_frac=0.2`,
`neg_samples=8`.

---

## 2. Uso pretendido

- **Objetivo:** recomendar produtos relevantes a partir do histórico de navegação/compra.
- **Usuários-alvo:** o time de personalização da loja (geração de vitrines/"para você").
- **Fora de escopo:** decisões sensíveis (crédito, preço individualizado), públicos infantis,
  ou qualquer uso que não seja recomendação de catálogo. Não é um modelo de ranking
  patrocinado nem leva em conta margem/estoque.

---

## 3. Dados de treino

Dataset **sintético** gerado com viés realista (`EnrichedDatasetGenerator`): 50.000
interações, 2.000 usuários, 500 itens, 5 categorias, janela de 90 dias. Cada usuário
tem afinidade por 2 categorias (personalização), e o funil view→cart→purchase é
respeitado. Timestamps ancorados a uma data fixa (reprodutível).

O modelo usa apenas `user_id` e `item_id`; `category` e `user_gender` trafegam no
pipeline para análises de fairness e features futuras, mas **não** entram no treino.

---

## 4. Performance

Avaliação em split de teste (holdout por interações), top-k = 10, contra dois baselines.
Métricas geradas pelo pipeline DVC ([`metrics/`](../metrics)).

| Métrica | Popularidade | SVD (MF, sklearn) | **Embedding (MLP)** |
|---|---|---|---|
| Precision@10 | 0.0093 | 0.0096 | **0.0122** |
| Recall@10 | 0.0188 | 0.0207 | **0.0264** |
| NDCG@10 | 0.0146 | 0.0158 | **0.0213** |
| MAP@10 | 0.0064 | 0.0071 | **0.0106** |

O embedding supera **ambos** os baselines nas quatro métricas. O contraste com o SVD
(fatoração linear) isola o ganho vindo da **não-linearidade** do MLP: as duas abordagens
aprendem fatores latentes, mas só o MLP modela interações não-lineares user×item.

**Gate de promoção:** a versão só é promovida de `Staging` a `Production` se superar o
baseline na métrica primária (NDCG@10) — evita colocar em produção um modelo pior.

---

## 5. Limitações

- **Cold-start:** usuários ou itens ausentes no treino não têm embedding; `recommend`
  retorna lista vazia (sem fallback silencioso para popularidade).
- **Valores absolutos baixos:** o dado sintético com afinidade por categoria é
  deliberadamente difícil; em dados reais (histórico mais longo, sinais de compra) as
  métricas absolutas tendem a ser bem maiores.
- **Escala de inferência:** o score é calculado contra todo o catálogo (OK para 500
  itens). Catálogos de milhões exigiriam busca aproximada de vizinhos (FAISS/ScaNN).
- **Sem contexto temporal/sessão:** o modelo ignora ordem e recência das interações.
- **NDCG de validação ruidoso** em dataset pequeno: mitigado com `val_frac=0.2` e
  `patience=10`, mas o early stopping pode variar em datasets muito esparsos.

---

## 6. Vieses e considerações éticas

- **Viés de popularidade:** como todo recomendador colaborativo, tende a reforçar itens
  populares (efeito "rico fica mais rico"), reduzindo diversidade e exposição da cauda longa.
- **Fairness por gênero:** o dataset carrega `user_gender`. O modelo **não** usa esse
  atributo, mas afinidades correlacionadas a gênero podem ser aprendidas indiretamente.
  Recomenda-se auditar métricas por subgrupo antes de uso real.
- **Loop de feedback:** recomendar molda o comportamento futuro, que vira dado de treino;
  sem monitoramento isso amplifica vieses ao longo do tempo.
- **Dado sintético:** não representa pessoas reais — bom para o desafio, mas nenhuma
  conclusão sobre público real deve ser tirada daqui.

---

## 7. Manutenção e reprodução

```bash
poetry install --with dl
poetry run dvc repro                 # regenera dados → treina → avalia → promove
poetry run dvc metrics show          # compara baselines vs embedding
poetry run mlflow ui                 # Models → EmbeddingRecommender (Staging/Production)
```

Cada `dvc repro` registra uma nova versão em `Staging` e, se superar o baseline, a promove
a `Production` (arquivando a anterior). Histórico de versões preservado para auditoria e rollback.
