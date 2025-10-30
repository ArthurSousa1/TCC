# =========================================
# 1. Importações
# =========================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt

# =========================================
# 2. Carrega o dataset
# =========================================
df = pd.read_csv("test_with_score.csv")

# Garante que as notas estão como float
df["score"] = df["score"].astype(float)

# Divide em treino e validação
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# =========================================
# 3. Prepara os dados
# =========================================
train_examples = [
    InputExample(texts=[row["prompt"], row["essay"]], label=float(row["score"]) / 10)
    for _, row in train_df.iterrows()
]
val_examples = [
    InputExample(texts=[row["prompt"], row["essay"]], label=float(row["score"]) / 10)
    for _, row in val_df.iterrows()
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# =========================================
# 4. Carrega o modelo base multilíngue
# =========================================
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# =========================================
# 5. Define função de perda e avaliador
# =========================================
train_loss = losses.CosineSimilarityLoss(model)
val_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_examples, name="tcc-eval")

# =========================================
# 6. Treinamento (fine-tuning)
# =========================================
num_epochs = 2
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=val_evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    evaluation_steps=100,
    output_path="./modelo_tcc_finetuned"
)

# =========================================
# 7. Teste e avaliação estatística
# =========================================
# Recarrega o modelo fine-tunado
model = SentenceTransformer("./modelo_tcc_finetuned")

# Cria uma cópia do conjunto de validação para não modificar o original
val_results = val_df.copy()

# Gera previsões de similaridade para o conjunto de validação
pred_scores = []
for _, row in val_results.iterrows():
    emb_prompt, emb_essay = model.encode([row["prompt"], row["essay"]])
    sim = cosine_similarity([emb_prompt], [emb_essay])[0][0]
    score_model = ((sim + 1) / 2) * 10  # converte -1–1 para 0–10
    pred_scores.append(score_model)

# Cria uma nova coluna apenas com as previsões
val_results["score_predict"] = pred_scores

# =========================================
# 8. Teste T pareado
# =========================================
t_stat, p_value = stats.ttest_rel(val_results["score"], val_results["score_predict"])

print("\n===== RESULTADOS DO TESTE T =====")
print(f"Estatística t: {t_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("\n🔴 Há diferença estatisticamente significativa entre as médias.")
else:
    print("\n🟢 Não há diferença estatisticamente significativa entre as médias (modelo alinhado às notas humanas).")

# =========================================
# 9. Estatísticas resumidas
# =========================================
print("\n===== ESTATÍSTICAS RESUMIDAS =====")
print(val_results[["score", "score_predict"]].describe().round(2))

# =========================================
# 10. Exporta resultados para CSV
# =========================================
val_results.to_csv("avaliacao_modelo_tcc.csv", index=False)
print("\n✅ Resultados exportados para 'avaliacao_modelo_tcc.csv'")

# =========================================
# 11. Correlação e análise visual
# =========================================
corr = val_results["score"].corr(val_results["score_predict"])
print(f"\nCorrelação de Pearson: {corr:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(val_results["score"], val_results["score_predict"], alpha=0.7)
plt.plot([0, 10], [0, 10], color="red", linestyle="--", label="Ideal (y=x)")
plt.title("Correlação entre Nota Humana e Score do Modelo")
plt.xlabel("Nota Humana (score)")
plt.ylabel("Score Previsto (score_predict)")
plt.legend()
plt.grid(True)
plt.show()
