# =========================================
# 1. Importando as bibliotecas necessárias
# =========================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# =========================================
# 2. Carregando o dataset
# =========================================
df = pd.read_csv("datasets/redacoes.csv")
print(df.head())

# =========================================
# 3. Carregando o modelo de embeddings (PORTULAN)
# =========================================
# Modelos disponíveis:
# - "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"  -> leve
# - "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir" -> intermediário
# - "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder" -> mais robusto

model_name = "PORTULAN/serafim-900m-portuguese-pt-sentence-encoder"
print(f"🔹 Carregando modelo {model_name} ...")
model = SentenceTransformer(model_name)

# =========================================
# 4. Gerando embeddings
# =========================================
essay_embeddings = model.encode(
    df["essay"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)
prompt_embeddings = model.encode(
    df["prompt"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# =========================================
# 5. Calculando similaridade coseno
# =========================================
similarities = [
    cosine_similarity([essay_emb], [prompt_emb])[0][0]
    for essay_emb, prompt_emb in zip(essay_embeddings, prompt_embeddings)
]

# =========================================
# 6. Convertendo para escala de 0 a 10
# =========================================
scores = [((sim + 1) / 2) * 10 for sim in similarities]
scores_str = [f"{score:.2f}".replace('.', ',') for score in scores]
df["semantic_score"] = scores_str

# =========================================
# 7. Salvando resultados
# =========================================
os.makedirs("outputCSV", exist_ok=True)
df.to_csv("outputCSV/redacoes_com_score_Portulan.csv", index=False)

print("\n✅ Execução concluída com sucesso!")
print(df[["prompt", "essay", "semantic_score"]].head())
