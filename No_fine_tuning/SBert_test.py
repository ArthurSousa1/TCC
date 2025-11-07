# =========================================
# 1. Importando as bibliotecas necessárias
# =========================================
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 2. Carregando o dataset
# =========================================
# Exemplo: suponha que seu arquivo se chama "redacoes.csv"
# e possui as colunas 'essay' e 'prompt'
df = pd.read_csv("datasets/redacoes.csv")

# Mostra as primeiras linhas para garantir que está tudo certo
print(df.head())

# =========================================
# 3. Carregando o modelo de embeddings
# =========================================
# Este modelo entende bem português e é multilíngue
model_name = "paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(model_name)

# =========================================
# 4. Gerando embeddings (representações vetoriais)
# =========================================
# O modelo transforma o texto em vetores numéricos que capturam significado semântico
essay_embeddings = model.encode(df["essay"].tolist(), convert_to_numpy=True, show_progress_bar=True)
prompt_embeddings = model.encode(df["prompt"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# =========================================
# 5. Calculando similaridade coseno
# =========================================
# A similaridade coseno mede o quão próximos dois vetores estão (1 = idênticos, 0 = ortogonais)
similarities = [cosine_similarity([essay_emb], [prompt_emb])[0][0]
                for essay_emb, prompt_emb in zip(essay_embeddings, prompt_embeddings)]

# =========================================
# 6. Convertendo para escala de 0 a 10
# =========================================
# A similaridade coseno vai de -1 a 1.
# Vamos normalizar: (valor + 1) / 2 → 0–1, depois multiplicar por 10 → 0–10
scores = [((sim + 1) / 2) * 10 for sim in similarities]
scores_str = [f"{score:.2f}".replace('.', ',') for score in scores]

# Adiciona os resultados ao DataFrame
df["semantic_score"] = scores_str

# =========================================
# 7. Salvando os resultados
# =========================================
df.to_csv("outputCSV/redacoes_com_score_SBert.csv", index=False)

print("\nExemplo de saída:")
print(df[["prompt", "essay", "semantic_score"]].head())
