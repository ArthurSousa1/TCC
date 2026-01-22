# =========================================
# 1. Importando as bibliotecas necessárias
# =========================================
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 2. Carregando o dataset
# =========================================
df = pd.read_csv("datasets/redacoes.csv")
print(df.head())

# =========================================
# 3. Carregando o modelo SimCSE
# =========================================
model_name = "melll-uff/pt-br_simcse"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================================
# 4. Função auxiliar para gerar embeddings
# =========================================
@torch.no_grad()
def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings).numpy()

# =========================================
# 5. Gerando embeddings e calculando similaridade
# =========================================
essay_embeddings = get_embeddings(df["essay"].tolist())
prompt_embeddings = get_embeddings(df["prompt"].tolist())

similarities = [cosine_similarity([e], [p])[0][0]
                for e, p in zip(essay_embeddings, prompt_embeddings)]

scores = [((sim + 1) / 2) * 10 for sim in similarities]
scores_str = [f"{s:.2f}".replace('.', ',') for s in scores]
df["semantic_score"] = scores_str

# =========================================
# 6. Salvando os resultados
# =========================================
df.to_csv("outputCSV/redacoes_com_score_melll_uff.csv", index=False)

print("\nExemplo de saída:")
print(df[["prompt", "essay", "semantic_score"]].head())
