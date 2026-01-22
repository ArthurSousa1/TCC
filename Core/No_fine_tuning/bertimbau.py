import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/redacoes.csv")
print(df.head())

model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@torch.no_grad()
def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors='pt').to(device)
        outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings.cpu())
    return torch.cat(embeddings).numpy()

essay_embeddings = get_embeddings(df["essay"].tolist())
prompt_embeddings = get_embeddings(df["prompt"].tolist())

similarities = [cosine_similarity([e], [p])[0][0]
                for e, p in zip(essay_embeddings, prompt_embeddings)]

scores = [((sim + 1) / 2) * 10 for sim in similarities]
scores_str = [f"{s:.2f}".replace('.', ',') for s in scores]
df["semantic_score"] = scores_str

df.to_csv("outputCSV/redacoes_com_score_bertimbau.csv", index=False)
print("\nExemplo de saída:")
print(df[["prompt", "essay", "semantic_score"]].head())
