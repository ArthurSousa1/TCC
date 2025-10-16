# ==========================================
# 1. Importações
# ==========================================
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict
import torch
import pandas as pd
import numpy as np

# ==========================================
# 2. Carregar dataset
# ==========================================
df = pd.read_csv("train_dataset_2.csv")

# Criar coluna de nota final (média das competências)
df["score"] = df[["formal_register", "thematic_coherence", 
                       "narrative_rhetorical_structure", "cohesion"]].mean(axis=1)

# Dividir em treino e teste (80/20)
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Converter para formato Hugging Face
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# ==========================================
# 3. Tokenização
# ==========================================
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def preprocess_function(examples):
    return tokenizer(
        examples["essay"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": float(x["score"])}
)

# ==========================================
# 4. Modelo e métricas
# ==========================================
model = BertForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=1
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    mse = mean_squared_error(labels, predictions)
    corr, _ = pearsonr(labels, predictions)
    return {"mse": mse, "pearson": corr}

# ==========================================
# 5. Configuração do treinamento
# ==========================================
training_args = TrainingArguments(
    output_dir="./resultados",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ==========================================
# 6. Treinar e avaliar
# ==========================================
trainer.train()
results = trainer.evaluate()
print(results)

# ==========================================
# 7. Salvar modelo
# ==========================================
trainer.save_model("./bertimbau_redacoes")
tokenizer.save_pretrained("./bertimbau_redacoes")
