from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import os

# --- Configuração ---
# Certifique-se de que este caminho está correto (onde você salvou o modelo)
MODEL_PATH = "./bertimbau_redacoes" 
DATASET_FILE = "train_dataset_2.csv"
RANDOM_STATE = 42 # Deve ser o mesmo usado na divisão treino/teste

if not os.path.exists(MODEL_PATH):
    print(f"Erro: O diretório do modelo treinado '{MODEL_PATH}' não foi encontrado.")
    print("Verifique se o seu script de treinamento ('test.py') foi executado e salvou o modelo corretamente.")
    exit()

# --- 1. Carregar e Preparar Dataset de Teste ---
# Replicando a lógica de split do seu script original
df = pd.read_csv(DATASET_FILE)

# Criar a coluna 'score' (nota final)
df["score"] = df[["formal_register", "thematic_coherence", 
                       "narrative_rhetorical_structure", "cohesion"]].mean(axis=1)

# Divisão em treino e teste (80/20) - É CRUCIAL usar o mesmo random_state
train_df = df.sample(frac=0.8, random_state=RANDOM_STATE)
test_df = df.drop(train_df.index)

# Converter para formato Hugging Face
test_dataset = Dataset.from_pandas(test_df)

# --- 2. Tokenização ---
# Carregar o tokenizer salvo com o modelo
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH) 

def preprocess_function(examples):
    return tokenizer(
        examples["essay"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Adicionar a coluna 'labels' (essencial para o predict calcular métricas)
# (Usando a correção que foi feita no seu script)
tokenized_test_dataset = tokenized_test_dataset.map(
    lambda x: {"labels": float(x["score"])}
)


# --- 3. Função de Métricas ---
def compute_metrics(eval_pred):
    """Calcula o Erro Quadrático Médio (MSE) e a Correlação de Pearson."""
    predictions, labels = eval_pred
    
    # Para regressão, a saída das predições tem formato (N, 1), precisamos de (N,)
    predictions = np.squeeze(predictions) 
    
    # Garantir que labels também é (N,)
    if labels.ndim > 1:
         labels = np.squeeze(labels)

    mse = mean_squared_error(labels, predictions)
    corr, _ = pearsonr(labels, predictions)
    return {"mse": mse, "pearson_correlation": corr}

# --- 4. Carregar Modelo e Configurar Trainer ---
# Carregar os pesos do melhor modelo salvo
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Apenas argumentos essenciais para a avaliação
training_args = TrainingArguments(
    output_dir="./resultados_avaliacao", # Diretório de output da avaliação
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- 5. Prever e Avaliar ---
print("Realizando predições no dataset de teste...")
# predict retorna um objeto com metrics, predictions e label_ids
results = trainer.predict(tokenized_test_dataset) 

print("\n===========================================")
print("=== Resultados Finais de Assertividade ===")
print("===========================================")
print(results.metrics)
print("===========================================\n")

# Opcional: Salvar as predições para análise
predictions = results.predictions
true_labels = results.label_ids
output_df = pd.DataFrame({
    'true_score': true_labels.flatten(),
    'predicted_score': predictions.flatten()
})

output_df.to_csv("test_predictions.csv", index=False)
print(f"Detalhes das predições salvas em: test_predictions.csv")