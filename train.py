import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
# Usamos AutoModel e AutoTokenizer para lidar com o Longformer de forma flexível
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error
import os
import warnings

# Ignora warnings do Hugging Face para manter a saída limpa
warnings.filterwarnings("ignore")

# --- 1. Configurações ---
# MODELO: Longformer Base com 4096 tokens (Pré-treinado em Inglês/Multilíngue)
MODEL_NAME = "allenai/longformer-base-4096" 
TARGETS = ['formal_register', 'thematic_coherence', 'narrative_rhetorical_structure', 'cohesion']

# AUMENTO DO CONTEXTO: O Longformer suporta até 4096 tokens
MAX_LEN = 4096 
EPOCHS = 3    # Número de épocas de treinamento (ajuste conforme o tempo disponível)

# ATENÇÃO: Reduzir drasticamente o BATCH_SIZE para 4096 tokens é crucial!
# Recomendamos 2. Se a GPU falhar por falta de memória, mude para BATCH_SIZE = 1.
BATCH_SIZE = 2 


# --- 2. Custom Dataset Class (Herança de torch.utils.data.Dataset) ---
class EssayDataset(Dataset):
    """Dataset personalizado para tokenizar pares de textos e preparar targets."""
    def __init__(self, essays, prompts, targets=None, tokenizer=None, max_len=MAX_LEN):
        self.essays = essays
        self.prompts = prompts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.essays)

    def __getitem__(self, idx):
        essay = str(self.essays[idx])
        prompt = str(self.prompts[idx])

        # Tokenização: Combina essay e prompt, usando o token [SEP] para separá-los.
        # Longformer (baseado em RoBERTa) não usa token_type_ids, por isso não o incluímos no retorno.
        encoding = self.tokenizer.encode_plus(
            essay,
            prompt,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False, # <-- Longformer não usa
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        # Longformer requer 'global_attention_mask' para o token [CLS] (primeiro token)
        # 0 = atenção local (janela deslizante), 1 = atenção global (atende a todos)
        global_attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        global_attention_mask[0] = 1 # O token CLS recebe atenção global
        item['global_attention_mask'] = global_attention_mask


        if self.targets is not None:
            # Converte as notas (targets) para um tensor float (Regressão)
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)

        return item

# --- 3. Custom Multi-Output Regression Model (Adaptado para Longformer) ---
class LongformerForMultiOutputRegression(torch.nn.Module):
    """Modelo Longformer com um 'head' de regressão para as 4 notas."""
    def __init__(self, n_outputs):
        super(LongformerForMultiOutputRegression, self).__init__()
        # Carrega o modelo base Longformer
        self.longformer = AutoModel.from_pretrained(MODEL_NAME)
        # Camada dropout
        self.dropout = torch.nn.Dropout(0.1)
        # Camada de regressão: saída para o número de targets (4)
        self.regressor = torch.nn.Linear(self.longformer.config.hidden_size, n_outputs)
        # Função de perda (Loss): MSE (Mean Squared Error) para regressão
        self.loss_fn = torch.nn.MSELoss()

    # Mude a assinatura da função forward para aceitar global_attention_mask e remover token_type_ids
    def forward(self, input_ids, attention_mask, global_attention_mask=None, labels=None):
        
        # A saída do Longformer é (last_hidden_state, pooler_output, ...)
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
            # token_type_ids REMOVIDO
        )
        
        # O Longformer/RoBERTa usa o pooler_output (representação de [CLS] para regressão)
        pooled_output = outputs['pooler_output'] 

        # Aplica dropout e a camada de regressão
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output) # Predições

        loss = None
        if labels is not None:
            # Calcula a perda
            loss = self.loss_fn(logits, labels)

        # Retorna um objeto com as predições e a perda
        return {'loss': loss, 'logits': logits}


# --- 4. Função de Métrica ---
def compute_metrics(p):
    """Calcula as métricas de avaliação (RMSE)"""
    predictions = p.predictions
    labels = p.label_ids

    # Calcula o RMSE (Root Mean Squared Error) para cada target
    rmse_scores = np.sqrt(mean_squared_error(labels, predictions, multioutput='raw_values'))

    metrics = {
        'avg_rmse': np.mean(rmse_scores),
        f'rmse_{TARGETS[0]}': rmse_scores[0],
        f'rmse_{TARGETS[1]}': rmse_scores[1],
        f'rmse_{TARGETS[2]}': rmse_scores[2],
        f'rmse_{TARGETS[3]}': rmse_scores[3],
    }
    return metrics

# --- 5. Main Execution ---
def main():
    # Carrega datasets (assumindo que estão no mesmo diretório)
    print("--- Carregando datasets... ---")
    try:
        train_df = pd.read_csv("train_dataset.csv")
        test_df = pd.read_csv("test_dataset.csv")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado. Certifique-se de que os arquivos 'train_dataset.csv' e 'test_dataset.csv' estão no mesmo diretório. Detalhe: {e}")
        return

    # Preparar targets (notas)
    y_train = train_df[TARGETS].values
    y_test = test_df[TARGETS].values

    # Inicializar Tokenizer, Modelo e Datasets
    # Usando AutoTokenizer e AutoModel para carregar o Longformer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LongformerForMultiOutputRegression(n_outputs=len(TARGETS))

    train_dataset = EssayDataset(
        essays=train_df['essay'].values,
        prompts=train_df['prompt'].values,
        targets=y_train,
        tokenizer=tokenizer
    )

    test_dataset = EssayDataset(
        essays=test_df['essay'].values,
        prompts=test_df['prompt'].values,
        targets=y_test,
        tokenizer=tokenizer
    )

    # Configurar TrainingArguments (usando a correção de compatibilidade de argumentos)
    training_args = TrainingArguments(
        output_dir='./longformer_results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./longformer_logs',
        logging_steps=100,
        
        # Argumentos Simplificados para evitar o TypeError e garantir avaliação
        do_eval=True,               
        save_total_limit=1,         
        save_steps=100,             
        
        metric_for_best_model='avg_rmse', 
        greater_is_better=False,
    )

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, 
        compute_metrics=compute_metrics
    )

    # Treinamento
    print(f"\n--- Iniciando o treinamento do Longformer ({MAX_LEN} tokens) ---")
    trainer.train()
    print("--- Treinamento Concluído ---")

    # Avaliação Final (Predição)
    print("\n--- Avaliando o melhor modelo no dataset de teste ---")
    prediction_output = trainer.predict(test_dataset)

    # As predições (notas flutuantes)
    predictions = prediction_output.predictions

    # As métricas de avaliação
    test_metrics = prediction_output.metrics
    print("\nMétricas de Avaliação no Teste (RMSE - Root Mean Squared Error):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Gerar CSV de Output
    # Arredondar as predições para o número inteiro mais próximo, limitando a [0, 5]
    predicted_scores = np.clip(np.round(predictions), 0, 5).astype(int)

    results_df = test_df[['id']].copy()

    # Adiciona as notas verdadeiras (TRUE)
    for target in TARGETS:
        results_df[f'{target}_TRUE'] = test_df[target]

    # Adiciona as notas preditas (PREDICTED)
    for i, target in enumerate(TARGETS):
        results_df[f'{target}_PREDICTED'] = predicted_scores[:, i]

    # Salvar o resultado
    output_file = "longformer_test_predictions_results.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResultados da predição salvos em: {output_file}")
    print("\n--- Resultados (Predições vs. True Scores - Primeiras 5 Linhas) ---")
    print(results_df.head())


if __name__ == '__main__':
    main()