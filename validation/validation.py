import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Carregar os dados
# Certifique-se que o arquivo está na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Conj_Resultados.csv')
df = pd.read_csv(file_path)

# 2. Limpeza: Converter colunas de notas de texto (com vírgula) para número (float)
# Isso é necessário porque o Python usa ponto como separador decimal padrão
nota_columns = [col for col in df.columns if col.startswith('nota_')]
for col in nota_columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 3. Definir colunas
coluna_referencia = 'nota_original'
colunas_teste = ['nota_tcc', 'nota_gpt', 'nota_gemini']

resultados = []

for coluna in colunas_teste:
    # Remove nulos para garantir o cálculo
    temp_df = df[[coluna_referencia, coluna]].dropna()
    
    y_true = temp_df[coluna_referencia]
    y_pred = temp_df[coluna]
    
    # Cálculo das métricas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlacao = y_true.corr(y_pred)
    
    resultados.append({
        'Modelo': coluna,
        'MAE (Erro Médio)': round(mae, 4),
        'RMSE (Penaliza Erros Grandes)': round(rmse, 4),
        'Correlação (Tendência)': round(correlacao, 4)
    })

# 4. Exibir Tabela de Resultados
df_res = pd.DataFrame(resultados)
print("--- Análise de Proximidade ---")
print(df_res.to_string(index=False))

# 5. Visualização Gráfica
plt.figure(figsize=(10, 6))
sns.barplot(data=df_res, x='Modelo', y='MAE (Erro Médio)', hue='Modelo', palette='viridis', legend=False)
plt.title('Comparativo de Erro Médio Absoluto (Menor é melhor)')
plt.ylabel('Erro Médio (Pontos)')
plt.show()

# Conclusão automática baseada no menor MAE
vencedor = df_res.loc[df_res['MAE (Erro Médio)'].idxmin(), 'Modelo']
print(f"\nConclusão: O modelo '{vencedor}' é o que mais se aproxima da nota original em termos de valores absolutos.")