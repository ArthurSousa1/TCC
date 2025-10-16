import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import ast # Importado para converter a string da lista em uma lista real
import matplotlib.pyplot as plt # <- Biblioteca adicionada para o gráfico

# --- CONFIGURAÇÕES ---
COMPETENCIA_ALVO = 'c1'
NUMERO_DE_AMOSTRAS = 2000

# --- 1. CARREGAR E PREPARAR O DATASET DO AUTOR ---
try:
    # Use o arquivo da versão BÁSICA do dataset
    df = pd.read_csv('essay-br.csv')
    print(f"Dataset carregado com sucesso! Total de {len(df)} redações.")
except FileNotFoundError:
    print("Erro: Arquivo 'essay-br.csv' não encontrado.")
    print("Por favor, baixe o dataset do GitHub (Ipln_ufpi/essay-br) e coloque-o na mesma pasta do script.")
    exit()

if NUMERO_DE_AMOSTRAS:
    df = df.head(NUMERO_DE_AMOSTRAS)
    print(f"Usando um subconjunto de {NUMERO_DE_AMOSTRAS} amostras para um treinamento mais rápido.")


# --- 2. PRÉ-PROCESSAMENTO (CORRIGIDO PARA O DATASET BÁSICO) ---
print("\nIniciando o pré-processamento...")

df['essay'] = df['essay'].fillna('')

# --- INÍCIO DA CORREÇÃO ---
# O .csv lê a coluna 'competence' como uma string '[160, 120, ...]', precisamos convertê-la para uma lista
df['competence_list'] = df['competence'].apply(lambda x: ast.literal_eval(x))

# Agora, criamos as 5 colunas 'c1' a 'c5' a partir da lista
competences_df = pd.DataFrame(df['competence_list'].tolist(), index=df.index, columns=['c1', 'c2', 'c3', 'c4', 'c5'])

# Juntamos as novas colunas ao dataframe original
df = pd.concat([df, competences_df], axis=1)
# --- FIM DA CORREÇÃO ---


# Normalização das notas (exatamente como o autor fez)
for i in range(1, 6):
    coluna = f'c{i}'
    df[coluna] = df[coluna] / 40.0
print("Notas normalizadas para o intervalo de 0 a 5.")


# --- 3. GERAÇÃO DE EMBEDDINGS COM BERTimbau ---
print("\nCarregando o modelo BERTimbau para gerar embeddings...")
model_bert = SentenceTransformer('neuralmind/bert-base-portuguese-cased')

print("Gerando embeddings das redações... (Isso pode levar alguns minutos)")
redacoes = df['essay'].tolist()
embeddings = model_bert.encode(redacoes, show_progress_bar=True)


# --- 4. CONSTRUÇÃO DA REDE NEURAL (Fiel à arquitetura do autor) ---
print(f"\nConstruindo a rede neural para prever a nota da {COMPETENCIA_ALVO}...")

X = np.array(embeddings)
y = df[COMPETENCIA_ALVO].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_nn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model_nn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='mean_squared_error'
)

model_nn.summary()


# --- 5. TREINAMENTO E AVALIAÇÃO ---
print("\nIniciando o treinamento da rede neural...")
history = model_nn.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=4,
    validation_split=0.1,
    verbose=1
)

print("\nTreinamento concluído! Avaliando o modelo no conjunto de teste...")
loss = model_nn.evaluate(X_test, y_test, verbose=0)
print(f"Erro Quadrático Médio (MSE) no conjunto de teste: {loss:.4f}")

previsoes_normalizadas = model_nn.predict(X_test).flatten()

notas_reais = y_test * 40
notas_previstas = previsoes_normalizadas * 40
notas_previstas_arredondadas = np.round(notas_previstas / 40) * 40

print("\n--- Comparação de Notas (Reais vs. Previstas) ---")
df_resultados = pd.DataFrame({
    'Nota Real (0-200)': notas_reais.astype(int),
    'Nota Prevista (0-200)': notas_previstas_arredondadas.astype(int)
})
print(df_resultados.head(100).to_string())


# --- 6. GERAÇÃO DO GRÁFICO ---
print("\nGerando o gráfico de comparação de notas...")

# Cria a figura para o gráfico
plt.figure(figsize=(10, 8))

# Gráfico de dispersão (scatter plot) para comparar os valores
plt.scatter(df_resultados['Nota Real (0-200)'], df_resultados['Nota Prevista (0-200)'], alpha=0.7, label='Previsões do Modelo')

# Adiciona uma linha de referência para a "previsão perfeita" (onde Real = Previsto)
perfect_line = [0, 200]
plt.plot(perfect_line, perfect_line, color='r', linestyle='--', linewidth=2, label='Previsão Perfeita')

# Adiciona títulos e rótulos aos eixos
plt.title('Comparação entre Notas Reais e Previstas', fontsize=16)
plt.xlabel('Nota Real (Atribuída por Humanos)', fontsize=12)
plt.ylabel('Nota Prevista (Pelo Modelo)', fontsize=12)

# Define os marcadores dos eixos para corresponderem às notas do ENEM
ticks = [0, 40, 80, 120, 160, 200]
plt.xticks(ticks)
plt.yticks(ticks)

# Adiciona uma grade para facilitar a visualização
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Garante que a escala dos eixos seja a mesma para uma visualização correta da linha de 45 graus
plt.axis('equal')
plt.xlim(left=-10, right=210)
plt.ylim(bottom=-10, top=210)

# Salva o gráfico em um arquivo de imagem
plt.savefig('comparacao_notas.png')

print("Gráfico 'comparacao_notas.png' gerado e salvo com sucesso.")