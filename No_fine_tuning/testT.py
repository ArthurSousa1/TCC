# =========================================
# 1. Importando as bibliotecas
# =========================================
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =========================================
# 2. Carregando o dataset
# =========================================
df = pd.read_csv("redacoes_com_score_pred.csv")

# Converte as colunas numéricas (removendo vírgulas e transformando em float)
df["semantic_score"] = df["semantic_score"].astype(float)
df["score"] = df["score"].astype(float)

# =========================================
# 3. Teste t pareado
# =========================================
t_stat, p_value = stats.ttest_rel(df["score"], df["semantic_score"])

print("===== RESULTADOS DO TESTE T =====")
print(f"Estatística t: {t_stat:.4f}")
print(f"Valor-p: {p_value:.4f}")

# =========================================
# 4. Interpretação
# =========================================
alpha = 0.05  # nível de significância de 5%

if p_value < alpha:
    print("\n🔴 Resultado: Existe diferença estatisticamente significativa entre as médias.")
    print("→ O modelo NÃO está produzindo notas equivalentes às humanas.")
else:
    print("\n🟢 Resultado: Não há diferença estatisticamente significativa entre as médias.")
    print("→ O modelo está alinhado com as notas humanas (boa acurácia estatística).")

# =========================================
# 5. (Opcional) Cálculo da correlação
# =========================================
corr = df["score"].corr(df["semantic_score"])
print(f"\nCorrelação de Pearson: {corr:.4f}")

plt.scatter(df["score"], df["semantic_score"], alpha=0.6)
plt.title("Relação entre Nota Humana e Score Semântico")
plt.xlabel("Nota Humana (score)")
plt.ylabel("Score Semântico (modelo)")
plt.grid(True)
plt.show()