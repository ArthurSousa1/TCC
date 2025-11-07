# =========================================
# 1. Importações
# =========================================
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =========================================
# 2. Configurações
# =========================================
base_path = "outputCSV"  # pasta com os arquivos dos modelos
human_file = "datasets/redacoes_com_score.csv"  # notas humanas
alpha = 0.05  # nível de significância

# =========================================
# 3. Carrega notas humanas
# =========================================
print("📥 Carregando dataset humano...")
human_df = pd.read_csv(human_file)
human_df["score"] = human_df["score"].astype(str).str.replace(",", ".").astype(float)
print(f"✅ Dataset humano carregado com {len(human_df)} registros.\n")

# =========================================
# 4. Lista de arquivos dos modelos
# =========================================
model_files = [f for f in os.listdir(base_path) if f.endswith(".csv")]
if not model_files:
    print("⚠️ Nenhum arquivo de modelo encontrado em 'outputCSV'.")
    exit()

# =========================================
# 5. Loop sobre os modelos
# =========================================
results = []

for file in model_files:
    model_path = os.path.join(base_path, file)
    print(f"📄 Analisando modelo: {file}")

    try:
        model_df = pd.read_csv(model_path)
        model_df["semantic_score"] = model_df["semantic_score"].astype(str).str.replace(",", ".").astype(float)

        # Faz merge pelo texto da redação
        merged = pd.merge(
            human_df, model_df[["essay", "semantic_score"]],
            on="essay", how="inner"
        )

        # Teste T pareado
        t_stat, p_value = stats.ttest_rel(merged["score"], merged["semantic_score"])
        corr = merged["score"].corr(merged["semantic_score"])

        significancia = "🔴 Diferente (p < 0.05)" if p_value < alpha else "🟢 Similar (p ≥ 0.05)"

        results.append({
            "Modelo": file.replace("redacoes_com_score_", "").replace(".csv", ""),
            "N": len(merged),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "correlacao": round(corr, 4),
            "significancia": significancia
        })

        print(f"  → t = {t_stat:.4f} | p = {p_value:.4f} | corr = {corr:.4f} | {significancia}\n")

    except Exception as e:
        print(f"❌ Erro com {file}: {e}\n")

# =========================================
# 6. Tabela resumo
# =========================================
if results:
    summary = pd.DataFrame(results)
    print("\n===== 📊 RESULTADOS COMPARATIVOS =====")
    print(summary.to_string(index=False))

    # Salva resumo em CSV
    summary.to_csv("Stats/resultados_testeT.csv", index=False)
    print("\n💾 Resultados salvos em: resultados_testeT.csv")

    # =========================================
    # 7. Visualização gráfica
    # =========================================
    plt.figure(figsize=(8, 5))
    plt.bar(summary["Modelo"], summary["correlacao"], color="mediumseagreen")
    plt.title("Correlação entre Nota Humana e Score Semântico por Modelo")
    plt.ylabel("Correlação de Pearson")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Nenhum resultado foi gerado.")
