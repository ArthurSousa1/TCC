"""
Script para treinar um modelo de predição de grades baseado em respostas de alunos
Usa um CSV com colunas: Question, Base_answer, Student_answer, Grade
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# CONFIGURAÇÃO - ALTERE AQUI COM O CAMINHO REAL DO SEU CSV
GRADES_CSV = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'grading_training_data.csv')  
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'Service', 'models', 'semantic_grade_scaler.pkl')

def load_semantic_transformer():
    """Carrega o transformer de embeddings"""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def extract_features_from_response(student_answer, base_answer, similarity):
    """
    Extrai features da resposta do aluno para alimentar o modelo
    
    Features:
    - similarity: Similaridade semântica com a resposta base (0-1)
    - length_ratio: Razão do comprimento da resposta vs resposta base
    - word_count: Número de palavras na resposta
    - sentence_count: Número de sentenças
    """
    
    student_words = len(student_answer.split()) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0
    
    # Contar sentenças (simplista, apenas pontos, interrogação, exclamação)
    student_sentences = max(1, len([s for s in student_answer.split('.') if s.strip()]))
    
    length_ratio = student_words / base_words if base_words > 0 else 0
    
    return np.array([[
        similarity,
        length_ratio,
        student_words,
        student_sentences,
    ]])

def train_model():
    """
    Treina o modelo de predição de grades
    """
    
    if not os.path.exists(GRADES_CSV):
        print(f"❌ Erro: Arquivo CSV não encontrado em: {GRADES_CSV}")
        print(f"   Crie um arquivo CSV com as colunas: Question, Base_answer, Student_answer, Grade")
        return False
    
    print("📚 Carregando dados...")
    try:
        df = pd.read_csv(GRADES_CSV)
    except Exception as e:
        print(f"❌ Erro ao carregar CSV: {e}")
        return False
    
    # Validar colunas
    required_cols = ['Question', 'Base_answer', 'Student_answer', 'Grade']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Erro: CSV deve ter as colunas: {required_cols}")
        print(f"   Colunas encontradas: {df.columns.tolist()}")
        return False
    
    print(f"✅ Dados carregados: {len(df)} amostras")
    
    # Limpar dados
    df = df.dropna(subset=required_cols)
    print(f"✅ Após limpeza: {len(df)} amostras")
    
    if len(df) < 10:
        print(f"❌ Erro: Você precisa de pelo menos 10 amostras para treinar. Tem {len(df)}")
        return False
    
    # Carregar modelo semântico
    print("🤖 Carregando modelo de embeddings...")
    transformer = load_semantic_transformer()
    
    # Calcular similaridades e features
    print("🔍 Calculando features...")
    features_list = []
    grades = []
    
    for idx, row in df.iterrows():
        try:
            student_answer = str(row['Student_answer'])
            base_answer = str(row['Base_answer'])
            grade = float(row['Grade'])
            
            # Calcular similaridade
            embeddings = transformer.encode([student_answer, base_answer])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Extrair features
            features = extract_features_from_response(student_answer, base_answer, similarity)
            features_list.append(features[0])
            grades.append(grade)
            
            if (idx + 1) % 100 == 0:
                print(f"   Processadas {idx + 1}/{len(df)} amostras...")
                
        except Exception as e:
            print(f"⚠️  Erro ao processar linha {idx}: {e}")
            continue
    
    if len(features_list) < 10:
        print("❌ Erro: Não há dados suficientes após processamento")
        return False
    
    X = np.array(features_list)
    y = np.array(grades)
    
    print(f"\n📊 Estatísticas das grades:")
    print(f"   Min: {y.min():.2f}, Max: {y.max():.2f}, Média: {y.mean():.2f}, Std: {y.std():.2f}")
    
    # Normalizar features
    print("\n⚙️  Normalizando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir dados
    print("📋 Dividindo dados (80% treino, 20% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Treinar modelos
    print("\n🏋️  Treinando modelos...")
    
    # Modelo 1: Ridge Regression (mais rápido, mais interpretável)
    print("   - Ridge Regression...", end="", flush=True)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    print(f" ✅ R²: {ridge_r2:.4f}, MAE: {ridge_mae:.4f}")
    
    # Modelo 2: Random Forest (mais potente, mas mais lento)
    print("   - Random Forest...", end="", flush=True)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print(f" ✅ R²: {rf_r2:.4f}, MAE: {rf_mae:.4f}")
    
    # Selecionar melhor modelo
    best_model = rf if rf_r2 > ridge_r2 else ridge
    best_name = "Random Forest" if rf_r2 > ridge_r2 else "Ridge Regression"
    best_r2 = max(rf_r2, ridge_r2)
    best_mae = rf_mae if rf_r2 > ridge_r2 else ridge_mae
    
    print(f"\n🏆 Melhor modelo: {best_name}")
    print(f"   R² Score: {best_r2:.4f} (quanto maior, melhor)")
    print(f"   MAE: {best_mae:.4f} (erro médio em pontos)")
    
    # Salvar modelo
    print(f"\n💾 Salvando modelo...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"   ✅ Modelo: {MODEL_PATH}")
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler: {SCALER_PATH}")
    
    print("\n✨ Treinamento concluído com sucesso!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Treinador de Modelo de Avaliação de Respostas")
    print("=" * 60)
    
    # Permitir passar o caminho do CSV como argumento
    if len(sys.argv) > 1:
        GRADES_CSV = sys.argv[1]
        print(f"📁 Usando CSV: {GRADES_CSV}\n")
    
    success = train_model()
    sys.exit(0 if success else 1)
