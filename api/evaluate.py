from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np

# Carrega o modelo uma única vez para reutilizá-lo
_model = None
_grade_predictor = None
_grade_scaler = None

def get_model():
    """Retorna o modelo de IA, carregando-o se necessário"""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

def load_grade_prediction_model():
    """Carrega o modelo pré-treinado de predição de grades"""
    global _grade_predictor, _grade_scaler
    
    if _grade_predictor is None:
        model_path = os.path.join(os.path.dirname(__file__), 'semantic_grade_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'semantic_grade_scaler.pkl')
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    _grade_predictor = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    _grade_scaler = pickle.load(f)
                print("✅ Modelo de predição de grades carregado!")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo de grades: {e}")
    
    return _grade_predictor, _grade_scaler


def calculate_score(similarity, max_score=10):
    """Calcula a nota baseada na similaridade"""
    if similarity < 0:
        similarity = 0.0
    return min(similarity * max_score, max_score)

def extract_features(student_answer, base_answer, similarity):
    """
    Extrai features da resposta para o modelo de predição
    """
    student_words = len(student_answer.split()) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0
    student_sentences = max(1, len([s for s in student_answer.split('.') if s.strip()]))
    
    length_ratio = student_words / base_words if base_words > 0 else 0
    
    return np.array([[
        similarity,
        length_ratio,
        student_words,
        student_sentences,
    ]])

def predict_grade(student_answer, base_answer, similarity):
    """
    Prediz a grade usando o modelo treinado
    Se não houver modelo, usa fallback baseado em similaridade
    """
    grade_predictor, grade_scaler = load_grade_prediction_model()
    
    if grade_predictor is None or grade_scaler is None:
        # Fallback: converter similaridade (0-1) em grade (0-10)
        return calculate_score(similarity, max_score=10)
    
    try:
        features = extract_features(student_answer, base_answer, similarity)
        features_scaled = grade_scaler.transform(features)
        predicted_grade = grade_predictor.predict(features_scaled)[0]
        
        # Garantir que está entre 0 e 10
        return np.clip(float(predicted_grade), 0.0, 10.0)
    except Exception as e:
        print(f"⚠️ Erro ao prever grade: {e}")
        # Fallback em caso de erro
        return calculate_score(similarity, max_score=10)


def get_grade_label(score):
    """Retorna a label/feedback baseado na nota"""
    if score == 10:
        return "Perfeito, você é incrível!"
    if score >= 9:
        return "Top dmais mermão!"
    if score >= 7:
        return "É foi bom, mas da pra melhorar!"
    if score >= 5:
        return "Perigoso isso ai hein"
    if score >= 3:
        return "Rapaz... ai vai sobrar pra P3"
    return "Exame de formando é em Agosto..."


def evaluate_answer(student_answer, reference_answer):
    """
    Avalia a resposta do aluno comparando com a resposta de referência.
    
    Processa:
    1. Calcula similaridade semântica entre as respostas
    2. Extrai features estruturais (comprimento, etc)
    3. Usa modelo treinado para prever a grade (0-10)
    4. Gera feedback baseado na grade
    
    Args:
        student_answer (str): A resposta fornecida pelo aluno
        reference_answer (str): A resposta de referência para comparação
    
    Returns:
        dict: Dicionário contendo:
            - similarity: Valor de similaridade semântica (0-1)
            - score: Nota predita pelo modelo (0-10)
            - label: Feedback em texto sobre a nota
    """
    # Validação básica
    if not student_answer or not reference_answer:
        raise ValueError("Resposta do aluno e resposta de referência são obrigatórias")
    
    # Carrega o modelo semântico
    model = get_model()
    
    # Gera embeddings para ambas as respostas
    embeddings = model.encode([student_answer, reference_answer])
    
    # Calcula a similaridade semântica
    similarity = float(cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0])
    
    # Usa modelo treinado para prever a grade
    score = predict_grade(student_answer, reference_answer, similarity)
    
    # Gera feedback
    label = get_grade_label(round(score, 2))
    
    return {
        "similarity": similarity,
        "score": round(score, 2),
        "label": label
    }