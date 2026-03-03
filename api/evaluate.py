from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o modelo uma única vez para reutilizá-lo
_model = None

def get_model():
    """Retorna o modelo de IA, carregando-o se necessário"""
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model


def calculate_score(similarity, max_score=10):
    """Calcula a nota baseada na similaridade"""
    if similarity < 0:
        similarity = 0.0
    return min(similarity * max_score, max_score)


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
    
    Args:
        student_answer (str): A resposta fornecida pelo aluno
        reference_answer (str): A resposta de referência para comparação
    
    Returns:
        dict: Dicionário contendo:
            - similarity: Valor de similaridade semântica (0-1)
            - score: Nota atribuída (0 a max_score)
            - label: Feedback em texto sobre a nota
    """
    # Validação básica
    if not student_answer or not reference_answer:
        raise ValueError("Resposta do aluno e resposta de referência são obrigatórias")
    
    max_score = 10  # Nota máxima fixa para simplificar
    if max_score <= 0:
        raise ValueError("Nota máxima deve ser maior que zero")
    
    # Carrega o modelo
    model = get_model()
    
    # Gera embeddings para ambas as respostas
    embeddings = model.encode([student_answer, reference_answer])
    
    # Calcula a similaridade semântica
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    
    # Calcula a nota
    score = calculate_score(float(similarity), max_score)
    
    # Gera a label de feedback
    label = get_grade_label(round(score, 2))
    
    return {
        "similarity": float(similarity),
        "score": round(score, 2),
        "label": label
    }