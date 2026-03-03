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


def evaluate_answer(student_answer, reference_answer, max_score=10):
    """
    Avalia a resposta do aluno comparando com a resposta de referência.
    
    Args:
        student_answer (str): A resposta fornecida pelo aluno
        reference_answer (str): A resposta de referência para comparação
        max_score (float): A nota máxima para a questão (padrão: 10)
    
    Returns:
        dict: Dicionário contendo:
            - similarity: Valor de similaridade semântica (0-1)
            - score: Nota atribuída (0 a max_score)
            - max_score: Nota máxima da questão
    """
    # Validação básica
    if not student_answer or not reference_answer:
        raise ValueError("Resposta do aluno e resposta de referência são obrigatórias")
    
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
    
    return {
        "similarity": float(similarity),
        "score": round(score, 2),
        "max_score": max_score
    }


def read_multiline(label):
    """Função auxiliar para leitura de múltiplas linhas (modo terminal)"""
    print(f"\nDigite {label} (finalize com linha vazia):")
    lines = []

    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return " ".join(lines)


def main():
    """Função principal para uso interativo via terminal"""
    print("=== Motor de correção de respostas dissertativas ===")

    student_answer = read_multiline("a resposta do aluno")
    reference_answer = read_multiline("a resposta de referência")
    max_score = float(input("\nNota máxima da questão: "))

    print("\nCarregando modelo...")
    
    # Usa a função reutilizável
    result = evaluate_answer(student_answer, reference_answer, max_score)

    print("\n========== Resultado ==========")
    print(f"Similaridade semântica: {result['similarity']:.4f}")
    print(f"Nota atribuída: {result['score']:.2f} / {result['max_score']}")
    print("================================")


if __name__ == "__main__":
    main()
