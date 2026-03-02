from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_multiline(label):
    print(f"\nDigite {label} (finalize com linha vazia):")
    lines = []

    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    return " ".join(lines)


def calculate_score(similarity, max_score):
    if similarity < 0:
        similarity = 0.0
    return similarity * max_score


def main():
    print("=== Motor de correção de respostas dissertativas ===")

    student_answer = read_multiline("a resposta do aluno")
    reference_answer = read_multiline("a resposta de referência")

    max_score = float(input("\nNota máxima da questão: "))

    print("\nCarregando modelo...")
    model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(
        [student_answer, reference_answer]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    score = calculate_score(similarity, max_score)

    print("\n========== Resultado ==========")
    print(f"Similaridade semântica: {similarity:.4f}")
    print(f"Nota atribuída: {score:.2f} / {max_score}")
    print("================================")


if __name__ == "__main__":
    main()
