from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
import os

app = Flask(__name__)

# Carrega o modelo de IA
print("Carregando modelo de IA...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Modelo carregado com sucesso!")

# Carrega as questões
questions_path = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'questions.json')
try:
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Questões carregadas com sucesso! Total: {len(questions.get('questions', []))}")
except FileNotFoundError:
    questions = {"questions": []}
    print(f"Arquivo de questões não encontrado em {questions_path}")


def calculate_score(similarity, max_score=10):
    """Calcula a nota baseada na similaridade"""
    if similarity < 0:
        similarity = 0.0
    return min(similarity * max_score, max_score)


@app.route('/api/v1/allQuestions', methods=['GET'])
def get_all_questions():
    """Retorna todas as questões"""
    return jsonify({
        "status": "success",
        "results": len(questions.get("questions", [])),
        "data": questions
    }), 200


@app.route('/api/v1/randomQuestion', methods=['GET'])
def get_random_question():
    """Retorna uma questão aleatória"""
    try:
        questions_array = questions.get("questions", [])
        if not questions_array:
            return jsonify({
                "status": "error",
                "message": "Nenhuma questão disponível"
            }), 404
        
        random_question = random.choice(questions_array)
        return jsonify({
            "status": "success",
            "data": random_question
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/v1/answer/<int:question_id>', methods=['POST'])
def evaluate_answer(question_id):
    """Avalia a resposta do aluno"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Corpo da requisição vazio"
            }), 400
        
        student_answer = data.get("student_answer", "").strip()
        reference_answer = data.get("reference_answer", "").strip()
        max_score = float(data.get("max_score", 10))
        
        if not student_answer or not reference_answer:
            return jsonify({
                "status": "error",
                "message": "student_answer e reference_answer são obrigatórios"
            }), 400
        
        # Gera embeddings e calcula similaridade
        embeddings = model.encode([student_answer, reference_answer])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Calcula a nota
        score = calculate_score(float(similarity), max_score)
        
        return jsonify({
            "status": "success",
            "data": {
                "question_id": question_id,
                "similarity": float(similarity),
                "score": round(score, 2),
                "max_score": max_score
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Erro ao avaliar resposta",
            "error": str(e)
        }), 500


@app.route('/api/v1/evaluate', methods=['POST'])
def evaluate_only():
    """Endpoint de avaliação simples (sem question_id)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "Corpo da requisição vazio"
            }), 400
        
        student_answer = data.get("student_answer", "").strip()
        reference_answer = data.get("reference_answer", "").strip()
        max_score = float(data.get("max_score", 10))
        
        if not student_answer or not reference_answer:
            return jsonify({
                "status": "error",
                "message": "student_answer e reference_answer são obrigatórios"
            }), 400
        
        # Gera embeddings e calcula similaridade
        embeddings = model.encode([student_answer, reference_answer])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Calcula a nota
        score = calculate_score(float(similarity), max_score)
        
        return jsonify({
            "status": "success",
            "data": {
                "similarity": float(similarity),
                "score": round(score, 2),
                "max_score": max_score
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Erro ao avaliar resposta",
            "error": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Verifica se o serviço está rodando"""
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
