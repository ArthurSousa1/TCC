import json
import random
import os
import sys
from evaluate import evaluate_answer

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

questions_path = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'questions.json')
try:
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Questões carregadas com sucesso! Total: {len(questions.get('questions', []))}")
except FileNotFoundError:
    questions = {"questions": []}
    print(f"Arquivo de questões não encontrado em {questions_path}")

print("Modelo de IA será carregado na primeira requisição de avaliação...")


@app.route('/api/v1/questions', methods=['GET'])
def get_all_questions():
    """Retorna todas as questões"""
    return jsonify({
        "status": "success",
        "results": len(questions.get("questions", [])),
        "data": questions
    }), 200


@app.route('/api/v1/evaluate', methods=['POST'])
def evaluate():
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
        
        if not student_answer or not reference_answer:
            return jsonify({
                "status": "error",
                "message": "student_answer e reference_answer são obrigatórios"
            }), 400
        
        # Usa a função do Core/evaluate.py
        result = evaluate_answer(student_answer, reference_answer)
        
        return jsonify({
            "status": "success",
            "data": result
        }), 200
        
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Erro ao avaliar resposta",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)
