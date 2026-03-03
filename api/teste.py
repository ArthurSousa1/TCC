import json
import random
import os
import sys

questions_path = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'questions.json')
try:
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"Questões carregadas com sucesso! Total: {len(questions.get('questions', []))}")
except FileNotFoundError:
    questions = {"questions": []}
    print(f"Arquivo de questões não encontrado em {questions_path}")

print("Modelo de IA será carregado na primeira requisição de avaliação...")
