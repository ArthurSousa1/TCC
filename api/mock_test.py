import re
import os
import pickle
from flask import json

_answers = {"answers": []}

#Load answers from JSON
answers_path = os.path.join(os.path.dirname(__file__), 'Service', 'data', 'answers.json')
try:
    with open(answers_path, 'r', encoding='utf-8') as f:
        _answers = json.load(f)
    print(f"Respostas carregadas com sucesso! Total: {len(_answers.get('answers', []))}")
except FileNotFoundError:
    _answers = {"answers": []}
    print(f"Arquivo de respostas não encontrado em {answers_path}")