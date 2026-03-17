import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from helpers import _questions, get_model, load_grade_prediction_model, concatanate_feedback, normalize_text

### Done - All steps defined
def evaluate_answer(student_answer, question_id: int):
    #Begin validating parameters
    if not student_answer or not question_id:
        raise ValueError("Resposta do aluno e ID da questão são obrigatórias")
    
    #Get question data to compare
    reference_answer = []
    keywords = []
    for q in _questions.get("questions", []):
        if q["id"] == question_id:
            reference_answer = q["reference_answer"]
            keywords = q["keywords"]

    clean_reference_answer = normalize_text(reference_answer)
    clean_student_answer = normalize_text(student_answer)

    #If question not found or without reference answer/keywords, raise error
    if not reference_answer or not keywords:
        raise ValueError(f"Questão com ID {question_id} não encontrada ou sem resposta de referência/keywords")

    #Validate if student answer is identical to reference answer (ignoring case and punctuation)
    if try_same_text(clean_reference_answer, clean_student_answer):
        print("Resposta idêntica à referência, atribuindo nota máxima...")
        return {
            "score": 10.0,
            "feedback": "Perfeito meu amigo, copiou do gabarito, ta colando né?"
        }
    
    #If user input is not the same as reference answer, validate keywords presence and calculate semantic similarity
    keywords_score, missing_keywords = validate_keywords(keywords, clean_student_answer)
    if keywords_score == 0:
        return {
            "score": 0.0,
            "feedback": "Nenhuma palavra-chave encontrada, revise os conceitos e tente novamente."
        }
    
    semantic_similarity_score, semantic_similarity_feedback = validate_semantic_similarity(clean_reference_answer, clean_student_answer)
    final_score = keywords_score + semantic_similarity_score
    
    return {
        "score": round(final_score, 2),
        "feedback": concatanate_feedback(missing_keywords, semantic_similarity_feedback)
    }

### Done - Check if user input is the same as reference answer (ignoring case and punctuation)
def try_same_text(reference_answer, student_answer):
    palavras1 = re.findall(r'\w+', reference_answer.lower())
    palavras2 = re.findall(r'\w+', student_answer.lower())

    total = len(palavras1)
    iguais = 0

    for palavra in palavras1:
        if palavra in palavras2:
            iguais += 1

    similaridade = iguais / total
    return similaridade >= 0.90

### Done - checking if keyword is used, calculate the score for each and return all missing keywords + final keyword score 
def validate_keywords(keywords, student_answer):
    quantidade_keywords = len(keywords)
    
    if quantidade_keywords == 0:
        return 0, []

    peso_keyword = 10 / quantidade_keywords
    keyword_score = 0
    missing_keywords = []

    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        if keyword_lower in student_answer.lower():
            keyword_score += peso_keyword
        else:
            missing_keywords.append(keyword)

    return keyword_score, missing_keywords

### Under development
def validate_semantic_similarity(reference_answer, student_answer):
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
    # score = predict_grade(student_answer, question_id, similarity)
    return similarity, "feedback do que ficou faltando para melhorar a nota"

### Under development
def predict_grade(student_answer, base_answer, similarity):
    grade_predictor, grade_scaler = load_grade_prediction_model()
    
    if grade_predictor is None or grade_scaler is None:
        raise RuntimeError("Modelo de predição de grades não encontrado. Execute o treinamento e salve o modelo antes de avaliar.")
    
    try:
        features = extract_features(student_answer, base_answer, similarity)
        features_scaled = grade_scaler.transform(features)
        predicted_grade = grade_predictor.predict(features_scaled)[0]
        return np.clip(float(predicted_grade), 0.0, 10.0)
    except Exception as e:
        print(f"⚠️ Erro ao prever grade: {e}")
        raise

### Under development
def extract_features(student_answer, base_answer, similarity):

    student_words = len(student_answer.split()) if student_answer else 0
    base_words = len(base_answer.split()) if base_answer else 0
    # Contar sentenças com pontuação básica
    student_sentences = max(1, len([s for s in re.split(r"[\.\!?]+", student_answer) if s.strip()]))
    
    length_ratio = student_words / base_words if base_words > 0 else 0
    
    return np.array([[
        similarity,
        length_ratio,
        student_words,
        student_sentences,
    ]])


# ### Teste rápido
# (print ("=====================primeiro exemplo====================="))
# (print(evaluate_answer("Alguém que escuta, comunica bem, inspira confiança, toma decisões com responsabilidade e se importa com as pessoas da equipe.", 3)))

# ### Uma resposta média
# (print ("=====================segundo exemplo====================="))
# (print(evaluate_answer("Um bom líder ajuda seus liderados, tem uma boa comunicação, inspira os outros e toma decisões com responsabilidade.", 3)))

# ### Uma resposta ruim
# (print ("=====================terceiro exemplo====================="))
# (print(evaluate_answer("Um líder é alguém que manda na equipe e tem que ser respeitado.", 3)))



def testando_semantica(keywords, student_answer):
    model = get_model()
    similarity = []

    # Divide a resposta em frases
    sentences = [s.strip() for s in re.split(r'\.\s*', student_answer) if s.strip()]

    for k in keywords:
        keyword_lower = k.lower()
        similarities_per_keyword = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            # Verifica se a frase contém a keyword
            if keyword_lower in sentence_lower:
                embeddings = model.encode([sentence, k])

                sim = float(cosine_similarity(
                    [embeddings[0]],
                    [embeddings[1]]
                )[0][0])

                similarities_per_keyword.append(sim)

        # Calcula média das similaridades dessa keyword
        if similarities_per_keyword:
            #Se mais de uma frase possuir a keyword, calcula a média das similaridades para aquela keyword
            avg_similarity = sum(similarities_per_keyword) / len(similarities_per_keyword)
            similarity.append(avg_similarity)
        else:
            similarity.append(0)  # keyword não apareceu em nenhuma frase

    # Média final geral
    if similarity:
        final_similarity = sum(similarity) / len(similarity)
    else:
        final_similarity = 0

    return final_similarity, "feedback do que ficou faltando para melhorar a nota"


### Teste rápido
(print ("=====================primeiro exemplo====================="))
(print(testando_semantica(["bom líder", "escuta", "comunicação", "confiança", "responsabilidade"],"Alguém que escuta, comunica bem, inspira confiança, toma decisões com responsabilidade e se importa com as pessoas da equipe.")))

### Uma resposta média
(print ("=====================segundo exemplo====================="))
(print(testando_semantica(["bom líder", "escuta", "comunicação", "confiança", "responsabilidade"], "Um bom líder ajuda seus liderados, tem uma boa comunicação, inspira os outros e toma decisões com responsabilidade.")))

### Uma resposta ruim
(print ("=====================terceiro exemplo====================="))
(print(testando_semantica(["bom líder", "escuta", "comunicação", "confiança", "responsabilidade"], "Um líder é alguém que manda na equipe e tem que ser respeitado.")))