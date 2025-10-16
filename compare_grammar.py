import pandas as pd
import language_tool_python
import time

def avaliar_registro_formal(texto: str, ferramenta: language_tool_python.LanguageTool) -> tuple[int, list[str]]:
    """
    Avalia um texto com base na quantidade de erros gramaticais e ortográficos,
    retornando uma tupla com a nota (0-5) e a lista de erros encontrados.

    A nota é baseada na densidade de erros (erros por 100 palavras).
    - 5: 0 erros
    - 4: até 1 erro por 100 palavras
    - 3: até 3 erros por 100 palavras
    - 2: até 5 erros por 100 palavras
    - 1: até 7 erros por 100 palavras
    - 0: mais de 7 erros por 100 palavras
    """
    # Retorna 0 para textos vazios, nulos ou que não são strings
    if not isinstance(texto, str) or not texto.strip():
        return 0, ["Texto vazio ou inválido."]

    # Realiza a verificação de erros no texto
    erros = ferramenta.check(texto)
    num_erros = len(erros)
    
    palavras = texto.split()
    num_palavras = len(palavras)

    # Se não houver palavras, não há erros
    if num_palavras == 0:
        return 5, []

    # Calcula a densidade de erros (quantos erros a cada 100 palavras)
    erros_por_100_palavras = (num_erros / num_palavras) * 100
    
    # Extrai as mensagens de erro para feedback
    mensagens_de_erro = [f"Regra: {erro.ruleId} | Mensagem: '{erro.message}' | Trecho: '{texto[erro.offset:erro.offset+erro.errorLength]}'" for erro in erros]

    # Mapeia a densidade de erros para a nota de 0 a 5
    nota = 0
    if erros_por_100_palavras == 0:
        nota = 5
    elif erros_por_100_palavras <= 1:
        nota = 4
    elif erros_por_100_palavras <= 3:
        nota = 3
    elif erros_por_100_palavras <= 5:
        nota = 2
    elif erros_por_100_palavras <= 7:
        nota = 1
    else:
        nota = 0
        
    return nota, mensagens_de_erro


def processar_dataset(caminho_arquivo_entrada: str, caminho_arquivo_saida: str):
    """
    Lê um arquivo CSV, avalia a coluna 'essay' e salva um novo CSV com os resultados.
    """
    print("Iniciando o processo de avaliação...")
    
    # --- Carregamento da Ferramenta de Verificação ---
    print("Carregando o modelo de linguagem (Português-Brasil)...")
    print("(Isso pode demorar alguns instantes na primeira execução)")
    try:
        tool = language_tool_python.LanguageTool('pt-BR')
    except Exception as e:
        print(f"\n[ERRO] Não foi possível carregar o LanguageTool. Verifique sua conexão com a internet.")
        print(f"Detalhes do erro: {e}")
        return

    # --- Leitura do Arquivo CSV ---
    try:
        df = pd.read_csv(caminho_arquivo_entrada)
        print(f"Arquivo '{caminho_arquivo_entrada}' lido com sucesso. {len(df)} linhas encontradas.")
    except FileNotFoundError:
        print(f"\n[ERRO] O arquivo '{caminho_arquivo_entrada}' não foi encontrado.")
        print("Verifique se o nome do arquivo está correto e se ele está na mesma pasta do script.")
        return

    # --- Avaliação das Respostas ---
    print("Avaliando as respostas dos alunos... Por favor, aguarde.")
    start_time = time.time()
    
    # Aplica a função de avaliação em cada linha da coluna 'essay'
    # O resultado será uma tupla (nota, lista_de_erros), que será dividida em duas novas colunas
    resultados = df['essay'].apply(lambda texto: avaliar_registro_formal(texto, tool))
    df[['nota_registro_formal', 'erros_encontrados']] = pd.DataFrame(resultados.tolist(), index=df.index)

    end_time = time.time()
    print(f"Avaliação concluída em {end_time - start_time:.2f} segundos.")

    # --- Salvando o Resultado ---
    try:
        df.to_csv(caminho_arquivo_saida, index=False, encoding='utf-8-sig')
        print(f"\nProcesso finalizado! Os resultados foram salvos em '{caminho_arquivo_saida}'.")
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um problema ao salvar o arquivo de saída.")
        print(f"Detalhes do erro: {e}")

    # --- Finalização da Ferramenta ---
    tool.close()


# --- PONTO DE PARTIDA DO PROGRAMA ---
if __name__ == "__main__":
    # IMPORTANTE: Altere o nome do arquivo aqui para o nome do seu dataset
    NOME_ARQUIVO_ENTRADA = "test_dataset.csv"
    
    # Nome do arquivo que será gerado com os resultados
    NOME_ARQUIVO_SAIDA = "resultados_avaliacao.csv"
    
    # Chama a função principal para iniciar o processo
    processar_dataset(NOME_ARQUIVO_ENTRADA, NOME_ARQUIVO_SAIDA)