# Use Python 3.11 slim (leve e rápido)
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia requirements.txt
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação
COPY . .

# Expõe a porta
EXPOSE 3000

# Comando para iniciar a aplicação
CMD ["python", "app.py"]
