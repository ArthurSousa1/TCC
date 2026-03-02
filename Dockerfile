# Use Node.js 18 Alpine (leve e rápido)
FROM node:18-alpine

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia package.json e package-lock.json
COPY package*.json ./

# Instala as dependências
RUN npm install

# Copia todo o código da aplicação
COPY . .

# Expõe a porta (ajuste conforme necessário)
EXPOSE 3000

# Comando para iniciar a aplicação
CMD ["node", "Service/app.js"]
