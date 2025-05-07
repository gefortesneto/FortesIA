# Dockerfile para FortesIA (backend FastAPI com Uvicorn)
FROM python:3.10-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "fortesia_router_main:app", "--host", "0.0.0.0", "--port", "8000"]