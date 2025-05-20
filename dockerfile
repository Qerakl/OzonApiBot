# Базовый образ с Python 3.13
FROM python:3.13.0-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей Python
WORKDIR /app

COPY ./bot /app

RUN pip install --upgrade pip

# Установим зависимости (если есть requirements.txt)
RUN pip install pandas scikit-learn numpy fastapi uvicorn pydantic

# Открываем порт (если нужно)
EXPOSE 8000

# Команда запуска FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
