# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Копируем файл с зависимостями
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное приложение
# УБЕДИСЬ, что файл .task (например, face_landmarker_v2_with_blendshapes.task)
# лежит рядом с Dockerfile и app.py перед сборкой образа
COPY app.py .
COPY templates ./templates
COPY face_landmarker_v2_with_blendshapes.task . # <-- КОПИРУЕМ МОДЕЛЬ

# Открываем порт, который будет слушать Gunicorn
EXPOSE ${PORT}

# Команда для запуска приложения через Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
