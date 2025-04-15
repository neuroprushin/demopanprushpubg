# Используем официальный образ Python 3.11 (как в логе Render)
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Копируем файл с зависимостями
# УБЕДИСЬ, ЧТО ОН СОДЕРЖИТ ТОЛЬКО 5 НУЖНЫХ СТРОК!
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное приложение
# УБЕДИСЬ, что файл face_landmarker.task лежит рядом
COPY app.py .
COPY templates ./templates
COPY face_landmarker.task . # Копируем базовую модель

# Открываем порт, который будет слушать Gunicorn
EXPOSE ${PORT}

# Команда для запуска приложения через Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
