# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем переменные окружения
# PYTHONUNBUFFERED: чтобы логи Python сразу выводились в Docker logs
ENV PYTHONUNBUFFERED=1
# PORT: Порт, который будет слушать Gunicorn (Render может его переопределить)
ENV PORT=8080

# Копируем файл с зависимостями
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное приложение (app.py и папку templates)
COPY . .

# Открываем порт, который будет слушать Gunicorn
EXPOSE ${PORT}

# Команда для запуска приложения через Gunicorn
# --bind 0.0.0.0:${PORT}: Слушать на всех интерфейсах на указанном порту
# app:app: Искать объект 'app' в файле 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]
