FROM python:3.8

# Установка зависимостей
RUN apt-get update -qy && \
    apt-get install -qy python3-pip python3.8-dev

# Копирование приложения
COPY app /app

# Установка зависимостей Python
WORKDIR /app
RUN pip install -r requirements.txt

# Запуск приложения
CMD ["python", "app.py"]

