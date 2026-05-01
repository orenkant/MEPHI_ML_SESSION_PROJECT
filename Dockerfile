FROM python:3.11-slim

# рабочая директория внутри контейнера
WORKDIR /app

# копируем зависимости и устанавливаем
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# копируем весь проект
COPY . .

# открываем порт
EXPOSE 5000

# запускаем через gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.api:app"]
