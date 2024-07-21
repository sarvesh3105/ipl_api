From python:3.9-slim

WORKDIR /app

COPY requirements.txt

RUN pip install --no--cache--dir -r requirements.txt

COPY . .3

EXPOSE 8000

CMD ["uvicorn","app.mlapi:app","--host","0.0.0.0","--port","8000"]