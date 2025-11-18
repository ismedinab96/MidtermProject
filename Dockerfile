FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY model/ model/

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app.app:app", "--workers", "1", "--threads", "2"]
