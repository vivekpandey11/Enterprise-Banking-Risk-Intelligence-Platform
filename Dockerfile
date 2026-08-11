FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/staging ./data/staging
COPY dashboards ./dashboards

EXPOSE 8000

CMD ["uvicorn", "src.api.risk_api:app", "--host", "0.0.0.0", "--port", "8000"]
