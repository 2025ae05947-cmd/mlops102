FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY README.md /app/
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "heart_disease.app:app", "--host", "0.0.0.0", "--port", "8000"]
