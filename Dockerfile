# Optional: run PlateProof in a container instead of a local Python install.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY samples ./samples

RUN useradd --create-home appuser && mkdir -p /app/data /app/models && chown -R appuser /app/data /app/models
USER appuser

EXPOSE 8000
CMD ["python", "-m", "app"]
