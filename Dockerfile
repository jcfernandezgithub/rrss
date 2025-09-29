FROM python:3.11-slim

# Dependencias del sistema (fuentes para fpdf y locales opcionales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core locales \
 && rm -rf /var/lib/apt/lists/*

# (Opcional) intentar configurar locale es_ES
RUN sed -i 's/# es_ES.UTF-8 UTF-8/es_ES.UTF-8 UTF-8/' /etc/locale.gen && locale-gen || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Variables por defecto
ENV PORT=8000
EXPOSE 8000

# Iniciar FastAPI
CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]
