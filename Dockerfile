FROM python:3.11-slim

WORKDIR /app

# Copy ONLY the API package, but preserve src/api structure
COPY src/api src/api

# Install requirements inside api folder
COPY src/api/requirements.txt .
RUN pip install -r requirements.txt

# Copy model files
COPY models/trained models/trained

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
