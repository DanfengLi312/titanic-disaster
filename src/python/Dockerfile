FROM python:3.12-slim

WORKDIR /app

# Install pinned dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

# The data folder is mounted at runtime (see README), never baked into the image
ENV DATA_DIR=/data

CMD ["python", "titanic_model.py"]
