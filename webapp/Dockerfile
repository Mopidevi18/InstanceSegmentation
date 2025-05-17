# Dockerfile
FROM python:3.9-slim

# install system deps for OpenCV
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
 && rm -rf /var/lib/apt/lists/*

# now install your Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy your app
COPY . /app
WORKDIR /app

# ensure /app/model/ exists for downloaded weights
RUN mkdir -p /app/model

# run via gunicorn
CMD exec gunicorn --bind :$PORT app:app
