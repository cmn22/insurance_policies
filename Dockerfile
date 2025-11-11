# Use a slim image with manylinux wheels available
FROM python:3.11-slim

# System deps (git for DVC, ca-certificates for HTTPS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only what the serving container needs
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code & pipeline metadata
# Include .dvc dir, dvc.yaml, dvc.lock so `dvc pull` knows what to fetch
COPY . ./

# Expose FastAPI port
EXPOSE 8000

# Entrypoint: pull artifacts via DVC, then serve the API
CMD ["/bin/bash", "-lc", "dvc pull -v && uvicorn src.serve:APP --host 0.0.0.0 --port 8000"]