FROM python:3.12-slim

# Install JRE for language-tool-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dashboard]"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download sentence-transformers model (cached at build time)
COPY scripts/download_models.py scripts/
RUN python scripts/download_models.py

# Copy source
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8510 8511
