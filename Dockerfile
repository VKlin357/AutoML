# Reproducible environment for LLM-NAS.
# Build:  docker build -t llm-nas .
# Run:    docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY llm-nas make demo
FROM python:3.10-slim

WORKDIR /app

# System deps for LightGBM / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: regenerate the results tables from the committed raw logs (no GPU/API needed).
CMD ["python", "scripts/build_results_table.py"]
