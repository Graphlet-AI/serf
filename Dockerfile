FROM ubuntu:24.04

LABEL maintainer="rjurney@graphlet.ai"
LABEL description="SERF: Agentic Semantic Entity Resolution Framework"

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    git \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set Java home for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up working directory
WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --extra dev --no-install-project

# Copy the rest of the project
COPY . .

# Install the project itself
RUN uv sync --extra dev

# Pre-download the embedding model so it's cached in the image
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# Create data directories
RUN mkdir -p data/benchmarks logs

# Default entrypoint is the serf CLI
ENTRYPOINT ["uv", "run", "serf"]
CMD ["--help"]
