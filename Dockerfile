FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only files needed to install and run the package.
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY main.py ./
COPY ui/ ./ui/

RUN python -m pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
        "torch==2.2.2" "torchvision==0.17.2" \
    && pip install .

# Run as non-root user.
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "ultrasound.api"]


FROM base AS test

USER root
RUN pip install ".[dev]"
COPY tests/ ./tests/
USER appuser

CMD ["pytest", "tests/", "-v", "--tb=short"]
