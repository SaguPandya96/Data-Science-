# Optional container image for EvalForge.
# Docker is NOT required: `pip install -e ".[dev,dashboard]"` is the primary path.
#
#   docker build -t evalforge .
#   docker run --rm evalforge evalforge demo
#   docker run --rm -p 8501:8501 evalforge evalforge dashboard

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVALFORGE_PROVIDER=mock \
    EVALFORGE_LOG_FORMAT=json

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dashboard]"

EXPOSE 8501

CMD ["evalforge", "demo"]
