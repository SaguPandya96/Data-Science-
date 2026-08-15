FROM python:3.14.6-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTHENTITEXT_ROOT=/app

WORKDIR /app

RUN groupadd --system authentitext \
    && useradd --system --gid authentitext --home-dir /app --shell /usr/sbin/nologin authentitext

COPY requirements/runtime.lock requirements/runtime.lock
RUN python -m pip install --no-cache-dir --upgrade pip==26.1.2 \
    && python -m pip install --no-cache-dir -r requirements/runtime.lock

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/metadata ./data/metadata
RUN python -m pip install --no-cache-dir --no-deps . \
    && mkdir -p /app/artifacts/baselines/id \
    && chown -R authentitext:authentitext /app

USER authentitext

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=2).read()"]

CMD ["python", "-m", "uvicorn", "authentitext.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
