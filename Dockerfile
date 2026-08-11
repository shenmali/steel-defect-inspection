FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STEEL_INSPECTION_BACKEND=auto \
    STEEL_INSPECTION_SAVE_ANNOTATIONS=false
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN apt-get update && apt-get install --yes --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt && python -m pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/artifacts/results && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["python", "-m", "uvicorn", "steel_inspection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
