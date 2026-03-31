FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY latent_insights/ latent_insights/
RUN uv sync --frozen --no-dev

EXPOSE ${PORT:-8000}

CMD uv run uvicorn latent_insights.main:app --host 0.0.0.0 --port ${PORT:-8000}
