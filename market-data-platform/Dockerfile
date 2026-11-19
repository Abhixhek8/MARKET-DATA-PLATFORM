FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[ingest,api,dashboard,ml]"
COPY dashboard ./dashboard
COPY dags ./dags
CMD ["uvicorn", "mdp.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
