.PHONY: install demo api dashboard test lint docker
install:
	pip install -e ".[ingest,api,dashboard,ml,dev]"
demo:            ## run the pipeline on offline synthetic data
	python -m mdp.pipeline run --source synthetic
run:             ## run the pipeline on live data (falls back to synthetic)
	python -m mdp.pipeline run
api:
	uvicorn mdp.api.main:app --reload --port 8000
dashboard:
	streamlit run dashboard/app.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker compose up --build
