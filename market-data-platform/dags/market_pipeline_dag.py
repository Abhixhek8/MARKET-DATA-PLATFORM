"""Airflow DAG (2.x). Mount /dags into your Airflow instance and install this package."""
from datetime import datetime, timedelta

from airflow.decorators import dag, task


@dag(
    dag_id="market_data_pipeline",
    schedule="0 23 * * 1-5",  # weekdays after market close
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["market", "etl"],
)
def market_pipeline():
    @task
    def etl() -> int:
        from mdp.pipeline import run
        return run()

    @task
    def train(_rows: int) -> None:
        from mdp.pipeline import train as train_models
        train_models()

    train(etl())


market_pipeline()
