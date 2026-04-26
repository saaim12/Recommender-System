from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except Exception:  # pragma: no cover - Airflow is optional in local dev/test environments.
    DAG = None
    PythonOperator = None

from src.pipelines.train_pipeline import run_pipeline
from src.utils.config import load_config


config = load_config()
schedule = config.get("airflow", {}).get("schedule", "0 2 * * *")
start_date = config.get("airflow", {}).get("start_date", "2026-01-01")
if isinstance(start_date, str):
    start_date = datetime.fromisoformat(start_date)


if DAG is not None:
    with DAG(
        dag_id=config.get("airflow", {}).get("dag_id", "train_recommender"),
        start_date=start_date,
        schedule=schedule,
        catchup=False,
        tags=["recommender", "training"],
    ) as dag:
        train_task = PythonOperator(
            task_id="train_recommender_model",
            python_callable=run_pipeline,
        )

else:
    dag = None
