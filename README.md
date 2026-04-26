# Recommender System

This project now contains a complete, runnable baseline recommender pipeline for the MovieLens data in `data/raw/`.

## What is included

- Raw data loading and validation
- Movie catalog feature engineering
- A content-based recommender model
- Holdout evaluation metrics
- FastAPI serving endpoints
- An Airflow DAG for scheduled retraining
- Smoke tests for the core recommendation flow

## Train

```bash
python scripts/train.py
```

This writes the trained model to `data/models/recommender.pkl` and metrics to `data/models/metrics.json`.

## Serve

```bash
python scripts/serve.py
```

The API exposes:

- `GET /health`
- `POST /recommend`
- `POST /similar`

## Project layout

- `src/data_ingestion/` loads the raw CSV files and prepares the catalog
- `src/features/` builds item and user features
- `src/training/` fits and saves the recommender
- `src/evaluation/` computes ranking metrics
- `src/serving/` exposes the API
- `airflow/dags/` contains the retraining DAG
