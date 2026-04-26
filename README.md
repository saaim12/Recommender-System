# Recommender System

Movie recommendation service built on MovieLens data with a modular training and serving stack.

## Problem Statement

The project recommends movies to a user based on historical interactions and item metadata. It is designed as a production-ready baseline that can be trained, evaluated, deployed, and extended without rewriting the core architecture.

## Solution Overview

The system loads `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`, builds item features from genres and interaction statistics, trains a hybrid recommendation model, evaluates it with ranking metrics, and serves recommendations through FastAPI.

## Tech Stack

- Python 3.12.3
- pandas, numpy, scikit-learn
- FastAPI, Pydantic, Uvicorn
- Apache Airflow
- Redis optional cache
- Docker and Docker Compose
- GitHub Actions CI

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for Mermaid diagrams and system boundaries.

## ML Model

The recommender is primarily content-based, using movie genres, popularity features, and tag/rating aggregates. It adds a lightweight collaborative signal by using item co-occurrence from positive interactions. Recommendations are ranked by a blended score:

- content similarity between user profile and item vectors
- collaborative similarity from co-liked items
- popularity bias to stabilize cold-start results

## Features

- Modular ingestion, feature engineering, evaluation, and serving layers
- API key support for protected endpoints
- Request tracing and structured logging
- In-memory or Redis caching
- Evaluation metrics: Precision@K, Recall@K, Hit Rate@K, coverage
- Docker, docker-compose, and Render-ready deployment config

## Folder Structure

```text
airflow/
	dags/
configs/
data/
	models/
	processed/
	raw/
docs/
scripts/
src/
	core/
	data_ingestion/
	evaluation/
	features/
	pipelines/
	serving/
	training/
tests/
```

## Setup

1. Create a Python 3.12.3 virtual environment.
2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Create environment file:

```bash
copy .env.example .env
```

4. Edit `.env` if you want API keys, Redis, or MLflow configured.

## How to Run Locally

Start the API:

```bash
python scripts/serve.py
```

Default URL: `http://localhost:8000`

## How to Use the API

Endpoints:

- `GET /health`
- `POST /recommend`
- `POST /similar`
- `POST /reload-model`

Example request:

```json
{
	"user_id": 1,
	"top_n": 10
}
```

Example response:

```json
{
	"user_id": 1,
	"recommendations": [
		{
			"movieId": 858,
			"title": "Godfather, The (1972)",
			"genres": "Crime|Drama",
			"score": 0.83
		}
	]
}
```

Detailed API examples are in [docs/API.md](docs/API.md).

## How to Test

Run automated tests:

```bash
pytest -q
```

Compile check:

```bash
python -m compileall src scripts tests airflow docs
```

Postman, curl, and Python request examples are in [docs/API.md](docs/API.md).

## Docker

Build image:

```bash
docker build -t recommender-system .
```

Run with Redis:

```bash
docker compose up --build
```

## Deployment Guide

- Render: use `render.yaml`
- AWS ECS/Fargate: use `Dockerfile`
- GCP Cloud Run: use `Dockerfile`

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Example Outputs

- `GET /health` returns model and cache status.
- `POST /recommend` returns ranked movie candidates with `score`.
- `POST /similar` returns nearest movies to a given `movie_id`.

## Future Improvements

- Add a stronger collaborative model or matrix factorization layer
- Add offline metrics logging and experiment tracking in MLflow
- Add Redis-backed rate limiting
- Add batch recommendation jobs for large-scale users

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
