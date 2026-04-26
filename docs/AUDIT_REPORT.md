# Recommender System Audit Report

## 1. Project Overview

This repository is a movie recommender system built around the MovieLens dataset. The project now has a production-oriented FastAPI serving layer, modular ML pipeline, evaluation metrics, logging, caching, Docker support, CI, and deployment docs.

The core recommendation strategy is hybrid:

- content-based similarity from genre and item feature vectors
- lightweight collaborative signal from item co-occurrence in positive ratings
- popularity blending for cold-start stability

## 2. File Structure

Top-level responsibilities:

- `src/data_ingestion/loader.py`: load CSVs, merge metadata, build catalog, split train/test
- `src/features/engineer.py`: genre parsing, item features, user profiles, score normalization
- `src/training/model.py`: recommendation model, save/load, ranking logic
- `src/evaluation/metrics.py`: Precision@K, Recall@K, Hit Rate@K, coverage
- `src/serving/api.py`: FastAPI endpoints
- `src/serving/service.py`: service wrapper, caching, model loading
- `src/serving/middleware.py`: request tracing and exception handling
- `src/core/settings.py`: environment-based configuration
- `src/core/logging.py`: logging setup
- `scripts/train.py` and `scripts/serve.py`: local entry points
- `airflow/dags/train_recommender.py`: retraining DAG
- `docs/*.md`: architecture, API, deployment, and this audit report

## 3. ML Explanation

### Data loading

The loader reads `movies.csv`, `ratings.csv`, `tags.csv`, and `links.csv`, then creates a catalog joined by `movieId`.

### Feature engineering

Item features combine:

- one-hot encoded genres
- rating count
- average rating
- rating standard deviation
- distinct user count
- tag count
- distinct tag count

### Recommendation logic

For a user:

1. Collect the user rating history.
2. Build a weighted user profile from rated movie vectors.
3. Compute cosine-like similarity against all items.
4. Add collaborative score from co-liked items.
5. Blend with popularity.
6. Remove already-watched items.
7. Rank by score.

### Metrics

The system evaluates holdout recommendations with:

- Precision@K
- Recall@K
- Hit Rate@K
- Coverage

## 4. API / Serving Layer

The repository now includes a FastAPI app with:

- `GET /health`
- `POST /recommend`
- `POST /similar`
- `POST /reload-model`

It also includes:

- request ID middleware
- unhandled exception middleware
- API key guard for protected endpoints
- in-memory or Redis cache support

## 5. How to Run

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run API:

```bash
python scripts/serve.py
```

## 6. How to Test

Run tests:

```bash
pytest -q
```

Compile check:

```bash
python -m compileall src scripts tests airflow docs
```

## 7. System Diagrams

- [Architecture](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Deployment Guide](DEPLOYMENT.md)

## 8. Issues & Fixes

Fixed or addressed:

- missing production API layer
- missing centralized logging
- missing input validation and middleware
- missing cache abstraction
- hardcoded runtime assumptions in scripts
- missing env template
- missing Docker/compose/CI files
- empty module structure without production orchestration

Residual risk:

- the model is still a baseline hybrid recommender, not a large-scale matrix factorization system
- training and runtime quality depend on the completeness of the MovieLens dataset in `data/raw/`

## 9. Improvements

Already added:

- modular service layer
- structured logging
- environment-driven settings
- Redis-ready cache path
- Docker deployment path
- GitHub Actions CI
- API documentation
- architecture documentation

Potential next step:

- stronger offline ranking model or matrix factorization
- request authentication and rate limiting
- experiment tracking with MLflow integration
