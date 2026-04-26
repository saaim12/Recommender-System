# API Documentation

Base URL: `http://localhost:8000`

If `API_KEY` is configured, include header:

- `x-api-key: <your-key>`

## Health

Endpoint: `GET /health`

Response example:

```json
{
  "status": "ok",
  "model_loaded": true,
  "cache_enabled": true,
  "cache_backend": "memory"
}
```

## Recommend for User

Endpoint: `POST /recommend`

Request:

```json
{
  "user_id": 1,
  "top_n": 10
}
```

Response:

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

## Similar Movies

Endpoint: `POST /similar`

Request:

```json
{
  "movie_id": 1,
  "top_n": 10
}
```

Response:

```json
{
  "movie_id": 1,
  "recommendations": [
    {
      "movieId": 3114,
      "title": "Toy Story 2 (1999)",
      "genres": "Adventure|Animation|Children|Comedy|Fantasy",
      "score": 0.91
    }
  ]
}
```

## Reload Model

Endpoint: `POST /reload-model`

Use this endpoint after retraining when you want the API to load the latest model artifact without service restart.
