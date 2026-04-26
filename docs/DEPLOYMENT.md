# Deployment Guide

## Option 1: Render

1. Push repository to GitHub.
2. In Render, create a new Web Service from the repo.
3. Render will detect `render.yaml`.
4. Configure environment variables from `.env.example`.
5. Deploy and verify `/health` endpoint.

## Option 2: AWS ECS (Fargate)

1. Build and push Docker image to Amazon ECR.
2. Create ECS task definition exposing port 8000.
3. Set environment variables in ECS task.
4. Deploy service behind an Application Load Balancer.
5. Add auto-scaling based on CPU and memory.

## Option 3: GCP Cloud Run

1. Build and push Docker image to Artifact Registry.
2. Deploy image as Cloud Run service.
3. Set required environment variables.
4. Configure min/max instances and concurrency.
5. Protect endpoint with Cloud Armor or API gateway if required.

## Post-deployment Checklist

- Confirm `/health` returns `model_loaded=true`.
- Run smoke requests against `/recommend` and `/similar`.
- Enable logs and alerts.
- Schedule retraining via Airflow or external scheduler.
