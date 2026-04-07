# Recommender System – EDA & Feature Engineering Checklist

This checklist outlines the steps for performing **Exploratory Data Analysis (EDA)** and **Feature Engineering** for your recommender system project which i will do and have done step by step through code

---

## Step 1: Load Data
- [ ] Load raw CSV or `.dat` files into Pandas
- [ ] Inspect first few rows: `head()`
- [ ] Check column names and data types: `info()`
- [ ] Check dataset shape: `shape`

---

## Step 2: Basic Exploration
- [ ] Check for missing values: `isnull().sum()`
- [ ] Check for duplicates: `duplicated().sum()`
- [ ] Summary statistics: `describe()`
- [ ] Count unique users and items: `nunique()`

---

## Step 3: Analyze Distributions
- [ ] Plot histogram of ratings
- [ ] Plot number of ratings per user
- [ ] Plot number of ratings per item
- [ ] Identify active users and popular items

---

## Step 4: Merge Datasets
- [ ] Merge ratings with movies metadata
- [ ] Check merged dataset for correctness
- [ ] Analyze trends by genres, tags, or other metadata

---

## Step 5: Handle Missing or Anomalous Data
- [ ] Fill or drop missing values
- [ ] Remove duplicates if any
- [ ] Optionally filter users/items with very few interactions
- [ ] Handle outliers if ratings are outside expected range

---

## Step 6: Feature Engineering – User Features
- [ ] Total ratings per user
- [ ] Average rating per user
- [ ] Activity score (interaction frequency)
- [ ] Optional: time-based activity trends

---

## Step 7: Feature Engineering – Item Features
- [ ] Average rating per item
- [ ] Popularity (number of ratings per item)
- [ ] Genre encoding (one-hot or multi-hot)
- [ ] Optional: tags / categories encoding

---

## Step 8: Feature Engineering – Interaction Features
- [ ] Implicit feedback signals (clicks, add-to-cart, purchase)
- [ ] Time-based features (timestamps, recency)
- [ ] Optional: rating deviation from user or item mean

---

## Step 9: Prepare User-Item Matrix
- [ ] Pivot ratings into matrix format (`userId` × `movieId`)
- [ ] Fill missing values (zeros for implicit feedback or NaN for explicit)

---

## Step 10: Normalize / Scale Features (Optional)
- [ ] Normalize ratings (mean normalization)
- [ ] Scale numeric features to 0–1 if needed
- [ ] Encode categorical features

---

## Step 11: Save Processed Data
- [ ] Save processed CSVs for modeling
- [ ] Organize into folders: `raw/` and `processed/`
- [ ] Ensure reproducibility for the pipeline

---

✅ Follow this checklist sequentially to prepare clean and feature-rich data for your recommender system.