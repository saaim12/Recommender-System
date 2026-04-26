from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
	df = pd.read_csv(BASE_DIR / "ratings.csv")
	print(df.head())
	print(df.info())
	print(df.shape)
	print(df.isnull().sum())
	print(df.duplicated().sum())
	print(df.describe())
	print(df["userId"].nunique())
	print(df["movieId"].nunique())
	print(df["rating"].value_counts().sort_index())


if __name__ == "__main__":
	main()
