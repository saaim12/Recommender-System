from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
	df = pd.read_csv(BASE_DIR / "links.csv")
	print(df.head())
	print(df.dtypes)
	print(df.info())
	print(df.shape)
	print(df.isnull().sum())
	print(df["tmdbId"].duplicated().sum())
	print(df.describe())
	print(df.nunique())


if __name__ == "__main__":
	main()