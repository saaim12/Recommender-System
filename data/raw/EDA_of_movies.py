from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
	df = pd.read_csv(BASE_DIR / "movies.csv")
	print(df.head())
	print(df.info())
	print(df.describe(include="all"))
	print(df.shape)
	print(df["genres"].str.split("|").explode().value_counts().head(10))
	duplicates = df[df["title"].duplicated(keep=False)]
	print(duplicates.head(20))
	print(df.duplicated(subset="title", keep=False).sum())


if __name__ == "__main__":
	main()