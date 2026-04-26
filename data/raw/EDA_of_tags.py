from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
	df = pd.read_csv(BASE_DIR / "tags.csv")
	print(df.head())
	print(df.info())
	print(df.shape)
	print(df.isnull().sum())
	print(df.duplicated().sum())
	print(df.describe(include="all"))
	print(df["tag"].astype(str).str.lower().value_counts().head(20))


if __name__ == "__main__":
	main()
