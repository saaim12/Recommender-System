import pandas as pd
# loading file
df=pd.read_csv('./links.csv')


# seeing first few rows for the inspecting
print(df.head())


# checking data types
print(df.dtypes)
# or 
print(df.info())

# no of rows and columns
print(df.shape)



## Basic exploration of the data
# checking for missing values
print(df.isnull().sum())
# movieId      0
# imdbId       0
# tmdbId     124

# checking for duplicates
print(df['tmdbId'].duplicated().sum())

## statistics of the data
print(df.describe())

print(df.nunique())

print(df.shape)