import pandas as pd

df=pd.read_csv('./movies.csv')

print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)

#print(df['genres'].str.split('|').explode().value_counts())
# dublicatesdf=df[df['title'].duplicated(keep=False)]
# print(dublicatesdf.head(20))
print(df.duplicated(subset='title', keep=False).sum())