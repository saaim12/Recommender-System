## Datasets

- **MovieLens** (ml-latest-small recommended for prototyping)  
- Raw CSV files: `ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`  
- Sources form where i got it: [GroupLens Datasets](https://grouplens.org/datasets/movielens/)

### `links.csv` Explanation

| Column | Description |
|--------|-------------|
| **movieId** | MovieLens internal ID |
| **imdbId** | IMDb ID for the movie, useful for fetching external metadata |
| **tmdbId** | TMDb ID for the movie, useful for fetching images, genres, release date |

*Example:*
movieId,imdbId,tmdbId
1,0114709,862
2,0113497,8844
3,0113228,15602