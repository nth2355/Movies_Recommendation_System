import pandas as pd
from ast import literal_eval as le
#parse
def clean_movies_ids(movies: pd.DataFrame) -> pd.DataFrame:
    original_rows = len(movies)

    movies = movies[
        pd.to_numeric(
            movies["id"],
            errors="coerce"
        ).notnull()
    ].copy()
    movies["id"] = movies["id"].astype(int)
    removed_rows = original_rows - len(movies)
    print(f"Đã xóa {removed_rows} dòng không hợp lệ.")
    return movies

def extract_name(items):
    return [item['name'] for item in items]


def parse_genres(movies: pd.DataFrame)->pd.DataFrame:
    movies = movies.copy()
    movies['genres'] = (
        movies['genres'].fillna("[]").apply(le)
    )
    movies['genres'] = movies['genres'].apply(extract_name)
    
    return movies

def parse_keywords(keywords:pd.DataFrame)->pd.DataFrame:
    keywords = keywords.copy()
    keywords['keywords'] = (keywords['keywords'].fillna("[]").apply(le))
    keywords['keywords'] = keywords['keywords'].apply(extract_name)
    return keywords

def parse_cast(credits:pd.DataFrame)->pd.DataFrame:
    credits = credits.copy()
    credits["cast"] = credits["cast"].apply(le)
    credits["cast"] = credits["cast"].apply(
        lambda x: [actor["name"] for actor in x[:3]]
    )
    return credits

def get_director(crew):
    for member in crew:
        if member["job"] == "Director":
            return member["name"]
    return ""

def parse_director(credits):
    credits = credits.copy()
    credits["crew"] = credits["crew"].apply(le)
    credits["director"] = credits["crew"].apply(get_director)
    return credits

#merge
def merge_movie_data(movies, keywords, credits):
    merged = movies.merge(
        keywords,
        on="id"
    )
    merged = merged.merge(
        credits[["id", "cast", "director"]],
        on="id"
    )
    return merged

def create_year_feature(movies_df):
    movies_df = movies_df.copy()

    movies_df["year"] = pd.to_datetime(
        movies_df["release_date"],
        errors="coerce"
    ).dt.year

    return movies_df


def create_title_year(movies_df):
    movies_df = movies_df.copy()

    movies_df["title_year"] = (
        movies_df["title"]
        + " ("
        + movies_df["year"].fillna(0)
        .astype(int)
        .astype(str)
        + ")"
    )

    return movies_df