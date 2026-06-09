import pandas as pd
from ast import literal_eval as le

def clean_movies_ids(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies[movies["id"].notnull()]
    movies["id"] = movies["id"].astype(int)

    return movies

def extract_name(items):
    if not isinstance(items, list):
        return []
    return [
        item.get("name", "")
        for item in items
        if isinstance(item, dict)
    ]

def parse_genres(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()

    movies["genres"] = (
        movies["genres"]
        .fillna("[]")
        .apply(le)
        .apply(extract_name)
    )

    return movies

def parse_keywords(keywords: pd.DataFrame) -> pd.DataFrame:
    keywords = keywords.copy()

    keywords["keywords"] = (
        keywords["keywords"]
        .fillna("[]")
        .apply(le)
        .apply(extract_name)
    )

    return keywords

def parse_cast(credits: pd.DataFrame) -> pd.DataFrame:
    credits = credits.copy()

    credits["cast"] = credits["cast"].fillna("[]").apply(le)

    credits["cast"] = credits["cast"].apply(
        lambda x: [
            actor.get("name", "")
            for actor in x[:3]
            if isinstance(actor, dict)
        ]
    )

    return credits

def get_director(crew):
    if not isinstance(crew, list):
        return ""

    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def parse_director(credits: pd.DataFrame) -> pd.DataFrame:
    credits = credits.copy()

    credits["crew"] = credits["crew"].fillna("[]").apply(le)
    credits["director"] = credits["crew"].apply(get_director)

    return credits

def merge_movie_data(movies, keywords, credits):
    merged = movies.merge(keywords, on="id", how="inner")
    merged = merged.merge(
        credits[["id", "cast", "director"]],
        on="id",
        how="inner"
    )
    return merged


def create_year_feature(movies_df):
    movies_df = movies_df.copy()

    movies_df["year"] = pd.to_datetime(
        movies_df["release_date"],
        errors="coerce"
    ).dt.year

    movies_df["year"] = movies_df["year"].fillna(-1).astype(int)

    return movies_df


def create_title_year(movies_df):
    movies_df = movies_df.copy()

    movies_df["title"] = movies_df["title"].fillna("Unknown")

    movies_df["title_year"] = (
        movies_df["title"].astype(str)
        + " ("
        + movies_df["year"].astype(str)
        + ")"
    )

    return movies_df

def clean_list(items):
    if not isinstance(items, list):
        return []
    return [
        str(item).replace(" ", "").lower()
        for item in items
        if item is not None
    ]


def clean_name(name):
    if not isinstance(name, str):
        return ""
    return name.replace(" ", "").lower()