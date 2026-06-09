import pandas as pd
import pickle

from src.data_preprocessing import (
    clean_movies_ids,
    parse_genres,
    parse_keywords,
    parse_cast,
    parse_director,
    merge_movie_data,
    create_year_feature,
    create_title_year
)

from src.content_based import (
    create_soup,
    build_count_matrix,
    create_indices
)


def train_and_save():

    movies = pd.read_csv("data/movies_metadata.csv")
    keywords = pd.read_csv("data/keywords.csv")
    credits = pd.read_csv("data/credits.csv")

    # preprocess
    movies = clean_movies_ids(movies)
    movies = parse_genres(movies)
    keywords = parse_keywords(keywords)
    credits = parse_cast(credits)
    credits = parse_director(credits)

    movies = merge_movie_data(movies, keywords, credits)
    movies = create_year_feature(movies)
    movies = create_title_year(movies)

    # create soup
    movies["soup"] = movies.apply(create_soup, axis=1)

    # vector
    count_matrix = build_count_matrix(movies)
    indices = create_indices(movies)

    # movie id map
    movie_id_map = {row["id"]: idx for idx, row in movies.iterrows()}

    # save
    import os
    os.makedirs("models", exist_ok=True)

    with open("models/movies_df.pkl", "wb") as f:
        pickle.dump(movies, f)

    with open("models/count_matrix.pkl", "wb") as f:
        pickle.dump(count_matrix, f)

    with open("models/indices.pkl", "wb") as f:
        pickle.dump(indices, f)

    with open("models/movie_id_map.pkl", "wb") as f:
        pickle.dump(movie_id_map, f)

    print("Training done and saved!")


if __name__ == "__main__":
    train_and_save()