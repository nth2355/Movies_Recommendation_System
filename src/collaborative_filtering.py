import pandas as pd


def prepare_ratings_data(
    ratings,
    links,
    movies_df
):

    links = links.dropna(
        subset=["tmdbId"]
    )

    links["tmdbId"] = (
        links["tmdbId"]
        .astype(int)
    )

    movies_cf = links.merge(
        movies_df,
        left_on="tmdbId",
        right_on="id",
        how="inner"
    )

    ratings_movies = ratings.merge(
        movies_cf[
            [
                "movieId",
                "title",
                "title_year"
            ]
        ],
        on="movieId",
        how="inner"
    )
    return ratings_movies



def build_user_item_matrix(ratings_movies):
    user_item_matrix = ratings_movies.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return user_item_matrix