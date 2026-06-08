import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def clean_name(name):
    return str(name).replace(" ", "").lower()


def clean_list(items):
    return [
        str(item).replace(" ", "").lower()
        for item in items
    ]




def create_soup(row):
    return " ".join(
        row["genres"]
        + row["keywords"]
        + row["cast"]
        + [row["director"]] * 3
    )



def build_count_matrix(movies_df):

    vectorizer = CountVectorizer(
        stop_words="english"
    )

    count_matrix = vectorizer.fit_transform(
        movies_df["soup"]
    )

    return count_matrix



def create_indices(movies_df):

    indices = pd.Series(
        movies_df.index,
        index=movies_df["title_year"]
    )

    return indices



def recommend(
    title,
    movies_df,
    count_matrix,
    indices,
    top_n=10
):

    if title not in indices:
        raise ValueError(
            f"Movie '{title}' not found."
        )

    idx = indices[title]
    movie_vector = count_matrix[idx]

    sim_scores = cosine_similarity(
        movie_vector,
        count_matrix
    ).flatten()

    sim_scores = list(
        enumerate(sim_scores)
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )
    sim_scores = sim_scores[1: top_n + 1]
    movie_indices = [
        i[0]
        for i in sim_scores
    ]
    return movies_df.iloc[movie_indices][
        [
            "title",
            "vote_average",
            "release_date"
        ]
    ]