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


import numpy as np


import numpy as np

class MatrixFactorization:

    def __init__(
        self,
        n_factors=40,
        learning_rate=0.01,
        reg=0.005,
        epochs=50
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs

    def fit(self, ratings_df):

        # ===== MAP IDS =====
        self.user_ids = ratings_df["userId"].unique()
        self.movie_ids = ratings_df["movieId"].unique()

        self.user_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_map = {mid: i for i, mid in enumerate(self.movie_ids)}

        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)

        # ===== GLOBAL BIAS =====
        self.mu = ratings_df["rating"].mean()

        # ===== INIT PARAMETERS =====
        self.P = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.Q = np.random.normal(scale=0.1, size=(n_movies, self.n_factors))

        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_movies)

        # ===== TRAIN =====
        for epoch in range(self.epochs):

            total_error = 0

            # shuffle mỗi epoch (QUAN TRỌNG)
            ratings_df = ratings_df.sample(frac=1).reset_index(drop=True)

            for row in ratings_df.itertuples():

                u = self.user_map[row.userId]
                i = self.movie_map[row.movieId]

                rating = row.rating

                # ===== PREDICT =====
                pred = (
                    self.mu
                    + self.b_u[u]
                    + self.b_i[i]
                    + np.dot(self.P[u], self.Q[i])
                )

                error = rating - pred
                total_error += error ** 2

                # ===== UPDATE BIAS =====
                self.b_u[u] += self.learning_rate * (
                    error - self.reg * self.b_u[u]
                )

                self.b_i[i] += self.learning_rate * (
                    error - self.reg * self.b_i[i]
                )

                # ===== UPDATE LATENT FACTORS =====
                pu = self.P[u].copy()

                self.P[u] += self.learning_rate * (
                    error * self.Q[i] - self.reg * self.P[u]
                )

                self.Q[i] += self.learning_rate * (
                    error * pu - self.reg * self.Q[i]
                )

            rmse = np.sqrt(total_error / len(ratings_df))
            print(f"Epoch {epoch+1}/{self.epochs} - RMSE: {rmse:.4f}")

        return self

    def predict(self, user_id, movie_id):

        if user_id not in self.user_map:
            return None
        if movie_id not in self.movie_map:
            return None

        u = self.user_map[user_id]
        i = self.movie_map[movie_id]

        pred = (
            self.mu
            + self.b_u[u]
            + self.b_i[i]
            + np.dot(self.P[u], self.Q[i])
        )

        return np.clip(pred, 1, 5)