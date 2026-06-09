import numpy as np

def rmse(model, test_df):
    errors = []

    for row in test_df.itertuples():

        pred = model.predict(row.userId, row.movieId)

        if pred is None:
            continue

        errors.append((row.rating - pred) ** 2)

    return np.sqrt(np.mean(errors))