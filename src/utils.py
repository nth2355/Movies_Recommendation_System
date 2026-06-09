from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_content_score(movie_id, movie_id_to_idx, count_matrix):

    idx = movie_id_to_idx[movie_id]

    vec = count_matrix[idx]

    sims = cosine_similarity(vec, count_matrix).flatten()

    return np.mean(sims)