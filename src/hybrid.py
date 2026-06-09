import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(
        self,
        cf_model,
        movie_id_to_idx,
        count_matrix,
        alpha=0.7
    ):
        self.cf = cf_model
        self.movie_id_to_idx = movie_id_to_idx
        self.count_matrix = count_matrix
        self.alpha = alpha

    def _get_content_score(self, movie_id):
        if movie_id not in self.movie_id_to_idx:
            return 0.0 
            
        idx = self.movie_id_to_idx[movie_id]
        
        # Lấy ra vector của bộ phim hiện tại và reshape thành ma trận 2D (1, số_tính_chất)
        vec = self.count_matrix[idx].reshape(1, -1)
        
        # TỐI ƯU RAM: Chỉ tính toán tương đồng giữa 1 dòng duy nhất với toàn bộ ma trận
        sims = cosine_similarity(vec, self.count_matrix).flatten()
        
        return np.mean(sims)

    def recommend(self, user_id, movie_ids, top_n=10):
        results = []

        for movie_id in movie_ids:
            # CF SCORE 
            cf_score = self.cf.predict(user_id, movie_id)

            if cf_score is None:
                continue

            #content score(Tính on-demand từng phim)
            content_score = self._get_content_score(movie_id)

            #hybrid combination
            final_score = (
                self.alpha * cf_score +
                (1 - self.alpha) * content_score
            )

            results.append((movie_id, final_score))

        # Sort top-k
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]