from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from src.content_based import recommend

app = Flask(__name__)

# LOAD MODEL
movies_df = pickle.load(open("models/movies_df.pkl", "rb"))
count_matrix = pickle.load(open("models/count_matrix.pkl", "rb"))
indices = pickle.load(open("models/indices.pkl", "rb"))

# list titles for autocomplete
movie_titles = list(indices.index)


# ===== HOME =====
@app.route("/")
def home():
    return render_template("index.html")


# ===== AUTOCOMPLETE =====
@app.route("/suggest")
def suggest():
    query = request.args.get("q", "").lower()

    if not query:
        return jsonify([])

    results = [
        t for t in movie_titles
        if query in t.lower()
    ][:8]

    return jsonify(results)


# ===== SEARCH =====
@app.route("/search", methods=["POST"])
def search():

    data = request.json
    title = data.get("title")

    if title not in indices:
        return jsonify({"error": "not_found"})

    results = recommend(
        title=title,
        movies_df=movies_df,
        count_matrix=count_matrix,
        indices=indices,
        top_n=10
    )

    return jsonify(results.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)