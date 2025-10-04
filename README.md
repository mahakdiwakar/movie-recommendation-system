# -------------------------------------------
# Step 0: Install required libraries
# -------------------------------------------
!pip install gradio pandas numpy scikit-learn tensorflow

# -------------------------------------------
# Step 1: Import Libraries
# -------------------------------------------
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# -------------------------------------------
# Step 2: Download and Load MovieLens Dataset
# -------------------------------------------
!wget -nc http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip -n ml-latest-small.zip

# Load datasets
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# -------------------------------------------
# Step 3: Preprocess Data
# -------------------------------------------
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for x, i in movie2movie_encoded.items()}

ratings["user"] = ratings["userId"].map(user2user_encoded)
ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie2movie_encoded)

X = ratings[["user", "movie"]].values
y = ratings["rating"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# Step 4: Build Neural Collaborative Filtering Model
# -------------------------------------------
def create_model(init_method="glorot_uniform"):
    user_input = tf.keras.layers.Input(shape=(1,))
    user_emb = tf.keras.layers.Embedding(num_users, 50, embeddings_initializer=init_method)(user_input)
    user_emb = tf.keras.layers.Flatten()(user_emb)

    movie_input = tf.keras.layers.Input(shape=(1,))
    movie_emb = tf.keras.layers.Embedding(num_movies, 50, embeddings_initializer=init_method, name="movie_embedding")(movie_input)
    movie_emb_flat = tf.keras.layers.Flatten()(movie_emb)

    concat = tf.keras.layers.Concatenate()([user_emb, movie_emb_flat])
    dense = tf.keras.layers.Dense(128, activation="relu", kernel_initializer=init_method)(concat)
    dense = tf.keras.layers.Dense(64, activation="relu", kernel_initializer=init_method)(dense)
    output = tf.keras.layers.Dense(1, kernel_initializer=init_method)(dense)

    model = tf.keras.Model([user_input, movie_input], output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

model = create_model("he_normal")

# -------------------------------------------
# Step 5: Train Model
# -------------------------------------------
history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    validation_data=([X_test[:, 0], X_test[:, 1]], y_test),
    epochs=3, batch_size=64
)

# -------------------------------------------
# Step 6: Precompute Movie Embeddings
# -------------------------------------------
movie_embeddings_layer = model.get_layer("movie_embedding")
movie_embeddings = movie_embeddings_layer.get_weights()[0]  # shape = (num_movies, 50)

# -------------------------------------------
# Step 7: Recommendation Function (Cosine Similarity)
# -------------------------------------------
def recommend_movies(movie_name):
    if movie_name not in movies["title"].values:
        return "Movie not found! Please try another name."

    movie_id = movies[movies["title"] == movie_name]["movieId"].values[0]
    movie_idx = movie2movie_encoded[movie_id]

    # Cosine similarity between selected movie and all movies
    sim = cosine_similarity([movie_embeddings[movie_idx]], movie_embeddings)[0]

    # Get top 5 similar movies (exclude the movie itself)
    top_indices = sim.argsort()[-6:-1][::-1]
    recommended_ids = [movie_encoded2movie[idx] for idx in top_indices]
    recommended_titles = movies[movies["movieId"].isin(recommended_ids)]["title"].tolist()

    # Numbered essay-style output
    essay_style = ""
    for i, title in enumerate(recommended_titles, 1):
        essay_style += f"({i}) {title}\n"

    return essay_style

# -------------------------------------------
# Step 8: Gradio Frontend
# -------------------------------------------
iface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Textbox(label="Enter a Movie Name"),
    outputs=gr.Textbox(label="Top 5 Recommended Movies", lines=7),
    title="🎬 Movie Recommendation System",
    description="Type any movie name from the dataset to get 5 similar/recommended movies."
)

iface.launch()
