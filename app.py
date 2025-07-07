import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib 


df = pd.read_csv(r"E:\TakeYourFilm\Cleaned_dataset\merged_movies.csv")

model = joblib.load("film_model.pkl") 

encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎬 Find Your Film")
st.write("Enter a genre to get top movie recommendations!")

genre_name = st.selectbox("Select Genre:", ['Fantasy', 'Family', 'Sci-fi', 'Action', 'Thriller', 'Mystery', 'Romance', 'Horror'])

def recommend_movies(genre_name, top_n=10):
    genre_movies = df[df['genre'].str.contains(genre_name, case=False, na=False)]
    
    if genre_movies.empty:
        st.warning(f"No movies found for genre: {genre_name}")
        return None

    X_genre = genre_movies[['genre', 'rating', 'votes']]
    genre_encoded = encoder.transform(X_genre[['genre']])
    scaled_features = scaler.transform(X_genre[['rating', 'votes']])
    X_genre_processed = np.hstack((genre_encoded, scaled_features))

    predicted_scores = model.predict(X_genre_processed)
    
    genre_movies = genre_movies.copy()
    genre_movies['predicted_score'] = predicted_scores
    top_movies = genre_movies.sort_values(by='predicted_score', ascending=False).head(top_n)

    return top_movies[['movie_name', 'rating', 'votes', 'predicted_score']]

if st.button("Top Rated movies"):
    recommended_movies = recommend_movies(genre_name)
    
    if recommended_movies is not None:
        st.write(recommended_movies)




#  HOW TO RUN THIS PROGRAM
# python -m streamlit run app.py