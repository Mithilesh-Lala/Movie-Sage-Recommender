import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import joblib
import os

# File paths
MODEL_PATH = 'recommendation_model.joblib'
DATA_PATH = 'movies.csv'
RATINGS_PATH = 'ratings.csv'

# Load and preprocess data
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH) or not os.path.exists(RATINGS_PATH):
        st.error(f"Data files not found. Please ensure '{DATA_PATH}' and '{RATINGS_PATH}' are in the project folder.")
        return None, None
    
    try:
        movies = pd.read_csv(DATA_PATH)
        ratings = pd.read_csv(RATINGS_PATH)
        
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        movies_with_stats = pd.merge(movies, movie_stats, on='movieId')
        popular_movies = movies_with_stats[movies_with_stats['num_ratings'] >= 50].reset_index(drop=True)
        
        return popular_movies, ratings
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load sentence transformer model for encoding
@st.cache_resource
def load_sentence_model():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {str(e)}")
        return None

def create_movie_description(row):
    return f"{row['title']} is a {row['genres']} movie. It has an average rating of {row['avg_rating']:.2f} from {row['num_ratings']} ratings."

def build_recommendation_model(popular_movies, sentence_model):
    st.info("Building recommendation model. This may take a few minutes...")
    
    try:
        # Content-based features
        popular_movies['description'] = popular_movies.apply(create_movie_description, axis=1)
        content_features = popular_movies['description'].apply(sentence_model.encode)
        
        # Convert to numpy array for efficient storage
        content_features_array = np.array(content_features.tolist())
        
        # Save the features
        joblib.dump(content_features_array, MODEL_PATH)
        
        st.success("Recommendation model built and saved successfully!")
        return content_features_array
    except Exception as e:
        st.error(f"Error building recommendation model: {str(e)}")
        return None

def load_recommendation_model(popular_movies, sentence_model):
    if os.path.exists(MODEL_PATH):
        try:
            content_features = joblib.load(MODEL_PATH)
            st.success("Recommendation model loaded successfully!")
            return content_features
        except Exception as e:
            st.warning(f"Error loading existing model: {str(e)}. Attempting to rebuild...")
    
    return build_recommendation_model(popular_movies, sentence_model)

def get_recommendations(input_text, popular_movies, content_features, sentence_model, n=10):
    if popular_movies is None or content_features is None or sentence_model is None:
        st.error("Recommendation system is not properly initialized.")
        return pd.DataFrame()
    
    try:
        # Extract features from input text
        input_features = sentence_model.encode([input_text])
        
        # Calculate similarity
        similarities = cosine_similarity(input_features, content_features)
        
        # Get top n similar movie indices
        similar_indices = similarities.argsort()[0][-n:][::-1]
        
        # Get the corresponding movies
        recommendations = popular_movies.iloc[similar_indices]
        return recommendations[['title', 'genres', 'avg_rating', 'num_ratings']]
    except Exception as e:
        st.error(f"An error occurred while getting recommendations: {str(e)}")
        return pd.DataFrame()

# Streamlit GUI
st.title('Advanced Movie Recommender System')

# Initialize system
popular_movies, ratings = load_data()
sentence_model = load_sentence_model()

if popular_movies is not None and ratings is not None and sentence_model is not None:
    content_features = load_recommendation_model(popular_movies, sentence_model)
    
    if content_features is not None:
        st.write("Enter a movie description or name, and we'll find similar movies for you using our advanced AI!")
        
        user_input = st.text_area('Enter a movie name:', height=100)
        
        if user_input:
            with st.spinner('Analyzing your input and finding the best matches...'):
                recommendations = get_recommendations(user_input, popular_movies, content_features, sentence_model)
            
            if not recommendations.empty:
                st.write("Top 10 recommendations based on your input:")
                for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                    st.write(f"**{i}. {movie['title']}** ({movie['genres']})")
                    st.write(f"Average Rating: {movie['avg_rating']:.2f} from {movie['num_ratings']} ratings")
                    st.write("---")

                # Provide an option to see more details
                if st.button('Show more details'):
                    st.table(recommendations)
            else:
                st.warning("No recommendations found. Please try a different input.")
        
        # Option to rebuild the model
        if st.button('Rebuild Recommendation Model'):
            content_features = build_recommendation_model(popular_movies, sentence_model)
            if content_features is not None:
                st.success("Model rebuilt successfully!")
    else:
        st.error("Failed to initialize the recommendation model. Please check the error messages above and try again.")
else:
    st.error("Failed to initialize the recommendation system. Please check the error messages above and ensure all required files are present.")