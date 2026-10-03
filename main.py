"""
(c) Copyright 2026 | Prince Sibanda

This is an ML model for an artist recommendation system.

main.py - Training code
"""

# Import needed modules
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load data from CSV
df = pd.read_csv('most_streamed_artists.csv')

# Important features needed to avoid overfitting and underfitting
FEATURES = ['Country of Origin', 'Primary Language', 'Primary Genre', 'Debut Year']

# Return necessary features
def get_features():
    return df[FEATURES]

X = get_features()

# Preprocessor for turning everything into numbers and squishing numbers
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Country of Origin', 'Primary Language', 'Primary Genre']),
('num', StandardScaler(), ['Debut Year'])
])

X_processed = preprocessor.fit_transform(X)

similarity_matrix = cosine_similarity(X_processed)

# def recommend(artist_name, top_n=5):
#     idx = df[df['Artist Name'] == artist_name].index[0]

#     sim_scores = list(enumerate(similarity_matrix[idx]))

#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

#     artist_indices = [i[0] for i in sim_scores]

#     return df[['Artist Name', 'Country of Origin', 'Primary Language', 'Primary Genre']].iloc[artist_indices]

# print(recommend('Eminem', 100))

#  Save model
def save_model():
    joblib.dump(similarity_matrix, 'artists_similarity_matrix.pkl')
    print('MODEL SAVED!!!')

save_model()