"""
(c) Copyright 2026 | Prince Sibanda

This is an ML model for an artist recommendation system.

inference.py - ML model running
"""

import pandas as pd
import joblib
import sys
import argparse

df = pd.read_csv('most_streamed_artists.csv')

# Load matrix with similarity scores for every artist
similarity_matrix = joblib.load('artists_similarity_matrix.pkl')

def recommend(artist_name, top_n=5):
    try:
        idx = df[df['Artist Name'] == artist_name].index[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        artist_indices = [i[0] for i in sim_scores]
        return df[['Artist Name', 'Country of Origin', 'Primary Language', 'Primary Genre']].iloc[artist_indices]
    except:
        print("Artist not found in the db")

# Testing
# print(recommend('', top_n=10))

if "main" in __name__:
    artist_name = ' '.join(sys.argv[1:])
    # print(artist_name)
    print(recommend(artist_name))