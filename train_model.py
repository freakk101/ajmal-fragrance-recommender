#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from data_preprocessor import FragranceDataPreprocessor
from sklearn_recommender import FragranceRecommender

def train_fragrance_recommender():
    """Train the fragrance recommendation model"""
    print("🌸 Starting fragrance recommendation model training...")
    
    # Initialize data preprocessor
    csv_path = "chrome_ajmal_fragrance_data_20250909_202423.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print("📊 Loading and preprocessing data...")
    preprocessor = FragranceDataPreprocessor(csv_path)
    df, vectors, notes_list = preprocessor.get_processed_data()
    
    print(f"✅ Loaded {len(df)} fragrances with {len(notes_list)} unique notes")
    print(f"📏 Feature vector dimension: {vectors.shape[1]}")
    
    # Initialize and train the recommender
    print("🧠 Initializing neural network...")
    recommender = FragranceRecommender(embedding_dim=32)
    
    print("🏋️ Training model...")
    recommender.train(vectors, epochs=500)
    
    # Save the trained model and preprocessed data
    print("💾 Saving model and data...")
    recommender.save_model("fragrance_model.pkl")
    
    # Save processed dataframe and notes list
    df.to_pickle("processed_fragrances.pkl")
    np.save("fragrance_vectors.npy", vectors)
    np.save("fragrance_notes.npy", np.array(notes_list, dtype=object))
    
    print("✅ Training completed successfully!")
    print(f"📂 Model saved as: fragrance_model.pkl")
    print(f"📂 Data saved as: processed_fragrances.pkl")
    
    # Test the model with a sample recommendation
    print("\n🧪 Testing with sample recommendations...")
    test_indices = [0, 1]  # First two fragrances
    recommendations = recommender.recommend_fragrances(test_indices, top_k=5)
    
    print(f"Selected fragrances: {df.iloc[test_indices]['product_name'].tolist()}")
    print("Top 5 recommendations:")
    for idx, score in recommendations:
        print(f"  - {df.iloc[idx]['product_name']} (score: {score:.3f})")

if __name__ == "__main__":
    train_fragrance_recommender()