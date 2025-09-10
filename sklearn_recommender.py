import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import pickle

class FragranceRecommender:
    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=embedding_dim)
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(64, embedding_dim, 64),
            activation='tanh',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        self.fragrance_embeddings = None
        self.is_trained = False
        
    def create_enhanced_features(self, vectors: np.ndarray) -> np.ndarray:
        """Create enhanced features using clustering and PCA"""
        # Normalize the input vectors
        vectors_scaled = self.scaler.fit_transform(vectors)
        
        # Add clustering features
        n_clusters = min(10, len(vectors) // 3)  # Adaptive cluster count
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors_scaled)
            
            # Create cluster distance features
            cluster_distances = []
            for i, center in enumerate(kmeans.cluster_centers_):
                distances = euclidean_distances(vectors_scaled, center.reshape(1, -1)).flatten()
                cluster_distances.append(distances)
            
            cluster_features = np.column_stack(cluster_distances)
            enhanced_vectors = np.hstack([vectors_scaled, cluster_features])
        else:
            enhanced_vectors = vectors_scaled
            
        return enhanced_vectors
    
    def train(self, vectors: np.ndarray, epochs: int = 500):
        """Train the autoencoder-based recommender"""
        print(f"Training recommender with {len(vectors)} fragrances...")
        
        # Create enhanced features
        enhanced_vectors = self.create_enhanced_features(vectors)
        
        # Train autoencoder (using the vectors as both input and target)
        self.autoencoder.fit(enhanced_vectors, enhanced_vectors)
        
        # Get the hidden layer activations as embeddings
        # We'll use the encoder part for embeddings
        embeddings = self._get_embeddings(enhanced_vectors)
        
        # Apply PCA to reduce dimensionality further and capture main components
        self.fragrance_embeddings = self.pca.fit_transform(embeddings)
        
        self.is_trained = True
        print(f"Training completed. Generated {self.fragrance_embeddings.shape[1]}D embeddings.")
        
    def _get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from the trained autoencoder"""
        # Get hidden layer activations
        hidden_layers = []
        current_input = X
        
        # Forward pass through encoder layers
        for i, layer in enumerate(self.autoencoder.coefs_):
            if i < len(self.autoencoder.coefs_) // 2:  # Encoder part
                current_input = np.tanh(current_input @ layer + self.autoencoder.intercepts_[i])
                hidden_layers.append(current_input)
        
        # Return the bottleneck layer (middle layer)
        return hidden_layers[-1] if hidden_layers else X
    
    def recommend_fragrances(self, selected_indices: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Recommend fragrances based on user selections using advanced similarity"""
        if not self.is_trained or self.fragrance_embeddings is None:
            raise ValueError("Model must be trained first")
        
        # Get embeddings of selected fragrances
        selected_embeddings = self.fragrance_embeddings[selected_indices]
        
        # Create user profile using weighted average (more weight to first selection)
        weights = np.array([1.0 - 0.1 * i for i in range(len(selected_indices))])
        weights = weights / weights.sum()
        user_profile = np.average(selected_embeddings, axis=0, weights=weights)
        
        # Calculate multiple similarity metrics
        similarities = []
        for i, embedding in enumerate(self.fragrance_embeddings):
            if i not in selected_indices:
                # Cosine similarity
                cosine_sim = cosine_similarity([user_profile], [embedding])[0][0]
                
                # Euclidean distance (inverted and normalized)
                euclidean_dist = euclidean_distances([user_profile], [embedding])[0][0]
                euclidean_sim = 1 / (1 + euclidean_dist)
                
                # Dot product similarity
                dot_product = np.dot(user_profile, embedding)
                
                # Combined similarity score
                combined_score = (
                    0.5 * cosine_sim + 
                    0.3 * euclidean_sim + 
                    0.2 * dot_product
                )
                
                similarities.append((i, combined_score))
        
        # Sort by similarity score and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_fragrance_clusters(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """Cluster fragrances based on their embeddings"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.fragrance_embeddings)
        
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
            
        return clusters
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_data = {
            'autoencoder': self.autoencoder,
            'pca': self.pca,
            'scaler': self.scaler,
            'embeddings': self.fragrance_embeddings,
            'embedding_dim': self.embedding_dim,
            'is_trained': self.is_trained
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load a trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.autoencoder = model_data['autoencoder']
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self.fragrance_embeddings = model_data['embeddings']
        self.embedding_dim = model_data['embedding_dim']
        self.is_trained = model_data['is_trained']