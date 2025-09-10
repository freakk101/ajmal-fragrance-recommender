import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F

class FragranceEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 96]):
        super(FragranceEmbeddingNet, self).__init__()
        
        # Create encoder layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        # Final embedding layer
        layers.extend([
            nn.Linear(current_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh()  # Normalize embeddings to [-1, 1]
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Decoder for reconstruction (autoencoder approach)
        decoder_layers = []
        current_dim = embedding_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction
    
    def get_embedding(self, x):
        with torch.no_grad():
            return self.encoder(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

class FragranceRecommender:
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FragranceEmbeddingNet(input_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss()
        self.fragrance_embeddings = None
        
    def create_training_pairs(self, vectors: np.ndarray, similarity_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create positive and negative pairs for contrastive learning"""
        n_samples = len(vectors)
        pairs_1, pairs_2, labels = [], [], []
        
        # Calculate cosine similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(vectors)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                pairs_1.append(vectors[i])
                pairs_2.append(vectors[j])
                
                # Label as similar (0) or dissimilar (1) based on threshold
                if similarities[i, j] > similarity_threshold:
                    labels.append(0)  # Similar
                else:
                    labels.append(1)  # Dissimilar
        
        return np.array(pairs_1), np.array(pairs_2), np.array(labels)
    
    def train(self, vectors: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the neural network"""
        self.model.train()
        
        # Create training pairs
        pairs_1, pairs_2, labels = self.create_training_pairs(vectors)
        
        # Convert to tensors
        vectors_tensor = torch.FloatTensor(vectors).to(self.device)
        pairs_1_tensor = torch.FloatTensor(pairs_1).to(self.device)
        pairs_2_tensor = torch.FloatTensor(pairs_2).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        n_batches = len(pairs_1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle pairs
            indices = torch.randperm(len(pairs_1))
            pairs_1_tensor = pairs_1_tensor[indices]
            pairs_2_tensor = pairs_2_tensor[indices]
            labels_tensor = labels_tensor[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_pairs_1 = pairs_1_tensor[start_idx:end_idx]
                batch_pairs_2 = pairs_2_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                embedding_1, recon_1 = self.model(batch_pairs_1)
                embedding_2, recon_2 = self.model(batch_pairs_2)
                
                # Calculate losses
                contrastive_loss = self.contrastive_loss(embedding_1, embedding_2, batch_labels)
                reconstruction_loss_1 = self.mse_loss(recon_1, batch_pairs_1)
                reconstruction_loss_2 = self.mse_loss(recon_2, batch_pairs_2)
                
                # Combined loss
                loss = contrastive_loss + 0.5 * (reconstruction_loss_1 + reconstruction_loss_2)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
        
        # Generate embeddings for all fragrances
        self.model.eval()
        with torch.no_grad():
            self.fragrance_embeddings = self.model.get_embedding(vectors_tensor).cpu().numpy()
    
    def recommend_fragrances(self, selected_indices: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """Recommend fragrances based on user selections"""
        if self.fragrance_embeddings is None:
            raise ValueError("Model must be trained first")
        
        # Average embeddings of selected fragrances
        selected_embeddings = self.fragrance_embeddings[selected_indices]
        user_profile = np.mean(selected_embeddings, axis=0)
        
        # Calculate similarities with all fragrances
        similarities = []
        for i, embedding in enumerate(self.fragrance_embeddings):
            if i not in selected_indices:  # Exclude already selected fragrances
                # Use both cosine similarity and euclidean distance
                cosine_sim = np.dot(user_profile, embedding) / (np.linalg.norm(user_profile) * np.linalg.norm(embedding))
                euclidean_dist = np.linalg.norm(user_profile - embedding)
                
                # Combined similarity score (higher is better)
                combined_score = cosine_sim - 0.1 * euclidean_dist
                similarities.append((i, combined_score))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embeddings': self.fragrance_embeddings
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.fragrance_embeddings = checkpoint['embeddings']