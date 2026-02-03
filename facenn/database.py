import os
import pandas as pd
import torch
import pickle
import numpy as np
from facenn.config import DEVICE
from facenn.utils.distance import find_cosine_distance, find_euclidean_distance

class FaceDB:
    def __init__(self, db_path: str = "facenn_db.pkl"):
        self.db_path = db_path
        self.embeddings = [] # List of tensors or numpy arrays
        self.identities = [] # List of strings/metadata
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.identities = data['identities']
    
    def save(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'identities': self.identities
            }, f)

    def add_face(self, embedding, identity):
        """
        Adds a face to the database.
        embedding: List, Numpy array, or Tensor
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy() # Store as numpy to save space/compat
            
        self.embeddings.append(embedding)
        self.identities.append(identity)

    def search(self, embedding, k=1, metric="cosine", threshold=0.4):
        """
        Searches for the k nearest faces.
        Returns: DataFrame with columns ['identity', 'distance']
        """
        if not self.embeddings:
            return pd.DataFrame(columns=['identity', 'distance'])

        # Convert DB to tensor on DEVICE for fast search
        # check if embeddings list is not empty and consists of numpy arrays
        # Stack them into a tensor
        db_tensor = torch.tensor(np.array(self.embeddings)).float().to(DEVICE)
        
        if isinstance(embedding, torch.Tensor):
            query_tensor = embedding.to(DEVICE)
        else:
            query_tensor = torch.tensor(np.array(embedding)).float().to(DEVICE)
        
        if len(query_tensor.shape) == 1:
            query_tensor = query_tensor.unsqueeze(0)
            
        # If query is (1, D) and db is (N, D), we expect output (1, N)
        if len(db_tensor.shape) == 3: # If stored as (1, D)
             db_tensor = db_tensor.squeeze(1)

        if metric == 'cosine':
            distances = find_cosine_distance(query_tensor, db_tensor)
        elif metric == 'euclidean':
            distances = find_euclidean_distance(query_tensor, db_tensor)
        else:
             raise ValueError(f"Unknown metric: {metric}")
             
        distances = distances.squeeze(0).cpu().numpy()
        
        # Filter by threshold
        indices = np.where(distances < threshold)[0]
        
        results = []
        for idx in indices:
            results.append({
                'identity': self.identities[idx],
                'distance': distances[idx]
            })
            
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by='distance', ascending=True).head(k)
            
        return df
