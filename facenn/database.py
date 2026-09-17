from typing import Any, Dict, List, Optional, Union
import os
import sqlite3
import numpy as np
import pandas as pd
import torch

from facenn.config import Config


class FaceDB:
    """
    Persistent and secure vector database for face identities.
    Uses SQLite for metadata/storage and an in-memory normalized float32 matrix for fast search.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(Config.ensure_facenn_home(), "identities.sqlite")
        self.db_path = db_path

        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._matrix: Optional[np.ndarray] = None  # (N, D) float32 normalized
        self._identities: List[str] = []
        self._ids: List[int] = []

        self._init_db()
        self.load()

    def _init_db(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_identity ON faces(identity)")

    def load(self):
        """Loads all embeddings from the database into in-memory search matrix."""
        cursor = self._conn.execute("SELECT id, identity, dim, embedding FROM faces ORDER BY id ASC")
        rows = cursor.fetchall()

        self._ids = []
        self._identities = []
        embeddings = []

        for row in rows:
            arr = np.frombuffer(row["embedding"], dtype=np.float32).reshape(row["dim"])
            # Ensure normalized
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            embeddings.append(arr)
            self._ids.append(row["id"])
            self._identities.append(row["identity"])

        if embeddings:
            self._matrix = np.vstack(embeddings).astype(np.float32)
        else:
            self._matrix = None

    def add_face(self, embedding: Union[np.ndarray, torch.Tensor, list], identity: str) -> int:
        """
        Adds a face embedding and its associated identity to the database.
        Returns the inserted record ID.
        """
        if isinstance(embedding, torch.Tensor):
            vec = embedding.detach().cpu().numpy()
        else:
            vec = np.asarray(embedding)

        vec = vec.astype(np.float32).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec_norm = vec / norm
        else:
            vec_norm = vec

        dim = len(vec_norm)
        blob = vec_norm.tobytes()

        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO faces (identity, dim, embedding) VALUES (?, ?, ?)",
                (identity, dim, blob),
            )
            new_id = cursor.lastrowid

        self._ids.append(new_id)
        self._identities.append(identity)

        if self._matrix is None:
            self._matrix = vec_norm.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vec_norm.reshape(1, -1)])

        return new_id

    def search(
        self,
        embedding: Union[np.ndarray, torch.Tensor, list],
        k: int = 5,
        metric: str = "cosine",
        threshold: float = 0.4,
    ) -> pd.DataFrame:
        """
        Searches for the k closest matching identities within the given distance threshold.
        Returns a pandas DataFrame with columns ['identity', 'distance', 'id'].
        """
        columns = ["identity", "distance", "id"]
        if self._matrix is None or len(self._identities) == 0:
            return pd.DataFrame(columns=columns)

        if isinstance(embedding, torch.Tensor):
            query = embedding.detach().cpu().numpy()
        else:
            query = np.asarray(embedding)

        query = query.astype(np.float32).ravel()
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        if metric == "cosine":
            # Since vectors are unit-normalized: cosine_distance = 1 - dot_product
            sims = np.dot(self._matrix, query)
            distances = 1.0 - sims
        elif metric == "euclidean":
            distances = np.linalg.norm(self._matrix - query, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'euclidean'.")

        valid_indices = np.where(distances <= threshold)[0]
        if len(valid_indices) == 0:
            return pd.DataFrame(columns=columns)

        matched_distances = distances[valid_indices]
        sort_order = np.argsort(matched_distances)[:k]
        top_indices = valid_indices[sort_order]

        results = [
            {
                "identity": self._identities[i],
                "distance": float(distances[i]),
                "id": self._ids[i],
            }
            for i in top_indices
        ]

        return pd.DataFrame(results)

    def delete(self, identity: Optional[str] = None, face_id: Optional[int] = None) -> int:
        """Deletes face records by identity string or ID."""
        with self._conn:
            if face_id is not None:
                cursor = self._conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            elif identity is not None:
                cursor = self._conn.execute("DELETE FROM faces WHERE identity = ?", (identity,))
            else:
                raise ValueError("Specify either identity or face_id to delete")
            deleted_count = cursor.rowcount

        self.load()
        return deleted_count

    def list_identities(self) -> List[str]:
        """Returns unique list of registered identities."""
        cursor = self._conn.execute("SELECT DISTINCT identity FROM faces ORDER BY identity ASC")
        return [row["identity"] for row in cursor.fetchall()]

    def count(self) -> int:
        return len(self._identities)

    def clear(self):
        """Clears all records from the database."""
        with self._conn:
            self._conn.execute("DELETE FROM faces")
        self._matrix = None
        self._identities = []
        self._ids = []

    def save(self):
        """Legacy compatibility no-op (SQLite is saved on transaction commit)."""
        pass

