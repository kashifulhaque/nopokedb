import os
import json
import sqlite3
import hnswlib
import numpy as np

class NoPokeDB:
  """
  A small vector database with HNSW index (hnswlib) and SQLite-backed metadata.
  Persists both index and metadata on disk for durability.
  """
  def __init__(
    self,
    dim: int,
    max_elements: int,
    path: str = "./data",
    space: str = "cosine",
    M: int = 16,
    ef_construction: int = 200,
    ef: int = 50
  ):
    self.dim = dim
    self.max_elements = max_elements
    self.space = space
    self.M = M
    self.ef_construction = ef_construction
    self.ef = ef
    self.path = path
    os.makedirs(self.path, exist_ok=True)

    self.index_path = os.path.join(self.path, "hnsw_index.bin")
    self.db_path = os.path.join(self.path, "metadata.db")

    self.index = hnswlib.Index(space=self.space, dim=self.dim)
    if os.path.exists(self.index_path):
      self.index.load_index(self.index_path)
    else:
      self.index.init_index(
        max_elements=self.max_elements,
        M=self.M,
        ef_construction=self.ef_construction
      )
    self.index.set_ef(self.ef)

    self.conn = sqlite3.connect(self.db_path)
    self._ensure_table()
    self._next_id = self._get_max_id() + 1

  def _ensure_table(self):
    cur = self.conn.cursor()
    cur.execute(
      """
      CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY,
        data TEXT NOT NULL
      )
      """
    )
    self.conn.commit()

  def _get_max_id(self) -> int:
    cur = self.conn.cursor()
    cur.execute("SELECT MAX(id) FROM metadata")
    row = cur.fetchone()
    return row[0] or -1

  def add(self, vector: np.ndarray, metadata: dict):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (self.dim,):
      raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
    vid = self._next_id
    self._next_id += 1

    self.index.add_items(vector.reshape(1, -1), np.array([vid], dtype=int))
    cur = self.conn.cursor()
    cur.execute(
      "INSERT INTO metadata (id, data) VALUES (?, ?)",
      (vid, json.dumps(metadata))
    )
    self.conn.commit()
    return vid

  def query(self, vector: np.ndarray, k: int = 5):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (self.dim,):
      raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
    labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k)
    results = []
    for lbl, dist in zip(labels[0], distances[0]):
      sim = 1.0 - dist  # cosine similarity
      cur = self.conn.cursor()
      cur.execute("SELECT data FROM metadata WHERE id = ?", (int(lbl),))
      row = cur.fetchone()
      md = json.loads(row[0]) if row else None
      results.append({
        "id": int(lbl),
        "metadata": md,
        "score": float(sim)
      })
    return results

  def save(self):
    """
    Manually persist the HNSW index to disk.
    Metadata is auto-committed on each add.
    """
    self.index.save_index(self.index_path)

  def close(self):
    """
    Save index and close SQLite connection.
    """
    self.save()
    self.conn.close()
