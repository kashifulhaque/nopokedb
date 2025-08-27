import os
import json
import sqlite3
import hnswlib
import numpy as np
from typing import Dict, List, Sequence


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
    ef: int = 50,
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
        max_elements=self.max_elements, M=self.M, ef_construction=self.ef_construction
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

  def _fetch_metadata_bulk(self, ids: List[int]) -> Dict[int, dict]:
    """
    Fetch metadata for many ids in one shot. Returns {id: metadata}
    """
    if not ids:
      return {}

    qmarks = ",".join("?" for _ in ids)
    cur = self.conn.cursor()
    cur.execute(f"SELECT id, data FROM metadata WHERE id IN ({qmarks})", ids)
    return {int(i): json.loads(d) for i, d in cur.fetchall()}

  def add(self, vector: np.ndarray, metadata: dict):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (self.dim,):
      raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
    self._ensure_capacity(1)
    vid = self._next_id
    self._next_id += 1

    self.index.add_items(vector.reshape(1, -1), np.array([vid], dtype=int))
    cur = self.conn.cursor()
    cur.execute(
      "INSERT INTO metadata (id, data) VALUES (?, ?)", (vid, json.dumps(metadata))
    )
    self.conn.commit()
    return vid

  def add_many(
    self, vectors: np.ndarray | Sequence[Sequence[float]], metadatas: Sequence[dict]
  ) -> List[int]:
    """
    Batch insert. Faster for both HNSW and SQLite.
    Returns the assigned ids in order.
    """
    V = np.asarray(vectors, dtype=np.float32)
    if V.ndim != 2 or V.shape[1] != self.dim:
      raise ValueError(f"Expected vectors of shape (n, {self.dim}), got {V.shape}")
    if len(metadatas) != V.shape[0]:
      raise ValueError(
        f"metadatas length {len(metadatas)} != number of vectors {V.shape[0]}"
      )

    n = V.shape[0]
    self._ensure_capacity(n)
    ids = np.arange(self._next_id, self._next_id + n, dtype=int)
    self._next_id += n

    # Add to HNSW in one shot
    self.index.add_items(V, ids)

    # Batch insert metadata
    cur = self.conn.cursor()
    cur.executemany(
      "INSERT INTO metadata (id, data) VALUES (?, ?)",
      [(int(i), json.dumps(md)) for i, md in zip(ids, metadatas)],
    )
    self.conn.commit()
    return [int(i) for i in ids]

  def query(self, vector: np.ndarray, k: int = 5, ef: int | None = None):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (self.dim,):
      raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")

    # guard: empty index
    current = self.index.get_current_count()
    if current == 0:
      raise RuntimeError("Index is empty. Add vectors before querying.")

    # cap k to available elements
    k = min(max(1, k), current)

    # optional per-call ef override (higher => better recall, slower)
    if ef is not None:
      self.index.set_ef(int(ef))

    labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k)
    ids = [int(x) for x in labels[0] if x != -1]
    md_map = self._fetch_metadata_bulk(ids)

    results = []
    for lbl, dist in zip(labels[0], distances[0]):
      if lbl == -1:
        continue
      sim = 1.0 - dist  # cosine similarity
      md = md_map.get(int(lbl))

      results.append({"id": int(lbl), "metadata": md, "score": float(sim)})
    return results

  def save(self):
    """
    Manually persist the HNSW index to disk.
    Metadata is auto-committed on each add.
    """
    self.index.save_index(self.index_path)
    # atomic save: write to tmp and replace
    tmp = self.index_path + ".tmp"
    self.index.save_index(tmp)
    os.replace(tmp, self.index_path)

  def _ensure_capacity(self, to_add: int) -> None:
    """
    Ensure the HNSW index has room for `to_add` new elements.
    Grows geometrically to reduce resize churn.
    """
    current = self.index.get_current_count()
    maxel = self.index.get_max_elements()
    need = current + to_add
    if need <= maxel:
      return
    # geometric growth: next power-of-two-ish >= need
    new_cap = max(need, maxel * 2 if maxel > 0 else need)
    self.index.resize_index(new_cap)
    # keep our view in sync for reference
    self.max_elements = new_cap

  def close(self):
    """
    Save index and close SQLite connection.
    """
    self.save()
    self.conn.close()
