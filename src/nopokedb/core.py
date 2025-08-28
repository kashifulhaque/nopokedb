import io
import os
import json
import sqlite3
import hnswlib
import threading
import numpy as np
from typing import Dict, List, Sequence, Iterable


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
    self.oplog_path = os.path.join(self.path, "oplog.jsonl")
    self._lock = threading.RLock()

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

    ### attempt to replay any pending ops (idempotent)
    self._replay_oplog()

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
    with self._lock:
      vector = np.asarray(vector, dtype=np.float32)
      if vector.shape != (self.dim,):
        raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")
      self._ensure_capacity(1)
      vid = self._next_id
      self._next_id += 1

      ### write-ahead record (fsync)
      self._oplog_append(
        {"t": "add", "id": int(vid), "vec": vector.tolist(), "md": metadata}
      )

      ### do the work
      self.index.add_items(vector.reshape(1, -1), np.array([vid], dtype=np.int32))
      self._insert_metadata(vid, metadata)

      ### clear oplog on success (truncate)
      self._oplog_clear()
      return vid

  def add_many(
    self, vectors: np.ndarray | Sequence[Sequence[float]], metadatas: Sequence[dict]
  ) -> List[int]:
    with self._lock:
      n = vectors.shape[0]
      self._ensure_capacity(n)
      ids = np.arange(self._next_id, self._next_id + n, dtype=int)
      self._next_id += n
      # write N records into the oplog before mutating state
      self._oplog_append_many(
        {"t": "add", "id": int(i), "vec": vectors[j].tolist(), "md": metadatas[j]}
        for j, i in enumerate(ids)
      )
      # HNSW first, then metadata
      self.index.add_items(vectors, ids.astype(np.int32, copy=False))
      cur = self.conn.cursor()
      cur.executemany(
        "INSERT INTO metadata (id, data) VALUES (?, ?)",
        [(int(i), json.dumps(md)) for i, md in zip(ids, metadatas)],
      )
      self.conn.commit()
      self._oplog_clear()
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

    # be resilient to deleted nodes / small ef: retry with larger ef once, then shrink k
    tried_big_ef = False
    while True:
      try:
        labels, distances = self.index.knn_query(vector.reshape(1, -1), k=k)
        break
      except RuntimeError as e:
        msg = str(e).lower()
        if ("contiguous" in msg or "ef" in msg) and not tried_big_ef:
          # first retry: boost ef a lot
          new_ef = max(getattr(self.index, "ef", 50), k * 8, 200)
          self.index.set_ef(int(new_ef))
          tried_big_ef = True
          continue
        # final fallback: reduce k if still failing
        if k > 1:
          k -= 1
          continue
        raise
    ids = [int(x) for x in labels[0] if x != -1]
    md_map = self._fetch_metadata_bulk(ids)

    results = []
    for lbl, dist in zip(labels[0], distances[0]):
      if lbl == -1:
        continue

      # score: higher is better across all spaces; always return raw distance too
      d = float(dist)
      if self.space == "cosine":
        score = 1.0 - d
      else:
        # for l2/ip, hnswlib distances are "smaller is better"; flip sign for a monotone score
        score = -d

      md = md_map.get(int(lbl))

      results.append(
        {"id": int(lbl), "metadata": md, "score": float(score), "distance": float(d)}
      )
    return results

  # ---- CRUD-ish helpers ----
  def get(self, vid: int) -> dict | None:
    """
    Return stored metadata for id or None if not present.
    """
    cur = self.conn.cursor()
    cur.execute("SELECT data FROM metadata WHERE id = ?", (int(vid),))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None

  def delete(self, vid: int) -> bool:
    """
    Delete an item:
      - mark deleted in HNSW (ignored if not present)
      - remove row from SQLite (returns True if something was removed)
    """
    with self._lock:
      # mark_deleted is idempotent; ignore errors if id isn't present
      try:
        self.index.mark_deleted(int(vid))
      except Exception:
        pass
      cur = self.conn.cursor()
      cur.execute("DELETE FROM metadata WHERE id = ?", (int(vid),))
      self.conn.commit()
      return cur.rowcount > 0

  def upsert(
    self, vid: int, vector: np.ndarray | None = None, metadata: dict | None = None
  ) -> None:
    """
    Upsert semantics:
      - If metadata is provided: INSERT OR REPLACE metadata row for vid.
      - If vector is provided:
           if vid exists in index, mark_deleted then re-add same label with new vector
           else add new point with this label.
        Capacity will be ensured if this becomes a new insertion.
    """
    with self._lock:
      if metadata is not None:
        cur = self.conn.cursor()
        cur.execute(
          "INSERT INTO metadata (id, data) VALUES (?, ?) "
          "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
          (int(vid), json.dumps(metadata)),
        )
        self.conn.commit()

      if vector is not None:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self.dim,):
          raise ValueError(f"Expected vector of shape ({self.dim},), got {v.shape}")
        # If vid is currently active, mark it deleted so we can reuse the label.
        try:
          self.index.mark_deleted(int(vid))
        except Exception:
          pass
        # ensure capacity if adding a brand-new label (safe even if reusing)
        self._ensure_capacity(1)
        self.index.add_items(v.reshape(1, -1), np.array([int(vid)], dtype=np.int32))

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

  # ---- Oplog & Replay ----
  def _oplog_append(self, rec: dict) -> None:
    """
    Append one JSONL record and fsync to ensure durability.
    """
    os.makedirs(self.path, exist_ok=True)
    with open(self.oplog_path, "ab", buffering=0) as f:
      line = (json.dumps(rec) + "\n").encode("utf-8")
      f.write(line)
      f.flush()
      os.fsync(f.fileno())

  def _oplog_append_many(self, records: Iterable[dict]) -> None:
    """
    Append many records with a single fsync.
    """
    os.makedirs(self.path, exist_ok=True)
    buf = io.BytesIO()
    for rec in records:
      buf.write((json.dumps(rec) + "\n").encode("utf-8"))
    data = buf.getvalue()
    with open(self.oplog_path, "ab", buffering=0) as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())

  def _oplog_clear(self) -> None:
    """
    Atomically truncate the oplog after successful commits.
    """
    # write empty tmp and replace to avoid partial truncation
    tmp = self.oplog_path + ".tmp"
    with open(tmp, "wb") as f:
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, self.oplog_path)

  def _replay_oplog(self) -> None:
    if not os.path.exists(self.oplog_path):
      return
    # read all lines first
    try:
      with open(self.oplog_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
      return
    if not lines:
      return
    # build a quick set of ids present in index, to allow idempotent replays
    try:
      present_ids = set(map(int, self.index.get_ids_list()))
    except Exception:
      present_ids = set()

    self._ensure_capacity(len(lines))
    for ln in lines:
      try:
        op = json.loads(ln)
      except Exception:
        continue
      if not isinstance(op, dict) or op.get("t") != "add":
        continue
      vid = int(op["id"])
      vec = np.asarray(op["vec"], dtype=np.float32).reshape(1, -1)
      if vec.shape[1] != self.dim:
        # corrupted or mismatched oplog entry -> skip
        continue
      md = op.get("md", {}) if op.get("md", None) is not None else {}
      have_md = self._metadata_exists(vid)
      in_index = vid in present_ids
      # 4 cases:
      # 1) none exist -> add to index + insert metadata
      # 2) index only -> insert metadata
      # 3) metadata only -> add to index
      # 4) both exist -> nothing to do
      if not in_index and not have_md:
        # add to index (strict: raise on failure so tests catch it)
        self.index.add_items(vec, np.array([vid], dtype=np.int32))
        present_ids.add(vid)
        self._insert_metadata(vid, md)
      elif in_index and not have_md:
        self._insert_metadata(vid, md)
      elif (not in_index) and have_md:
        self.index.add_items(vec, np.array([vid], dtype=np.int32))
        present_ids.add(vid)
    # replay finished -> clear oplog
    self._oplog_clear()

  def _metadata_exists(self, vid: int) -> bool:
    cur = self.conn.cursor()
    cur.execute("SELECT 1 FROM metadata WHERE id=?", (vid,))
    return cur.fetchone() is not None

  def _insert_metadata(self, vid: int, metadata: dict) -> None:
    cur = self.conn.cursor()
    cur.execute(
      "INSERT INTO metadata (id, data) VALUES (?, ?)", (vid, json.dumps(metadata))
    )
    self.conn.commit()
