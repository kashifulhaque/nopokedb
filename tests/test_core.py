import numpy as np
import pytest

from math import isclose
from types import MethodType
from nopokedb import NoPokeDB


@pytest.fixture
def db(tmp_path):
  # Creates a temporary folder for each test
  db = NoPokeDB(dim=4, max_elements=10, path=str(tmp_path))
  yield db
  db.close()


def test_add_and_query(db):
  vec = np.array([1, 0, 0, 0], dtype=np.float32)
  metadata = {"name": "test_vector"}

  vid = db.add(vec, metadata)
  results = db.query(vec, k=1)

  assert len(results) == 1
  assert results[0]["id"] == vid
  assert results[0]["metadata"]["name"] == "test_vector"
  assert pytest.approx(results[0]["score"], rel=1e-5) == 1.0


def test_query_no_vectors(db):
  vec = np.array([1, 0, 0, 0], dtype=np.float32)
  with pytest.raises(RuntimeError):
    db.query(vec, k=1)


def test_query_caps_k(db):
  # add two vectors, ask for way-too-large k; should cap and return 2
  v0 = np.array([1, 0, 0, 0], dtype=np.float32)
  v1 = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
  db.add(v0, {"i": 0})
  db.add(v1, {"i": 1})
  res = db.query(v0, k=10)
  assert len(res) == 2
  assert {r["id"] for r in res} == set([0, 1]) or len({r["id"] for r in res}) == 2


def test_query_with_ef_override(db):
  v0 = np.array([1, 0, 0, 0], dtype=np.float32)
  v1 = np.array([0.9, 0.1, 0, 0], dtype=np.float32)
  db.add(v0, {"i": 0})
  db.add(v1, {"i": 1})
  res = db.query(v0, k=2, ef=100)
  assert len(res) == 2


def test_add_many_basic(db):
  # 5 vectors, 4-dim
  V = np.stack(
    [
      np.array([1, 0, 0, 0], np.float32),
      np.array([0, 1, 0, 0], np.float32),
      np.array([0, 0, 1, 0], np.float32),
      np.array([0, 0, 0, 1], np.float32),
      np.array([0.7, 0.7, 0, 0], np.float32),
    ]
  )
  MD = [{"i": i} for i in range(len(V))]
  ids = db.add_many(V, MD)
  assert len(ids) == len(V)
  # query near the first vector
  res = db.query(np.array([1, 0, 0, 0], np.float32), k=3)
  assert len(res) == 3
  # the nearest one should be the id of the first item
  assert any(r["id"] == ids[0] for r in res)
  # metadata present
  assert all("metadata" in r and r["metadata"] is not None for r in res)


def test_auto_resize_growth(tmp_path):
  # start with tiny max_elements to force growth
  db = NoPokeDB(dim=4, max_elements=2, path=str(tmp_path))
  try:
    n = 10
    V = np.random.RandomState(0).randn(n, 4).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    md = [{"i": i} for i in range(n)]
    ids = db.add_many(V, md)
    assert len(ids) == n
    # ensure we can still query top-5
    q = V[0]
    res = db.query(q, k=5)
    assert len(res) == 5
  finally:
    db.close()


def test_invalid_vector_shape(db):
  with pytest.raises(ValueError):
    db.add([1, 2], metadata={})


def test_oplog_replay_both_missing(tmp_path):
  db = NoPokeDB(dim=4, max_elements=10, path=str(tmp_path))
  try:
    v = np.array([1, 0, 0, 0], np.float32)

    # Replace ONLY this instance's add with a version that:
    # - writes oplog
    # - then raises before touching index/sqlite
    def crash_after_oplog(self, vector, metadata):
      vector = np.asarray(vector, np.float32)
      assert vector.shape == (self.dim,)
      with self._lock:
        self._ensure_capacity(1)
        vid = self._next_id
        self._next_id += 1
        # write-ahead intent
        self._oplog_append(
          {"t": "add", "id": int(vid), "vec": vector.tolist(), "md": metadata}
        )
        # boom before index.add_items / _insert_metadata
        raise RuntimeError("simulated crash after oplog")

    db.add = MethodType(crash_after_oplog, db)

    with pytest.raises(RuntimeError):
      db.add(v, {"name": "crashy"})

    # At this point, oplog has the record, index/sqlite are untouched.
    # "Restart" the DB; replay should make it whole.
    db.close()
    db2 = NoPokeDB(dim=4, max_elements=10, path=str(tmp_path))
    try:
      res = db2.query(v, k=1)
      assert len(res) == 1
      assert res[0]["metadata"]["name"] == "crashy"
    finally:
      db2.close()
  finally:
    try:
      db.close()
    except Exception:
      pass


def test_oplog_replay_index_only(tmp_path):
  """
  Simulate a crash after index.add_items succeeds and the index is saved,
  but before metadata INSERT commits. On restart, replay should only insert metadata.
  """
  db = NoPokeDB(dim=4, max_elements=10, path=str(tmp_path))
  try:
    v = np.array([0, 1, 0, 0], np.float32)
    # patch _insert_metadata to save index then raise
    _original_insert = db._insert_metadata

    def save_then_boom(vid, md):
      # persist current index state to disk to simulate successful index write
      db.save()
      raise RuntimeError("simulated crash before metadata commit")

    db._insert_metadata = save_then_boom
    with pytest.raises(RuntimeError):
      db.add(v, {"name": "half"})
    # restart
    db.close()
    db2 = NoPokeDB(dim=4, max_elements=10, path=str(tmp_path))
    try:
      # replay should see id present in index but missing metadata, and insert md
      res = db2.query(v, k=1)
      assert len(res) == 1
      assert res[0]["metadata"]["name"] == "half"
    finally:
      db2.close()
  finally:
    try:
      db.close()
    except Exception:
      pass


def test_multi_metric_cosine_normalization(tmp_path):
  db = NoPokeDB(dim=3, max_elements=10, path=str(tmp_path), space="cosine")
  try:
    a = np.array([1.0, 0.0, 0.0], np.float32) * 10  # scaled
    b = np.array([0.0, 1.0, 0.0], np.float32)
    db.add(a, {"name": "a"})
    db.add(b, {"name": "b"})
    # query with differently scaled vector should still hit 'a' first due to normalization
    q = np.array([2.0, 0.0, 0.0], np.float32)
    res = db.query(q, k=2)
    assert res[0]["metadata"]["name"] == "a"
    # cosine score of identical dirs ~ 1.0
    assert isclose(res[0]["score"], 1.0, rel_tol=1e-5)
  finally:
    db.close()


def test_multi_metric_l2(tmp_path):
  db = NoPokeDB(dim=2, max_elements=10, path=str(tmp_path), space="l2")
  try:
    db.add(np.array([0.0, 0.0], np.float32), {"id": "o"})
    db.add(np.array([1.0, 0.0], np.float32), {"id": "x"})
    res = db.query(np.array([0.1, 0.0], np.float32), k=2)
    # closer to origin than to (1,0)
    assert res[0]["metadata"]["id"] == "o"
    # for non-cosine spaces, we expose distance and use score=-distance (monotonic)
    assert "distance" in res[0] and isinstance(res[0]["distance"], float)
  finally:
    db.close()


def test_delete_and_get_and_upsert(tmp_path):
  db = NoPokeDB(dim=3, max_elements=10, path=str(tmp_path))
  try:
    v1 = np.array([1, 0, 0], np.float32)
    v2 = np.array([0, 1, 0], np.float32)
    id1 = db.add(v1, {"k": "v1"})
    id2 = db.add(v2, {"k": "v2"})
    # get
    assert db.get(id1)["k"] == "v1"
    # delete id2
    assert db.delete(id2) is True
    assert db.get(id2) is None
    # ensure deleted one doesn't show up
    res = db.query(v1, k=2)
    assert all(r["id"] != id2 for r in res)
    # upsert metadata only
    db.upsert(id1, metadata={"k": "v1b", "x": 1})
    assert db.get(id1)["k"] == "v1b"
    # upsert vector (same id, different direction)
    db.upsert(id1, vector=np.array([0, 0, 1], np.float32))
    # should now be closer to [0,0,1]
    nearer = db.query(np.array([0, 0, 1], np.float32), k=1)[0]
    assert nearer["id"] == id1
  finally:
    db.close()
