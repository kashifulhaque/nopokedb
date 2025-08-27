import numpy as np
import pytest
import random
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
