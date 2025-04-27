import numpy as np
import pytest
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

def test_invalid_vector_shape(db):
  with pytest.raises(ValueError):
    db.add([1, 2], metadata={})
