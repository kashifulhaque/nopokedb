# 🐹 NoPokeDB

A **tiny, disk-backed vector database** built on [hnswlib](https://github.com/nmslib/hnswlib) (for approximate nearest neighbor search) and SQLite (for metadata).  
Simple, lightweight, and crash-safe — good for prototypes, side projects, or when you don’t want a heavyweight vector DB service.


## ✨ Features

- **Durable**: HNSW index + SQLite metadata, persisted on disk  
- **Crash safety**: write-ahead oplog with replay  
- **Fast**: batch inserts, bulk metadata fetch, auto-resize index  
- **Flexible metrics**: cosine / L2 / inner product  
- **CRUD**: `add`, `add_many`, `get`, `delete`, `upsert`  
- **Consistent scoring**: higher = better for all metrics (with raw `distance` included)  

## 📦 Installation

```bash
pip install nopokedb
```

## 🚀 Quickstart

```python
import numpy as np
from nopokedb import NoPokeDB

# Create (or load existing) DB with 128-d vectors
db = NoPokeDB(dim=128, max_elements=10_000, path="./vdb_data", space="cosine")

# Insert one vector
vec = np.random.rand(128).astype(np.float32)
vid = db.add(vec, metadata={"name": "foo"})

# Insert many at once (faster)
V = np.random.rand(5, 128).astype(np.float32)
metas = [{"i": i} for i in range(len(V))]
ids = db.add_many(V, metas)

# Query nearest neighbors
q = np.random.rand(128).astype(np.float32)
hits = db.query(q, k=3)
for h in hits:
    print(h["id"], h["score"], h["metadata"])

# Get / update / delete
print(db.get(vid))
db.upsert(vid, metadata={"name": "bar"})
db.delete(ids[0])

# Persist & close
db.save()
db.close()
```

Example query result:

```python
{'id': 0, 'metadata': {'name': 'foo'}, 'score': 0.991, 'distance': 0.009}
```


## 🛠 Development

### Bump version

The Makefile automates version bumps in `pyproject.toml` and `src/nopokedb/__init__.py`:

```bash
make bump V=0.2.0
```

That will:

* update both files
* commit with message `Bump to v0.2.0`
* tag `v0.2.0`
* push tag + main branch

### Publish to PyPI

1. Set your API key once:

   ```bash
   export PYPI_API_KEY=your-pypi-token
   ```

2. Build + upload:

   ```bash
   make publish
   ```

## 📜 License

MIT
