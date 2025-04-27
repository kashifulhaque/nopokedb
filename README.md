# **NoPokeDB**

Disk-based HNSW vector DB with SQLite metadata

## Installation

```shell
pip install nopokedb
```

## Usage

```py
import numpy as np
from nopokedb import NoPokeDB

# Initialize (or load existing) store
db = NoPokeDB(dim=128, max_elements=10000, path="./vdb_data")

# Prepare dummy vectors (128-dim)
my_vector = np.random.rand(128).astype(np.float32)
query_vector = np.random.rand(128).astype(np.float32)

# Add to the database
db.add(my_vector, metadata={"name": "foo"})

# Query for nearest neighbors
results = db.query(query_vector, k=5)
for hit in results:
  print(hit)

# Persist & close
db.close()
```
