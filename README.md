# **NoPokeDB**

Disk-based HNSW vector DB with SQLite metadata

## Installation

```shell
pip install nopokedb
```

## Usage

```py
from nopokedb import NoPokeDB

# Initialize (or load existing) store
db = PersistentVectorDB(dim=128, max_elements=10000, path="./vdb_data")

# Add
db.add(my_vector, metadata={"name": "foo"})

# Query
results = db.query(query_vector, k=5)
for hit in results:
  print(hit)

# Persist & close
db.close()
```
