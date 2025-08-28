#!/usr/bin/env python3
"""
Simple ANN benchmarks: NoPokeDB vs FAISS vs Chroma vs (optional) Qdrant.

Metrics:
- build: vectors/sec
- query latency: p50 / p95 (ms)
- recall@k: vs exact neighbors from brute-force (sklearn)
"""

import time
import json
import argparse
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np


# --- ground truth (exact) ---
def brute_force_topk(
  X: np.ndarray, q: np.ndarray, k: int, metric: str
) -> Tuple[np.ndarray, np.ndarray]:
  if metric == "cosine":
    # cosine distance = 1 - cosine similarity
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q, keepdims=True) + 1e-12)
    sims = Xn @ qn
    order = np.argsort(-sims)[:k]
    d = 1.0 - sims[order]
    return order, d
  elif metric == "l2":
    d2 = np.sum((X - q) ** 2, axis=1)
    order = np.argsort(d2)[:k]
    return order, d2[order]
  elif metric == "ip":
    sims = X @ q
    order = np.argsort(-sims)[:k]
    # map to a distance-like value for consistency
    d = -sims[order]
    return order, d
  else:
    raise ValueError("metric must be one of {cosine,l2,ip}")


# --- results container ---
@dataclass
class BenchResult:
  backend: str
  n: int
  dim: int
  k: int
  metric: str
  build_vectors_per_s: float
  qps: float
  p50_ms: float
  p95_ms: float
  recall_at_k: float


# --- backends ---
class BackendBase:
  def build(self, X: np.ndarray, metric: str) -> None: ...
  def query(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]: ...


# NoPokeDB (local, file-backed)
class BackendNoPokeDB(BackendBase):
  def __init__(self, path="./.bench_nopokedb", ef=100):
    from nopokedb import NoPokeDB  # type: ignore

    self.NoPokeDB = NoPokeDB
    self.path = path
    self.ef = ef
    self.db = None

  def build(self, X: np.ndarray, metric: str) -> None:
    dim = X.shape[1]
    self.db = self.NoPokeDB(
      dim=dim, max_elements=X.shape[0], path=self.path, space=metric
    )
    # batch add
    metas = [{} for _ in range(X.shape[0])]
    self.db.add_many(X.astype(np.float32), metas)

  def query(self, q: np.ndarray, k: int):
    assert self.db is not None
    res = self.db.query(q.astype(np.float32), k=k, ef=self.ef)
    # return ids, distances (convert from result objects)
    ids = np.array([r["id"] for r in res], dtype=np.int64)
    # if cosine: distance already stored; otherwise distance present too
    d = np.array([r["distance"] for r in res], dtype=np.float32)
    return ids, d


# FAISS (in-memory)
class BackendFaiss(BackendBase):
  def __init__(self):
    import faiss  # type: ignore

    self.faiss = faiss
    self.index = None
    self.metric_map = {
      "l2": faiss.METRIC_L2,
      "ip": faiss.METRIC_INNER_PRODUCT,
      "cosine": faiss.METRIC_INNER_PRODUCT,  # normalize + IP
    }

  def build(self, X: np.ndarray, metric: str) -> None:
    faiss = self.faiss
    X = X.astype(np.float32)
    if metric == "cosine":
      X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    dim = X.shape[1]
    quantizer = faiss.IndexFlat(
      dim, self.metric_map["ip" if metric == "cosine" else metric]
    )
    # HNSW is not always compiled; use HNSWFlat if available, else IVFFlat or IndexFlat as fallback
    try:
      index = faiss.IndexHNSWFlat(
        dim, 32, self.metric_map["ip" if metric == "cosine" else metric]
      )
    except Exception:
      index = quantizer
    if metric == "ip":
      # nothing extra
      pass
    self.index = index
    self.index.add(X)

  def query(self, q: np.ndarray, k: int):
    faiss = self.faiss
    q = q.astype(np.float32)
    if isinstance(self.index, faiss.IndexHNSWFlat) or isinstance(
      self.index, faiss.IndexFlat
    ):
      # cosine uses normalized + IP
      # if cosine, normalize query
      # we can’t easily inspect metric; normalize anyway if norm>0 and let IP benefit
      nq = q / (np.linalg.norm(q, keepdims=True) + 1e-12)
      qv = nq
    else:
      qv = q
    D, I = self.index.search(qv.reshape(1, -1), k)  # noqa: E741
    # For cosine we searched with IP over normalized vectors, distance = 1 - sim
    sims = D[0]
    # guess if cosine by checking if sims in [-1,1] (not perfect but fine for our local search)
    if np.all(sims <= 1.0001) and np.all(sims >= -1.0001):
      dist = 1.0 - sims
    else:
      dist = sims  # for L2 it’s distances; for IP we return negative-sim-like numbers
    return I[0], dist


# --- runner ---
def run_once(
  backend: BackendBase, X: np.ndarray, Q: np.ndarray, k: int, metric: str
) -> BenchResult:
  # build
  t0 = time.perf_counter()
  backend.build(X, metric)
  t1 = time.perf_counter()
  build_vps = X.shape[0] / (t1 - t0 + 1e-9)

  # queries
  lat_ms: List[float] = []
  correct = 0
  total = Q.shape[0] * k

  for i in range(Q.shape[0]):
    q = Q[i]
    gt_ids, _ = brute_force_topk(X, q, k, metric)
    s0 = time.perf_counter()
    ids, _ = backend.query(q, k)
    s1 = time.perf_counter()
    lat_ms.append((s1 - s0) * 1000.0)
    # recall@k: intersection size / k
    correct += len(set(map(int, ids)).intersection(set(map(int, gt_ids))))

  qps = Q.shape[0] / (sum(lat_ms) / 1000.0 + 1e-9)
  p50 = np.percentile(lat_ms, 50.0)
  p95 = np.percentile(lat_ms, 95.0)
  recall = correct / float(total)

  return BenchResult(
    backend=backend.__class__.__name__.replace("Backend", "").lower(),
    n=X.shape[0],
    dim=X.shape[1],
    k=k,
    metric=metric,
    build_vectors_per_s=build_vps,
    qps=qps,
    p50_ms=p50,
    p95_ms=p95,
    recall_at_k=recall,
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--backend", required=True, choices=["nopokedb", "faiss", "chroma", "qdrant"]
  )
  ap.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"])
  ap.add_argument("--n", type=int, default=10000)
  ap.add_argument("--q", type=int, default=100)
  ap.add_argument("--dim", type=int, default=128)
  ap.add_argument("--k", type=int, default=10)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument(
    "--normalize",
    action="store_true",
    help="Pre-normalize dataset for cosine/IP (some backends do this internally)",
  )
  args = ap.parse_args()

  rng = np.random.default_rng(args.seed)
  X = rng.normal(size=(args.n, args.dim)).astype(np.float32)
  if args.normalize or args.metric == "cosine":
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
  Q = rng.normal(size=(args.q, args.dim)).astype(np.float32)
  if args.normalize or args.metric == "cosine":
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12

  if args.backend == "nopokedb":
    backend = BackendNoPokeDB()
  elif args.backend == "faiss":
    backend = BackendFaiss()
  else:
    raise ValueError("unknown backend")

  res = run_once(backend, X, Q, args.k, args.metric)
  print(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
  main()
