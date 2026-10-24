"""Bounded, decoded float64 pools for exact scoped vector scans.

NumPy only selects a conservative shortlist. Returned scores are recomputed in
the original Python operation order, including clamping and stable tie order.
There is no approximate index, quantization or model dependency here.
"""
from __future__ import annotations

from array import array
import json
import math
import sys

try:
    import numpy as np
except ImportError:
    np = None


class VectorPool:
    def __init__(self, records, *, backend="auto"):
        self.ids = []
        self.groups = []
        self.values = array("d")
        self.norms = array("d")
        self.owners = array("q")
        self.dimension = None
        by_id = {}
        for memory_id, encoded in records:
            vector = json.loads(encoded)
            if self.dimension is None:
                self.dimension = len(vector)
            if len(vector) != self.dimension:
                raise ValueError("Embedding dimension does not match stored vectors")
            owner = by_id.get(memory_id)
            if owner is None:
                owner = len(self.ids)
                by_id[memory_id] = owner
                self.ids.append(memory_id)
                self.groups.append([])
            self.groups[owner].append(len(self.norms))
            self.values.extend(vector)
            self.norms.append(sum(x*x for x in vector))
            self.owners.append(owner)
        self.nbytes = (len(self.values)*self.values.itemsize
                       + len(self.norms)*self.norms.itemsize
                       + len(self.owners)*self.owners.itemsize)
        self.matrix = None
        self.finite = all(math.isfinite(x) for x in self.norms)
        if np is not None and backend != "python" and self.dimension:
            # frombuffer views the compact array; it does not make another copy.
            self.matrix = np.frombuffer(self.values, dtype=np.float64).reshape(-1, self.dimension)
        self.backend = "numpy" if self.matrix is not None else "python"

    def rank(self, query, limit):
        if not self.ids:
            return []
        if len(query) != self.dimension:
            raise ValueError("Embedding dimension does not match stored vectors")
        count = min(limit if limit >= 0 else len(self.ids)+limit, len(self.ids))
        if count <= 0:
            return []
        query_norm = sum(x*x for x in query)
        selected = range(len(self.ids))
        if (self.matrix is not None and self.finite and math.isfinite(query_norm)
                and all(math.isfinite(query_norm*n) for n in self.norms)):
            norms = np.frombuffer(self.norms, dtype=np.float64)
            denominator = np.sqrt(norms * query_norm)
            scores = np.zeros(len(norms), dtype=np.float64)
            np.divide(self.matrix @ np.asarray(query, dtype=np.float64), denominator,
                      out=scores, where=denominator != 0)
            np.clip(scores, 0., 1., out=scores)
            owner_scores = np.zeros(len(self.ids), dtype=np.float64)
            np.maximum.at(owner_scores, np.frombuffer(self.owners, dtype=np.int64), scores)
            cutoff = np.partition(owner_scores, len(self.ids)-count)[len(self.ids)-count]
            # Include numerical neighbours and ties before exact rescoring.
            # The margin exceeds the float64 dot-product roundoff bound for
            # ordinary finite embeddings; high-dimensional extremes fall back.
            guard = max(1e-12, 16*(self.dimension+4)*sys.float_info.epsilon)
            if guard < 1e-3:
                selected = np.flatnonzero(owner_scores >= cutoff-guard).tolist()

        def exact_score(index):
            start = index*self.dimension
            vector = self.values[start:start+self.dimension]
            denominator = math.sqrt(query_norm*self.norms[index])
            return (max(0., min(1., sum(x*y for x, y in zip(query, vector))/denominator))
                    if denominator else 0.)

        ranked = [(self.ids[i], max(exact_score(j) for j in self.groups[i])) for i in selected]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[:count]
