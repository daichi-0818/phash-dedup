"""Clustering of perceptual hashes by Hamming distance.

Two strategies are provided:

- ``UnionFind`` + ``cluster_by_threshold``: build a graph where an edge
  connects any pair of hashes with Hamming distance <= threshold, then
  union-find the connected components. Works with plain Python ints and
  has no numpy dependency, so it also serves as the reference
  implementation used in tests.
- ``pairwise_hamming_matrix``: a numpy-vectorized pairwise Hamming
  distance matrix, useful when the number of hashes is large enough
  that an O(n^2) pure-Python double loop becomes a bottleneck.
"""

from __future__ import annotations

from typing import Sequence


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integer hashes."""
    return bin(a ^ b).count("1")


class UnionFind:
    """Disjoint-set (union-find) with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        """Map each root index to the list of member indices."""
        result: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            root = self.find(i)
            result.setdefault(root, []).append(i)
        return result


def cluster_by_threshold(
    hashes: Sequence[int], threshold: int = 5
) -> list[list[int]]:
    """Cluster hashes into connected components by Hamming-distance threshold.

    Any two hashes whose Hamming distance is <= ``threshold`` are
    connected by an edge; connected components (via union-find) become
    clusters. Singletons (no neighbor within threshold) form their own
    single-item cluster.

    Args:
        hashes: sequence of integer perceptual hashes.
        threshold: maximum Hamming distance for two hashes to be
            considered near-duplicates.

    Returns:
        A list of clusters, each a list of indices into ``hashes``,
        sorted by cluster size descending then by first index.
    """
    n = len(hashes)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                uf.union(i, j)

    clusters = list(uf.groups().values())
    clusters.sort(key=lambda members: (-len(members), members[0]))
    return clusters


def pairwise_hamming_matrix(hashes: Sequence[int], bits: int = 64):
    """Vectorized pairwise Hamming distance matrix using numpy.

    Converts each hash into a bit-vector of length ``bits`` and uses
    matrix operations to compute all pairwise Hamming distances at
    once. This is much faster than the pure-Python double loop in
    :func:`cluster_by_threshold` when the number of hashes is large
    (thousands+), at the cost of an O(n^2) memory allocation.

    Args:
        hashes: sequence of integer perceptual hashes.
        bits: bit-width of each hash (must be large enough to hold
            the largest value in ``hashes``).

    Returns:
        An (n, n) numpy integer array of pairwise Hamming distances.
    """
    import numpy as np

    n = len(hashes)
    bit_matrix = np.zeros((n, bits), dtype=np.uint8)
    for row, value in enumerate(hashes):
        for bit in range(bits):
            bit_matrix[row, bit] = (value >> bit) & 1

    # XOR every pair of bit-vectors and sum differing bits.
    # (n, 1, bits) XOR (1, n, bits) -> (n, n, bits) summed over bits.
    diff = bit_matrix[:, None, :] != bit_matrix[None, :, :]
    return diff.sum(axis=2)


def cluster_by_threshold_matrix(
    hashes: Sequence[int], threshold: int = 5, bits: int = 64
) -> list[list[int]]:
    """Same clustering result as :func:`cluster_by_threshold`, but uses the
    numpy-vectorized distance matrix from :func:`pairwise_hamming_matrix`
    to build the edge list, which is faster for large inputs.
    """
    n = len(hashes)
    uf = UnionFind(n)

    if n > 1:
        distances = pairwise_hamming_matrix(hashes, bits=bits)
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= threshold:
                    uf.union(i, j)

    clusters = list(uf.groups().values())
    clusters.sort(key=lambda members: (-len(members), members[0]))
    return clusters
