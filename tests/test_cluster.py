"""Tests for hamming_distance / UnionFind / cluster_by_threshold.

These tests feed hand-crafted integer hash values directly, so they run
without Pillow or numpy installed. The optional matrix-based clustering
test is skipped automatically if numpy is unavailable.
"""

from __future__ import annotations

import pytest

from phash_dedup.cluster import (
    UnionFind,
    cluster_by_threshold,
    hamming_distance,
)


def test_hamming_distance_identical():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_all_bits_differ():
    assert hamming_distance(0b0000, 0b1111) == 4


def test_hamming_distance_partial():
    assert hamming_distance(0b1100, 0b1010) == 2


def test_union_find_basic_union_and_find():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.find(3) != uf.find(0)


def test_union_find_groups():
    uf = UnionFind(4)
    uf.union(0, 1)
    groups = uf.groups()
    sizes = sorted(len(members) for members in groups.values())
    assert sizes == [1, 1, 2]


def test_cluster_by_threshold_two_near_duplicates_and_one_outlier():
    # h0 and h1 differ by 1 bit -> near-duplicate under threshold=5.
    # h2 differs from both by many bits -> distinct cluster.
    h0 = 0b0000_0000
    h1 = 0b0000_0001
    h2 = 0b1111_1111
    clusters = cluster_by_threshold([h0, h1, h2], threshold=5)

    # Expect one cluster of size 2 (h0, h1) and one singleton (h2).
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]

    pair_cluster = next(c for c in clusters if len(c) == 2)
    assert set(pair_cluster) == {0, 1}


def test_cluster_by_threshold_all_identical_forms_one_cluster():
    hashes = [42, 42, 42, 42]
    clusters = cluster_by_threshold(hashes, threshold=0)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2, 3]


def test_cluster_by_threshold_zero_threshold_separates_close_but_nonidentical():
    h0 = 0b0000
    h1 = 0b0001  # 1 bit away
    clusters = cluster_by_threshold([h0, h1], threshold=0)
    assert len(clusters) == 2


def test_cluster_by_threshold_empty_input():
    assert cluster_by_threshold([], threshold=5) == []


def test_cluster_by_threshold_single_input():
    clusters = cluster_by_threshold([123], threshold=5)
    assert clusters == [[0]]


def test_cluster_by_threshold_transitive_chain():
    # h0-h1 close, h1-h2 close, but h0-h2 not directly close enough
    # on their own would still merge transitively via union-find.
    h0 = 0b0000_0000
    h1 = 0b0000_0011  # distance 2 from h0
    h2 = 0b0000_1111  # distance 2 from h1, distance 4 from h0
    clusters = cluster_by_threshold([h0, h1, h2], threshold=2)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == [0, 1, 2]


def test_cluster_by_threshold_matrix_matches_union_find():
    numpy = pytest.importorskip("numpy")
    del numpy
    from phash_dedup.cluster import cluster_by_threshold_matrix

    hashes = [0b0000_0000, 0b0000_0001, 0b1111_1111, 0b1111_1110]
    expected = cluster_by_threshold(hashes, threshold=1)
    actual = cluster_by_threshold_matrix(hashes, threshold=1, bits=8)
    assert actual == expected
