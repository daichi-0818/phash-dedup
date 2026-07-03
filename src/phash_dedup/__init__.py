"""phash-dedup: perceptual-hash based near-duplicate image clustering."""

from .cluster import UnionFind, cluster_by_threshold, hamming_distance
from .hashing import image_to_phash

__all__ = [
    "UnionFind",
    "cluster_by_threshold",
    "hamming_distance",
    "image_to_phash",
]

__version__ = "0.1.0"
