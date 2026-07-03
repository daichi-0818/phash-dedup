# phash-dedup

Perceptual-hash (pHash) based near-duplicate image clustering, implemented
from scratch with Pillow and numpy — no external phash library dependency.

Point it at a folder of images and it groups near-duplicates (resized
copies, re-compressed copies, minor crops/edits) into clusters, picking a
representative image from each cluster.

## How it works

1. **Hashing** (`phash_dedup.hashing`): each image is grayscaled, resized to
   a small fixed square, and run through a from-scratch 2D DCT
   (Discrete Cosine Transform, implemented via cosine-basis matrix
   multiplication — no scipy needed). The low-frequency 8x8 block of DCT
   coefficients (minus the DC term) is thresholded against its median to
   produce a 64-bit hash. Perceptually similar images produce hashes that
   differ by only a few bits, even under resizing, recompression, or minor
   color adjustments.

2. **Clustering** (`phash_dedup.cluster`): hashes are compared pairwise by
   Hamming distance (number of differing bits). Any pair within a
   configurable threshold is connected by an edge; a Union-Find
   (disjoint-set) structure collapses these edges into connected
   components, i.e. duplicate clusters. Two implementations are provided:
   - A pure-Python O(n^2) double loop + union-find (`cluster_by_threshold`),
     dependency-free and easy to test.
   - A numpy-vectorized pairwise Hamming distance matrix
     (`pairwise_hamming_matrix` / `cluster_by_threshold_matrix`), useful
     when the number of images is large enough that the pure-Python loop
     becomes the bottleneck.

3. **Representative selection** (`phash_dedup.cli`): within each cluster,
   the CLI currently picks the largest file on disk as a simple proxy for
   "highest quality" copy, without needing to decode every image again.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.9, Pillow, and numpy.

## CLI usage

```bash
phash-dedup ./my_photos
phash-dedup ./my_photos --threshold 8 --only-duplicates
phash-dedup ./my_photos --hash-size 16 --use-matrix
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--threshold` | `5` | Max Hamming distance to consider two images duplicates |
| `--hash-size` | `8` | Side length of the retained DCT block (bits = hash_size^2) |
| `--use-matrix` | off | Use the numpy-vectorized distance matrix instead of the pure-Python loop |
| `--only-duplicates` | off | Only print clusters with more than one image |

Example output:

```
Cluster 1 (3 image(s)):
  [ ] my_photos/img_001.jpg
  [*] my_photos/img_001_large.jpg
  [ ] my_photos/img_001_copy.png
Cluster 2 (1 image(s)):
  [*] my_photos/unique.jpg
```

The `[*]` marks the representative image chosen for that cluster.

## Library usage

```python
from phash_dedup.hashing import image_to_phash
from phash_dedup.cluster import cluster_by_threshold

hashes = [image_to_phash(p) for p in image_paths]
clusters = cluster_by_threshold(hashes, threshold=5)
# clusters: list[list[int]] — each inner list is indices into image_paths
```

See `examples/README.md` for more.

## Tests

```bash
pytest tests/
```

`tests/test_cluster.py` exercises Hamming distance, Union-Find, and
threshold clustering by feeding hand-crafted integer hash values directly —
no images, Pillow, or numpy required for the core tests. The numpy-backed
matrix clustering test is skipped automatically if numpy isn't installed.

## Project layout

```
phash-dedup/
  src/phash_dedup/
    __init__.py
    hashing.py     # image_to_phash(path) - DCT-based pHash
    cluster.py      # hamming_distance, UnionFind, cluster_by_threshold
    cli.py           # folder -> duplicate cluster report
  tests/test_cluster.py
  examples/README.md
  LICENSE
  pyproject.toml
```

## License

MIT © 2026 daichi-0818
