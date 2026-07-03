# Examples

## Cluster hashes directly (no images needed)

```python
from phash_dedup.cluster import cluster_by_threshold

hashes = [0x0F0F0F0F, 0x0F0F0F0E, 0xFFFFFFFF]
clusters = cluster_by_threshold(hashes, threshold=3)
print(clusters)
# [[0, 1], [2]]
```

## Hash and cluster a folder of real images

```python
from pathlib import Path
from phash_dedup.hashing import image_to_phash
from phash_dedup.cluster import cluster_by_threshold

folder = Path("./my_photos")
paths = sorted(p for p in folder.glob("*.jpg"))
hashes = [image_to_phash(p) for p in paths]

for cluster in cluster_by_threshold(hashes, threshold=5):
    print([str(paths[i]) for i in cluster])
```

## CLI

```bash
phash-dedup ./my_photos --threshold 5 --only-duplicates
```
