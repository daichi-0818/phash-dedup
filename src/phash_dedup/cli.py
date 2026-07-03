"""Command-line interface: scan a folder of images and print duplicate clusters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cluster import cluster_by_threshold, cluster_by_threshold_matrix
from .hashing import bits_for_hash_size, image_to_phash

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def find_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def pick_representative(paths: list[Path]) -> Path:
    """Pick a representative image from a duplicate cluster.

    Strategy: choose the largest file by size on disk (a reasonable,
    dependency-free proxy for "highest quality" without decoding every
    image again). Ties broken by earliest path.
    """
    return max(paths, key=lambda p: (p.stat().st_size, ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phash-dedup",
        description="Find near-duplicate images in a folder using perceptual hashing.",
    )
    parser.add_argument("folder", type=Path, help="Folder to scan for images (recursive).")
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Maximum Hamming distance for two images to be considered duplicates (default: 5).",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="Side length of the pHash block; hash bit count is hash_size^2 (default: 8).",
    )
    parser.add_argument(
        "--use-matrix",
        action="store_true",
        help="Use the numpy-vectorized pairwise distance matrix instead of the "
        "pure-Python union-find loop (faster for large image sets).",
    )
    parser.add_argument(
        "--only-duplicates",
        action="store_true",
        help="Only print clusters with more than one image.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 1

    images = find_images(folder)
    if not images:
        print(f"No supported images found under {folder}")
        return 0

    hashes: list[int] = []
    valid_images: list[Path] = []
    for path in images:
        try:
            hashes.append(image_to_phash(path, hash_size=args.hash_size))
            valid_images.append(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"warning: skipping {path} ({exc})", file=sys.stderr)

    if not valid_images:
        print("No images could be hashed.")
        return 0

    bits = bits_for_hash_size(args.hash_size)
    if args.use_matrix:
        clusters = cluster_by_threshold_matrix(hashes, threshold=args.threshold, bits=bits)
    else:
        clusters = cluster_by_threshold(hashes, threshold=args.threshold)

    printed = 0
    for cluster_idx, members in enumerate(clusters, start=1):
        if args.only_duplicates and len(members) <= 1:
            continue
        printed += 1
        member_paths = [valid_images[i] for i in members]
        representative = pick_representative(member_paths)
        print(f"Cluster {cluster_idx} ({len(members)} image(s)):")
        for p in member_paths:
            marker = "*" if p == representative else " "
            print(f"  [{marker}] {p}")
    if printed == 0:
        print("No duplicate clusters found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
