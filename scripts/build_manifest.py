"""Thin legacy entry point for building a digest-locked MNIST/100 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mnist_kaiwu.artifacts import atomic_write_json
from examples.mnist_kaiwu.data import build_mnist100_manifest


def main(argv: list[str] | None = None) -> int:
    """Build the manifest without starting training."""
    parser = argparse.ArgumentParser(description="Freeze MNIST/100 manifest from local IDX files.")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "evidence" / "mnist100-manifest.json"))
    parser.add_argument("--download-if-missing", action="store_true")
    args = parser.parse_args(argv)
    manifest = build_mnist100_manifest(args.data_root, download_if_missing=args.download_if_missing)
    dest = Path(args.output)
    atomic_write_json(dest, manifest.to_canonical_dict())
    print("wrote", dest)
    print("manifest_sha256", manifest.manifest_sha256)
    print("train", len(manifest.train_indices), "test", len(manifest.test_indices))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
