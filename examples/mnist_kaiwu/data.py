"""Deterministic MNIST/100 subset with digest-locked manifests."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import struct
from pathlib import Path
import urllib.request

import numpy as np

from .artifacts import SCHEMA_VERSION, canonical_json, canonical_sha256, sha256_file, utc_now
from .failures import DATASET_DIGEST_MISMATCH, AlignmentError

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 10


@dataclass
class Mnist100Split:
    """Loaded MNIST subset split."""

    images: np.ndarray
    labels: np.ndarray
    indices: list[int]


@dataclass(frozen=True)
class FileDigest:
    """File size and SHA-256 digest."""

    size: int
    sha256: str


@dataclass(frozen=True)
class Mnist100Manifest:
    """Frozen description of the deterministic MNIST/100 subset."""

    schema_version: str
    source_urls: dict[str, str]
    files: dict[str, FileDigest]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_class_counts: tuple[int, ...]
    test_class_counts: tuple[int, ...]
    image_shape: tuple[int, int]
    flatten_order: str
    input_dtype: str
    preprocessing: str
    value_range: tuple[float, float]
    selection_rule: str
    created_at_utc: str
    manifest_sha256: str

    def to_canonical_dict(self) -> dict:
        """Return the canonical JSON-compatible manifest payload."""
        return {
            "schema_version": self.schema_version,
            "source_urls": dict(self.source_urls),
            "files": {
                name: {"size": digest.size, "sha256": digest.sha256}
                for name, digest in sorted(self.files.items())
            },
            "train_indices": list(self.train_indices),
            "test_indices": list(self.test_indices),
            "train_class_counts": list(self.train_class_counts),
            "test_class_counts": list(self.test_class_counts),
            "image_shape": list(self.image_shape),
            "flatten_order": self.flatten_order,
            "input_dtype": self.input_dtype,
            "preprocessing": self.preprocessing,
            "value_range": list(self.value_range),
            "selection_rule": self.selection_rule,
            "created_at_utc": self.created_at_utc,
            "manifest_sha256": self.manifest_sha256,
        }

    def to_canonical_json(self) -> str:
        """Serialize this manifest deterministically."""
        return canonical_json(self.to_canonical_dict())

    def compute_manifest_sha256(self) -> str:
        """Compute the digest over all fields except ``manifest_sha256``."""
        payload = dict(self.to_canonical_dict())
        payload.pop("manifest_sha256", None)
        return canonical_sha256(payload)

    def validate(self) -> None:
        """Validate schema, counts, indices and digest."""
        if self.schema_version != SCHEMA_VERSION:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "unsupported MNIST manifest schema")
        if len(self.train_indices) != 10 * TRAIN_PER_CLASS:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "train_indices must contain 1000 entries")
        if len(self.test_indices) != 10 * TEST_PER_CLASS:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "test_indices must contain 100 entries")
        if len(set(self.train_indices)) != len(self.train_indices):
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "train_indices are not unique")
        if len(set(self.test_indices)) != len(self.test_indices):
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "test_indices are not unique")
        if self.train_class_counts != tuple([TRAIN_PER_CLASS] * 10):
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "train class counts must be 100 each")
        if self.test_class_counts != tuple([TEST_PER_CLASS] * 10):
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "test class counts must be 10 each")
        if self.compute_manifest_sha256() != self.manifest_sha256:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "manifest_sha256 does not match contents")


def file_digest(path: Path) -> FileDigest:
    """Return size and SHA-256 for an existing file."""
    path = Path(path)
    return FileDigest(size=int(path.stat().st_size), sha256=sha256_file(path))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def mnist_raw_dir(root: str | Path) -> Path:
    """Return the raw IDX directory below a data root."""
    return Path(root) / "MNIST" / "raw"


def locate_mnist_files(root: str | Path, *, download_if_missing: bool = False) -> dict[str, Path]:
    """Locate the four required MNIST IDX files, optionally downloading them."""
    raw = mnist_raw_dir(root)
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for name, url in MNIST_URLS.items():
        dest = raw / name
        if not dest.is_file() or dest.stat().st_size <= 0:
            if download_if_missing:
                _download(url, dest)
            else:
                missing.append(name)
                continue
        paths[name] = dest
    if missing:
        raise AlignmentError(
            DATASET_DIGEST_MISMATCH,
            "MNIST source files are missing; refusing to invent a subset",
            evidence={"missing": missing, "raw_dir": str(raw)},
        )
    return paths


def verify_mnist_files(
    paths: dict[str, Path],
    expected: dict[str, FileDigest] | None = None,
) -> dict[str, FileDigest]:
    """Verify observed file digests against an optional frozen manifest."""
    observed = {name: file_digest(path) for name, path in paths.items()}
    if expected is None:
        return observed
    mismatches = []
    for name, digest in expected.items():
        actual = observed.get(name)
        if actual is None or actual.sha256 != digest.sha256 or actual.size != digest.size:
            mismatches.append(name)
    if mismatches:
        raise AlignmentError(
            DATASET_DIGEST_MISMATCH,
            "MNIST source digest does not match the frozen manifest",
            evidence={"mismatched_files": mismatches},
        )
    return observed


def _read_idx_images_uint8(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "bad image magic")
        if rows != 28 or cols != 28:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "unexpected image shape")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count * rows * cols:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "image payload length does not match header")
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, "bad label magic")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if data.size != count:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "label payload length does not match header")
    return data.astype(np.int64)


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Flatten uint8 images in C order and scale pixels to [0, 1]."""
    array = np.asarray(images)
    if array.dtype != np.uint8:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "pixel source must be uint8")
    return np.ascontiguousarray(array.reshape(array.shape[0], -1), dtype=np.float64) / 255.0


def first_per_class_indices(targets: np.ndarray, per_class: int) -> list[int]:
    """Select the first ``per_class`` indices for each class in original order."""
    chosen: list[int] = []
    counts = np.zeros(10, dtype=np.int64)
    for index, label_value in enumerate(np.asarray(targets).tolist()):
        label = int(label_value)
        if label < 0 or label > 9:
            raise AlignmentError(DATASET_DIGEST_MISMATCH, f"label {label} is outside 0-9")
        if counts[label] < per_class:
            chosen.append(index)
            counts[label] += 1
        if int(counts.sum()) >= 10 * per_class:
            break
    if int(counts.min()) < per_class:
        raise AlignmentError(
            DATASET_DIGEST_MISMATCH,
            f"MNIST does not contain {per_class} images for every class",
        )
    return chosen


def class_counts(labels: np.ndarray) -> tuple[int, ...]:
    """Return counts for classes 0 through 9."""
    counts = np.zeros(10, dtype=np.int64)
    for label in np.asarray(labels, dtype=np.int64).tolist():
        counts[int(label)] += 1
    return tuple(int(value) for value in counts)


def _read_idx_images(path: Path) -> np.ndarray:
    return normalize_pixels(_read_idx_images_uint8(path))


def load_raw_mnist(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw MNIST files, downloading missing files when explicitly requested."""
    paths = locate_mnist_files(root, download_if_missing=True)
    train_x = _read_idx_images(paths["train-images-idx3-ubyte.gz"])
    train_y = _read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    test_x = _read_idx_images(paths["t10k-images-idx3-ubyte.gz"])
    test_y = _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
    return train_x, train_y, test_x, test_y


def load_mnist100(
    root: str | Path,
    train_per_class: int = TRAIN_PER_CLASS,
    test_per_class: int = TEST_PER_CLASS,
) -> tuple[Mnist100Split, Mnist100Split]:
    """Load the deterministic first-per-class MNIST subset."""
    train_x, train_y, test_x, test_y = load_raw_mnist(root)
    train_idx = first_per_class_indices(train_y, train_per_class)
    test_idx = first_per_class_indices(test_y, test_per_class)
    return (
        Mnist100Split(train_x[train_idx], train_y[train_idx], train_idx),
        Mnist100Split(test_x[test_idx], test_y[test_idx], test_idx),
    )


def build_mnist100_manifest(
    root: str | Path,
    *,
    download_if_missing: bool = False,
) -> Mnist100Manifest:
    """Build and validate a digest-locked MNIST/100 manifest."""
    paths = locate_mnist_files(root, download_if_missing=download_if_missing)
    files = verify_mnist_files(paths)
    train_x_u8 = _read_idx_images_uint8(paths["train-images-idx3-ubyte.gz"])
    train_y = _read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    test_x_u8 = _read_idx_images_uint8(paths["t10k-images-idx3-ubyte.gz"])
    test_y = _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
    train_idx = tuple(first_per_class_indices(train_y, TRAIN_PER_CLASS))
    test_idx = tuple(first_per_class_indices(test_y, TEST_PER_CLASS))
    train_labels = train_y[list(train_idx)]
    test_labels = test_y[list(test_idx)]
    train_images = normalize_pixels(train_x_u8[list(train_idx)])
    test_images = normalize_pixels(test_x_u8[list(test_idx)])
    if train_images.min() < 0.0 or train_images.max() > 1.0 or test_images.min() < 0.0 or test_images.max() > 1.0:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "normalized pixels escaped [0,1]")
    manifest = Mnist100Manifest(
        schema_version=SCHEMA_VERSION,
        source_urls=dict(MNIST_URLS),
        files=files,
        train_indices=train_idx,
        test_indices=test_idx,
        train_class_counts=class_counts(train_labels),
        test_class_counts=class_counts(test_labels),
        image_shape=(28, 28),
        flatten_order="C",
        input_dtype="float64",
        preprocessing="uint8/255.0, no augmentation",
        value_range=(0.0, 1.0),
        selection_rule="first 100 train and 10 test images per class in original IDX order",
        created_at_utc=utc_now(),
        manifest_sha256="",
    )
    frozen = Mnist100Manifest(
        **{**manifest.__dict__, "manifest_sha256": manifest.compute_manifest_sha256()}
    )
    frozen.validate()
    return frozen


def manifest_from_json(payload: dict) -> Mnist100Manifest:
    """Parse and validate a manifest dictionary."""
    files = {
        name: FileDigest(size=int(item["size"]), sha256=str(item["sha256"]))
        for name, item in payload["files"].items()
    }
    manifest = Mnist100Manifest(
        schema_version=payload["schema_version"],
        source_urls=dict(payload["source_urls"]),
        files=files,
        train_indices=tuple(int(index) for index in payload["train_indices"]),
        test_indices=tuple(int(index) for index in payload["test_indices"]),
        train_class_counts=tuple(int(value) for value in payload["train_class_counts"]),
        test_class_counts=tuple(int(value) for value in payload["test_class_counts"]),
        image_shape=(int(payload["image_shape"][0]), int(payload["image_shape"][1])),
        flatten_order=payload["flatten_order"],
        input_dtype=payload["input_dtype"],
        preprocessing=payload["preprocessing"],
        value_range=(float(payload["value_range"][0]), float(payload["value_range"][1])),
        selection_rule=payload["selection_rule"],
        created_at_utc=payload["created_at_utc"],
        manifest_sha256=payload["manifest_sha256"],
    )
    manifest.validate()
    return manifest


def load_mnist100_from_manifest(
    root: str | Path,
    manifest: Mnist100Manifest,
) -> tuple[Mnist100Split, Mnist100Split]:
    """Load data while enforcing a frozen manifest."""
    manifest.validate()
    paths = locate_mnist_files(root, download_if_missing=False)
    verify_mnist_files(paths, manifest.files)
    train_x = _read_idx_images(paths["train-images-idx3-ubyte.gz"])
    train_y = _read_idx_labels(paths["train-labels-idx1-ubyte.gz"])
    test_x = _read_idx_images(paths["t10k-images-idx3-ubyte.gz"])
    test_y = _read_idx_labels(paths["t10k-labels-idx1-ubyte.gz"])
    train_idx = list(manifest.train_indices)
    test_idx = list(manifest.test_indices)
    train = Mnist100Split(train_x[train_idx], train_y[train_idx], train_idx)
    test = Mnist100Split(test_x[test_idx], test_y[test_idx], test_idx)
    if class_counts(train.labels) != manifest.train_class_counts:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "train labels drifted from the frozen manifest")
    if class_counts(test.labels) != manifest.test_class_counts:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "test labels drifted from the frozen manifest")
    if train.images.dtype != np.float64 or test.images.dtype != np.float64:
        raise AlignmentError(DATASET_DIGEST_MISMATCH, "loaded images must be float64")
    return train, test
