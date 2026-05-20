from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: str


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and _is_image(p)]


def discover_binary_split(dataset_root: Path, split: str) -> List[Sample]:
    split_dir = dataset_root / split
    ai_dir = split_dir / "ai"
    nature_dir = split_dir / "nature"

    ai_images = [Sample(path=p, label="ai") for p in _list_images(ai_dir)]
    nature_images = [Sample(path=p, label="nature") for p in _list_images(nature_dir)]
    return ai_images + nature_images


def discover_model_roots(dataset_root: Path) -> List[Path]:
    roots: List[Path] = []

    # Layout A: single merged root with train/val directly under dataset_root.
    if (
        (dataset_root / "train" / "ai").exists()
        and (dataset_root / "train" / "nature").exists()
        and (dataset_root / "val" / "ai").exists()
        and (dataset_root / "val" / "nature").exists()
    ):
        return [dataset_root]

    # Layout B: multiple model folders, each with train/val/ai/nature.
    for train_dir in dataset_root.rglob("train"):
        root = train_dir.parent
        if (
            (root / "train" / "ai").exists()
            and (root / "train" / "nature").exists()
            and (root / "val" / "ai").exists()
            and (root / "val" / "nature").exists()
        ):
            roots.append(root)

    # de-duplicate and stable sort
    unique = sorted({r.resolve() for r in roots}, key=lambda p: str(p))
    return [Path(p) for p in unique]


def discover_binary_split_multi_root(dataset_root: Path, split: str) -> List[Sample]:
    roots = discover_model_roots(dataset_root)
    samples: List[Sample] = []
    for root in roots:
        samples.extend(discover_binary_split(root, split))
    return samples


def discover_ai_subsource_from_roots(dataset_root: Path, split: str) -> List[Sample]:
    roots = discover_model_roots(dataset_root)
    samples: List[Sample] = []
    for root in roots:
        model_label = root.parent.name if root.parent != dataset_root else root.name
        ai_dir = root / split / "ai"
        for img_path in _list_images(ai_dir):
            samples.append(Sample(path=img_path, label=model_label))
    return samples


def discover_ai_subsource_split(dataset_root: Path, split: str) -> List[Sample]:
    split_dir = dataset_root / split
    ai_dir = split_dir / "ai"
    if not ai_dir.exists():
        return []

    samples: List[Sample] = []
    for subdir in sorted([p for p in ai_dir.iterdir() if p.is_dir()]):
        for img_path in _list_images(subdir):
            samples.append(Sample(path=img_path, label=subdir.name))
    return samples


def summarize_labels(samples: Sequence[Sample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.label] = counts.get(s.label, 0) + 1
    return counts


def validate_non_empty(samples: Sequence[Sample], split_name: str) -> None:
    if not samples:
        raise ValueError(
            f"No images found for split '{split_name}'. "
            "Expected folders like train/ai, train/nature, val/ai, val/nature."
        )
