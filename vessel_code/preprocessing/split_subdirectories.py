# Transferred from methods/data/split_subdirectories.py. See TRANSFER_MANIFEST.json.
from __future__ import annotations
from pathlib import Path
from typing import Iterable

"""Discover an existing train/validation/test NPZ directory split."""

def discover_npz_split_subdirectories(
    dataset_dir: str | Path,
    *,
    exclude_dir_names: Iterable[str] | None = None,
) -> dict[str, list[Path]]:
    """Return the supplied split without creating or reshuffling membership."""

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    train_dir = root / "train"
    test_dir = root / "test"
    validation_dirs = [
        path for path in (root / "validation", root / "val") if path.is_dir()
    ]
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing training directory: {train_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")
    if not validation_dirs:
        raise FileNotFoundError(
            f"Missing validation directory: expected {root / 'validation'} "
            f"(or the alias {root / 'val'})"
        )
    if len(validation_dirs) > 1:
        raise ValueError(
            f"Both {root / 'validation'} and {root / 'val'} exist; keep one "
            "canonical validation directory."
        )

    excluded = {
        str(name).strip().casefold() for name in (exclude_dir_names or ())
    }

    def files_under(directory: Path) -> list[Path]:
        paths = []
        for path in sorted(directory.rglob("*.npz")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            if any(part.casefold() in excluded for part in relative.parts[:-1]):
                continue
            paths.append(path.resolve())
        if not paths:
            raise FileNotFoundError(f"No NPZ files found under {directory}")
        return paths

    split = {
        "train": files_under(train_dir),
        "val": files_under(validation_dirs[0]),
        "test": files_under(test_dir),
    }
    seen: dict[Path, str] = {}
    for split_name, paths in split.items():
        for path in paths:
            previous = seen.get(path)
            if previous is not None:
                raise ValueError(
                    f"Feature file {path} occurs in both {previous!r} and "
                    f"{split_name!r}."
                )
            seen[path] = split_name
    return split
