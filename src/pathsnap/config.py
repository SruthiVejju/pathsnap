"""Shared configuration loader. Everyone imports paths/constants from here."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()
SEED = CFG["seed"]


def data_root() -> Path:
    env = os.environ.get("PATHSNAP_DATA")
    return Path(env) if env else (ROOT / CFG["data_root"]).resolve()


def trip_ids(partition: str) -> list[str]:
    """partition: 'train' or 'student_test'. Evaluator data is intentionally unsupported."""
    if partition not in ("train", "student_test"):
        raise ValueError("Only 'train' and 'student_test' are allowed during development.")
    d = data_root() / CFG["train_dir" if partition == "train" else "student_test_dir"]
    return sorted(p.name for p in d.iterdir() if p.is_dir())
