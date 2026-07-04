#!/usr/bin/env python3
"""
gnn.py — odtwarzalny pipeline do hierarchicznej klasyfikacji wieloetykietowej
cząsteczek ChEBI.

Skrypt realizuje kompletny przebieg:
  1. walidacja train/test/submission,
  2. parsowanie ontologii OBO i operacje na DAG,
  3. kanonikalizacja SMILES i grupowy split 80/20,
  4. gotowy fallback Morgan/ECFP z awaryjnym submission,
  5. cache grafów molekularnych,
  6. GINE 4x192 (lub automatycznie 3x128),
  7. dobór progów per klasa i porównanie wariantów DAG,
  8. domyślnie inferencja ze skalibrowanego checkpointu walidacyjnego
     (opcjonalnie pełny retraining),
  9. zapis i pełna walidacja submission.parquet.

Domyślne uruchomienie:

    python gnn.py \
      --train chebi_dataset_train.parquet \
      --test chebi_dataset_test_empty.parquet \
      --ontology chebi_classes.obo \
      --sample-submission chebi_submission_example.parquet \
      --output-dir artifacts

Najważniejsze zależności:
    Python >= 3.10, numpy, pandas, pyarrow, torch, rdkit.

PyTorch Geometric NIE jest wymagany — GINE i batchowanie grafów są
zaimplementowane bezpośrednio w PyTorch, aby plik był samodzielny.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError as exc:  # pragma: no cover - komunikat użytkowy
    raise SystemExit(
        "Brak PyTorch. Zainstaluj torch zgodny z CUDA na docelowym serwerze."
    ) from exc

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Brak RDKit. Zainstaluj pakiet rdkit.") from exc

RDLogger.DisableLog("rdApp.warning")

LOGGER = logging.getLogger("chebi_gnn")
GRAPH_FEATURE_VERSION = "graph_v3_pure_torch_gine"
ECFP_FEATURE_VERSION = "ecfp_morgan_chiral_v4"
CLASS_COUNT = 500
CLASS_NAMES = [f"class_{i}" for i in range(CLASS_COUNT)]

ATOM_CARDINALITIES = [
    119,  # atomic number 0..118
    11,   # degree 0..10
    17,   # formal charge -8..8
    8,    # hybridization
    2,    # aromatic
    9,    # explicit H 0..8
    9,    # implicit H 0..8
    2,    # in ring
    4,    # chirality
]
EDGE_CARDINALITIES = [
    5,  # bond type
    2,  # aromatic
    2,  # conjugated
    2,  # in ring
    7,  # stereo
    7,  # direction
]

THRESHOLD_GRID_DEFAULT = [
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


# ---------------------------------------------------------------------------
# Konfiguracja, logowanie i narzędzia
# ---------------------------------------------------------------------------


@dataclass
class Config:
    train: Path
    test: Path
    ontology: Path
    sample_submission: Optional[Path]
    output_dir: Path
    submission_path: Path
    model: str = "auto"  # auto | gine | ecfp
    device: str = "auto"
    seed: int = 42
    val_fraction: float = 0.20
    num_workers: int = 4
    feature_workers: int = 1
    target_atoms_per_batch: int = 12000
    eval_atoms_per_batch: int = 16000
    bucket_size: int = 256
    num_layers: int = 4
    hidden_dim: int = 192
    head_hidden_dim: int = 384
    gnn_dropout: float = 0.15
    head_dropout: float = 0.20
    max_epochs: int = 20
    small_gnn_max_epochs: int = 18
    patience: int = 3
    evaluation_every: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    pos_weight_clip: float = 6.0
    working_threshold: float = 0.30
    use_bf16: bool = True
    time_limit_minutes: float = 60.0
    buffer_minutes: float = 5.0
    graph_preprocess_limit_minutes: float = 10.0
    benchmark_fast_seconds: float = 50.0
    benchmark_slow_seconds: float = 70.0
    ecfp_bits: int = 4096
    ecfp_radius: int = 2
    ecfp_hidden_dim: int = 1024
    ecfp_batch_size: int = 512
    ecfp_eval_batch_size: int = 1024
    ecfp_max_epochs: int = 15
    ecfp_patience: int = 3
    ecfp_descriptors: bool = False
    ecfp_variant: str = "auto"  # auto | mlp | linear
    threshold_grid: list[float] = field(
        default_factory=lambda: THRESHOLD_GRID_DEFAULT.copy()
    )
    force_class0: bool = True
    hierarchy_f1_tolerance: float = 0.01
    final_time_safety_factor: float = 1.35
    final_inference_reserve_seconds: float = 180.0
    overwrite_cache: bool = False
    dry_run: bool = False
    self_test: bool = False
    debug_sample: int = 0
    save_probabilities: bool = True
    prepare_ecfp_fallback: bool = True
    checkpoint_candidates: int = 3
    final_strategy: str = "validation_checkpoint"  # validation_checkpoint | full_retrain

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "reports"


class StageTimer:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        LOGGER.info("[START] %s", name)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            LOGGER.info("[KONIEC] %s — %.2f s", name, elapsed)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


class TimeBudget:
    def __init__(self, limit_minutes: float, buffer_minutes: float) -> None:
        self.started = time.perf_counter()
        self.limit_seconds = limit_minutes * 60.0
        self.buffer_seconds = buffer_minutes * 60.0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit_seconds - self.elapsed)

    @property
    def usable_remaining(self) -> float:
        return max(0.0, self.remaining - self.buffer_seconds)

    def summary(self) -> dict[str, float]:
        return {
            "elapsed_seconds": self.elapsed,
            "remaining_seconds": self.remaining,
            "usable_remaining_seconds": self.usable_remaining,
            "limit_seconds": self.limit_seconds,
            "buffer_seconds": self.buffer_seconds,
        }


class NeedSmallGNN(RuntimeError):
    pass


class NeedECFP(RuntimeError):
    pass


class TimeBudgetExceeded(RuntimeError):
    pass


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def ensure_dirs(cfg: Config) -> None:
    for path in [cfg.output_dir, cfg.cache_dir, cfg.checkpoint_dir, cfg.report_dir]:
        path.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: Config) -> None:
    issues: list[str] = []
    if not 0.0 < cfg.val_fraction < 1.0:
        issues.append("val_fraction musi należeć do (0, 1).")
    if cfg.evaluation_every < 1:
        issues.append("evaluation_every musi być >= 1.")
    if cfg.max_epochs < 1 or cfg.ecfp_max_epochs < 1:
        issues.append("Liczba epok musi być >= 1.")
    if cfg.patience < 1 or cfg.ecfp_patience < 1:
        issues.append("patience musi być >= 1.")
    if cfg.target_atoms_per_batch < 1 or cfg.eval_atoms_per_batch < 1:
        issues.append("Budżet atomów na batch musi być >= 1.")
    if cfg.ecfp_bits < 8 or cfg.ecfp_radius < 0:
        issues.append("Konfiguracja ECFP jest nieprawidłowa.")
    if cfg.num_layers < 1 or cfg.hidden_dim < 1 or cfg.head_hidden_dim < 1:
        issues.append("Wymiary i liczba warstw GNN muszą być dodatnie.")
    if cfg.ecfp_hidden_dim < 1:
        issues.append("ecfp_hidden_dim musi być >= 1.")
    if cfg.ecfp_batch_size < 1 or cfg.ecfp_eval_batch_size < 1:
        issues.append("Rozmiary batchy ECFP muszą być >= 1.")
    if cfg.bucket_size < 1:
        issues.append("bucket_size musi być >= 1.")
    if not 0.0 <= cfg.gnn_dropout < 1.0 or not 0.0 <= cfg.head_dropout < 1.0:
        issues.append("Dropout musi należeć do [0, 1).")
    if cfg.learning_rate <= 0.0:
        issues.append("learning_rate musi być > 0.")
    if cfg.weight_decay < 0.0 or cfg.gradient_clip <= 0.0:
        issues.append("weight_decay musi być >= 0, a gradient_clip > 0.")
    if cfg.pos_weight_clip < 1.0:
        issues.append("pos_weight_clip musi być >= 1.")
    if not 0.0 <= cfg.working_threshold <= 1.0:
        issues.append("working_threshold musi należeć do [0, 1].")
    if cfg.checkpoint_candidates < 1:
        issues.append("checkpoint_candidates musi być >= 1.")
    if cfg.final_strategy not in {"validation_checkpoint", "full_retrain"}:
        issues.append(
            "final_strategy musi mieć wartość validation_checkpoint lub full_retrain."
        )
    if cfg.time_limit_minutes <= 0 or cfg.buffer_minutes < 0:
        issues.append("Limity czasu są nieprawidłowe.")
    if cfg.buffer_minutes >= cfg.time_limit_minutes:
        issues.append("buffer_minutes musi być mniejszy niż time_limit_minutes.")
    if not cfg.threshold_grid or any(
        threshold < 0.0 or threshold > 1.0 for threshold in cfg.threshold_grid
    ):
        issues.append("Siatka progów musi być niepusta i należeć do [0, 1].")
    if issues:
        raise ValueError("Nieprawidłowa konfiguracja:\n- " + "\n- ".join(issues))


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def stable_signature(payload: dict[str, Any], length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def effective_dataset_hash(
    file_hash: str, dataframe: pd.DataFrame, cfg: Config, split_name: str
) -> str:
    """Hash cache'u uwzględniający podzbiór debug i kolejność rekordów."""
    row_ids = dataframe["mol_id"].astype(str).tolist() if "mol_id" in dataframe else []
    payload = {
        "file_hash": file_hash,
        "split": split_name,
        "debug_sample": int(cfg.debug_sample),
        "seed": int(cfg.seed),
        "row_count": len(dataframe),
        "row_ids_hash": hashlib.sha256("\n".join(row_ids).encode()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "rdkit": package_version("rdkit"),
        "pyarrow": package_version("pyarrow"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    # Informacje o GPU pobieramy tylko dla faktycznie wybranego urządzenia CUDA.
    # Na maszynie z GPU użytkownik nadal może świadomie uruchomić --device cpu.
    if device.type == "cuda":
        report.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_gb": torch.cuda.get_device_properties(device).total_memory
                / 1024**3,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return report


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano CUDA, ale torch.cuda.is_available() == False.")
    return device


def autocast_context(device: torch.device, use_bf16: bool):
    enabled = (
        use_bf16
        and device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Do obsługi Parquet potrzebny jest pyarrow. Zainstaluj: pip install pyarrow"
        ) from exc


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Do zapisu Parquet potrzebny jest pyarrow. Zainstaluj: pip install pyarrow"
        ) from exc


# ---------------------------------------------------------------------------
# Dane i walidacja
# ---------------------------------------------------------------------------


def load_and_validate_data(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]]:
    train_df = read_parquet(cfg.train)
    test_df = read_parquet(cfg.test)
    sample_df = read_parquet(cfg.sample_submission) if cfg.sample_submission else None

    if cfg.debug_sample > 0:
        n_train = min(cfg.debug_sample, len(train_df))
        n_test = min(max(1, cfg.debug_sample // 3), len(test_df))
        train_df = train_df.sample(n=n_train, random_state=cfg.seed).reset_index(drop=True)
        test_df = test_df.iloc[:n_test].reset_index(drop=True)
        if sample_df is not None:
            sample_df = sample_df.iloc[:n_test].reset_index(drop=True)
        LOGGER.warning(
            "Tryb debug: train=%d, test=%d. Wynik nie nadaje się do submission.",
            n_train,
            n_test,
        )

    expected_train_columns = ["mol_id", "SMILES", *CLASS_NAMES]
    expected_test_columns = ["mol_id", "SMILES"]
    issues: list[str] = []

    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    if train_columns != expected_train_columns:
        issues.append(
            "Kolumny train nie są dokładnie: mol_id, SMILES, class_0...class_499."
        )
    if test_columns != expected_test_columns:
        issues.append("Kolumny test nie są dokładnie: mol_id, SMILES.")
    if len(train_columns) != 502:
        issues.append(f"Train ma {len(train_columns)} kolumn zamiast 502.")
    if len(test_columns) != 2:
        issues.append(f"Test ma {len(test_columns)} kolumn zamiast 2.")
    if len(train_df) == 0:
        issues.append("Train jest pusty.")
    if len(test_df) == 0:
        issues.append("Test jest pusty.")

    missing_train = [name for name in expected_train_columns if name not in train_df.columns]
    missing_test = [name for name in expected_test_columns if name not in test_df.columns]
    if missing_train:
        issues.append(f"Brakujące kolumny train: {missing_train[:20]}.")
    if missing_test:
        issues.append(f"Brakujące kolumny test: {missing_test}.")

    # Nie odwołujemy się do brakujących kolumn. Dzięki temu błędny plik daje
    # czytelny raport walidacji zamiast przypadkowego KeyError.
    if missing_train or missing_test or len(train_df) == 0 or len(test_df) == 0:
        report = {
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "sample_shape": list(sample_df.shape) if sample_df is not None else None,
            "class_count": len(CLASS_NAMES),
            "issues": issues,
        }
        write_json(cfg.report_dir / "data_validation.json", report)
        raise ValueError("Walidacja danych nie powiodła się:\n- " + "\n- ".join(issues))

    if train_df["mol_id"].isna().any() or train_df["SMILES"].isna().any():
        issues.append("Train zawiera NaN w mol_id lub SMILES.")
    if test_df["mol_id"].isna().any() or test_df["SMILES"].isna().any():
        issues.append("Test zawiera NaN w mol_id lub SMILES.")
    if not train_df["mol_id"].is_unique:
        issues.append("mol_id w train nie są unikalne.")
    if not test_df["mol_id"].is_unique:
        issues.append("mol_id w test nie są unikalne.")

    overlap_ids = set(train_df["mol_id"]).intersection(set(test_df["mol_id"]))
    if overlap_ids:
        issues.append(f"Train i test mają {len(overlap_ids)} wspólnych mol_id.")

    labels = train_df[CLASS_NAMES].to_numpy(copy=False)
    if pd.isna(labels).any():
        issues.append("Macierz etykiet zawiera NaN.")
    try:
        unique_values = np.unique(labels)
        if not np.isin(unique_values, [0, 1]).all():
            issues.append(
                f"Etykiety nie są binarne; wartości={unique_values[:20].tolist()}."
            )
    except (TypeError, ValueError) as exc:
        unique_values = np.asarray([], dtype=np.float32)
        issues.append(f"Nie można zweryfikować binarności etykiet: {exc}.")

    if sample_df is not None:
        if list(sample_df.columns) != expected_train_columns:
            issues.append("Sample submission ma nieprawidłowe kolumny lub kolejność.")
        elif len(sample_df) != len(test_df):
            issues.append("Sample submission ma inną liczbę wierszy niż test.")
        else:
            if not sample_df["mol_id"].equals(test_df["mol_id"]):
                issues.append("mol_id w sample submission nie zgadzają się z test.")
            if not sample_df["SMILES"].equals(test_df["SMILES"]):
                issues.append("SMILES w sample submission nie zgadzają się z test.")

    label_density = float(np.asarray(labels, dtype=np.float64).mean())
    mean_labels = float(np.asarray(labels, dtype=np.float64).sum(axis=1).mean())
    report = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "sample_shape": list(sample_df.shape) if sample_df is not None else None,
        "class_count": len(CLASS_NAMES),
        "label_unique_values": unique_values.tolist(),
        "label_density": label_density,
        "mean_labels_per_record": mean_labels,
        "train_unique_ids": int(train_df["mol_id"].nunique()),
        "test_unique_ids": int(test_df["mol_id"].nunique()),
        "overlap_ids": len(overlap_ids),
        "issues": issues,
    }
    write_json(cfg.report_dir / "data_validation.json", report)
    if issues:
        raise ValueError("Walidacja danych nie powiodła się:\n- " + "\n- ".join(issues))
    return train_df, test_df, sample_df, report


# ---------------------------------------------------------------------------
# Ontologia OBO i DAG
# ---------------------------------------------------------------------------


class Ontology:
    def __init__(self, class_names: Sequence[str], edges: Sequence[tuple[str, str]]):
        self.class_names = list(class_names)
        self.index = {name: i for i, name in enumerate(self.class_names)}
        n = len(self.class_names)
        self.parents: list[list[int]] = [[] for _ in range(n)]
        self.children: list[list[int]] = [[] for _ in range(n)]
        edge_indices: list[tuple[int, int]] = []

        seen: set[tuple[int, int]] = set()
        for parent_name, child_name in edges:
            if parent_name not in self.index or child_name not in self.index:
                continue
            parent = self.index[parent_name]
            child = self.index[child_name]
            if (parent, child) in seen:
                continue
            seen.add((parent, child))
            self.parents[child].append(parent)
            self.children[parent].append(child)
            edge_indices.append((parent, child))
        self.edges = edge_indices
        self.topological_order = self._topological_sort()
        self.reverse_topological_order = list(reversed(self.topological_order))
        self.ancestors = self._build_ancestors()

    @classmethod
    def from_obo(cls, path: Path, class_names: Sequence[str]) -> "Ontology":
        terms: dict[str, list[str]] = {}
        current_id: Optional[str] = None
        current_parents: list[str] = []

        def flush() -> None:
            nonlocal current_id, current_parents
            if current_id is not None:
                terms[current_id] = current_parents.copy()
            current_id = None
            current_parents = []

        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == "[Term]":
                    flush()
                elif line.startswith("["):
                    flush()
                elif line.startswith("id:"):
                    current_id = line.split(":", 1)[1].strip()
                elif line.startswith("is_a:"):
                    parent = line.split(":", 1)[1].split("!", 1)[0].strip()
                    current_parents.append(parent)
            flush()

        missing = [name for name in class_names if name not in terms]
        if missing:
            raise ValueError(
                f"Ontologia nie zawiera {len(missing)} klas, np. {missing[:10]}."
            )
        edges = [
            (parent, child)
            for child, parents in terms.items()
            if child in set(class_names)
            for parent in parents
        ]
        return cls(class_names, edges)

    def _topological_sort(self) -> list[int]:
        indegree = np.array([len(p) for p in self.parents], dtype=np.int64)
        queue = deque(int(i) for i in np.flatnonzero(indegree == 0))
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.class_names):
            raise ValueError("Graf ontologii zawiera cykl.")
        return order

    def _build_ancestors(self) -> list[np.ndarray]:
        ancestor_sets: list[set[int]] = [set() for _ in self.class_names]
        for node in self.topological_order:
            for parent in self.parents[node]:
                ancestor_sets[node].add(parent)
                ancestor_sets[node].update(ancestor_sets[parent])
        return [np.array(sorted(values), dtype=np.int64) for values in ancestor_sets]

    def binary_closure(self, predictions: np.ndarray) -> np.ndarray:
        out = np.asarray(predictions, dtype=np.uint8).copy()
        if out.ndim != 2 or out.shape[1] != len(self.class_names):
            raise ValueError("Nieprawidłowy kształt predykcji dla closure DAG.")
        for child, ancestors in enumerate(self.ancestors):
            if ancestors.size:
                out[:, ancestors] |= out[:, [child]]
        return out

    def monotonic_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        out = np.asarray(probabilities, dtype=np.float32).copy()
        for child in self.reverse_topological_order:
            for parent in self.parents[child]:
                np.maximum(out[:, parent], out[:, child], out=out[:, parent])
        return out

    def inconsistency_counts(
        self, values: np.ndarray, eps: float = 0.0
    ) -> dict[str, float | int]:
        arr = np.asarray(values)
        if not self.edges:
            return {"total": 0, "mean_per_record": 0.0, "records_with_any": 0}
        parent_idx = np.fromiter((p for p, _ in self.edges), dtype=np.int64)
        child_idx = np.fromiter((c for _, c in self.edges), dtype=np.int64)
        violations = arr[:, child_idx] > (arr[:, parent_idx] + eps)
        per_row = violations.sum(axis=1)
        return {
            "total": int(per_row.sum()),
            "mean_per_record": float(per_row.mean()),
            "records_with_any": int((per_row > 0).sum()),
        }

    def report(self) -> dict[str, Any]:
        roots = [self.class_names[i] for i, parents in enumerate(self.parents) if not parents]
        return {
            "nodes": len(self.class_names),
            "edges": len(self.edges),
            "roots": roots,
            "is_dag": True,
            "max_ancestor_count": max((len(a) for a in self.ancestors), default=0),
        }


def load_ontology(cfg: Config) -> Ontology:
    ontology = Ontology.from_obo(cfg.ontology, CLASS_NAMES)
    report = ontology.report()
    issues: list[str] = []
    if report["nodes"] != 500:
        issues.append(f"Ontologia ma {report['nodes']} węzłów zamiast 500.")
    if report["edges"] != 748:
        issues.append(f"Ontologia ma {report['edges']} krawędzi zamiast 748.")
    report["issues"] = issues
    write_json(cfg.report_dir / "ontology_validation.json", report)
    if issues:
        raise ValueError("Walidacja ontologii nie powiodła się:\n- " + "\n- ".join(issues))
    return ontology


# ---------------------------------------------------------------------------
# Kanonikalizacja i grupowy split wieloetykietowy
# ---------------------------------------------------------------------------


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit nie może sparsować SMILES: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_cache_path(cfg: Config, train_hash: str) -> Path:
    signature = stable_signature(
        {
            "version": "canonical_isomeric_v2",
            "rdkit": package_version("rdkit"),
        }
    )
    return cfg.cache_dir / f"canonical_train_{train_hash[:12]}_{signature}.npy"


def get_canonical_smiles(
    train_df: pd.DataFrame, cfg: Config, train_hash: str
) -> np.ndarray:
    path = canonical_cache_path(cfg, train_hash)
    if path.exists() and not cfg.overwrite_cache:
        values = np.load(path, allow_pickle=True)
        if len(values) == len(train_df):
            LOGGER.info("Wczytano cache kanonicznych SMILES: %s", path)
            return values.astype(object)
        LOGGER.warning("Cache kanoniczny ma zły rozmiar — przebudowuję.")

    values: list[str] = []
    total = len(train_df)
    for i, smiles in enumerate(train_df["SMILES"].astype(str)):
        values.append(canonicalize_smiles(smiles))
        if (i + 1) % 5000 == 0 or i + 1 == total:
            LOGGER.info("Kanonikalizacja: %d/%d", i + 1, total)
    array = np.asarray(values, dtype=object)
    np.save(path, array, allow_pickle=True)
    return array


def duplicate_conflict_report(
    canonical: np.ndarray, labels: np.ndarray, mol_ids: Sequence[str]
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, value in enumerate(canonical):
        groups[str(value)].append(idx)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    conflicts: list[dict[str, Any]] = []
    for indices in duplicate_groups:
        unique_rows = np.unique(labels[indices], axis=0)
        if len(unique_rows) > 1:
            differing = np.flatnonzero(labels[indices].max(0) != labels[indices].min(0))
            conflicts.append(
                {
                    "canonical_smiles": str(canonical[indices[0]]),
                    "indices": indices,
                    "mol_ids": [str(mol_ids[i]) for i in indices],
                    "different_class_count": int(len(differing)),
                    "different_classes": [CLASS_NAMES[i] for i in differing[:50]],
                }
            )
    return {
        "duplicate_group_count": len(duplicate_groups),
        "records_in_duplicate_groups": int(sum(len(x) for x in duplicate_groups)),
        "conflicting_group_count": len(conflicts),
        "conflicts": conflicts[:100],
    }


def _custom_group_multilabel_split(
    group_labels: np.ndarray,
    group_sizes: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy stratification of groups with emphasis on rare labels.

    The objective is record-level: group label counts are expected to contain
    sums of labels inside a group, not only OR labels.
    """

    rng = np.random.default_rng(seed)
    n_groups, n_classes = group_labels.shape
    target_size = float(group_sizes.sum() * val_fraction)
    target_labels = group_labels.sum(axis=0).astype(np.float64) * val_fraction
    denom = np.maximum(target_labels, 1.0)

    presence = (group_labels > 0).astype(np.float64)
    group_frequency = presence.sum(axis=0)
    rarity_weights = 1.0 / np.maximum(group_frequency, 1.0)
    rarity_score = presence @ rarity_weights
    random_tie = rng.random(n_groups)
    order = np.lexsort((random_tie, -group_sizes, -rarity_score))

    val_mask = np.zeros(n_groups, dtype=bool)
    current_size = 0.0
    current_labels = np.zeros(n_classes, dtype=np.float64)

    def objective(size: float, label_counts: np.ndarray) -> float:
        size_error = abs(size - target_size) / max(target_size, 1.0)
        label_error = np.mean(np.abs(label_counts - target_labels) / denom)
        # Rozkład etykiet jest ważniejszy, ale rozmiar nie może odjechać.
        return 1.5 * label_error + size_error

    current_objective = objective(current_size, current_labels)
    for group_idx in order:
        new_size = current_size + float(group_sizes[group_idx])
        new_labels = current_labels + group_labels[group_idx]
        new_objective = objective(new_size, new_labels)
        must_fill = current_size < target_size * 0.92
        too_large = new_size > target_size * 1.08 and current_size >= target_size * 0.90
        if (new_objective < current_objective or must_fill) and not too_large:
            val_mask[group_idx] = True
            current_size = new_size
            current_labels = new_labels
            current_objective = new_objective

    # Korekta rozmiaru przez dodawanie/usuwanie grup o najmniejszym koszcie.
    for _ in range(max(100, n_groups * 2)):
        size_error = current_size - target_size
        if abs(size_error) <= max(1.0, target_size * 0.005):
            break
        candidates = np.flatnonzero(~val_mask if size_error < 0 else val_mask)
        if candidates.size == 0:
            break
        best_group = None
        best_obj = float("inf")
        for group_idx in rng.choice(
            candidates, size=min(len(candidates), 512), replace=False
        ):
            sign = 1.0 if size_error < 0 else -1.0
            candidate_size = current_size + sign * float(group_sizes[group_idx])
            candidate_labels = current_labels + sign * group_labels[group_idx]
            obj = objective(candidate_size, candidate_labels)
            if obj < best_obj:
                best_obj = obj
                best_group = int(group_idx)
        if best_group is None:
            break
        if size_error < 0:
            val_mask[best_group] = True
            current_size += float(group_sizes[best_group])
            current_labels += group_labels[best_group]
        else:
            val_mask[best_group] = False
            current_size -= float(group_sizes[best_group])
            current_labels -= group_labels[best_group]
        current_objective = best_obj

    val_groups = np.flatnonzero(val_mask)
    train_groups = np.flatnonzero(~val_mask)
    if len(val_groups) == 0 or len(train_groups) == 0:
        raise RuntimeError("Nie udało się utworzyć niepustego splitu grupowego.")
    return train_groups, val_groups


def make_or_load_split(
    train_df: pd.DataFrame,
    labels: np.ndarray,
    canonical: np.ndarray,
    cfg: Config,
    train_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    split_path = cfg.cache_dir / (
        f"split_{train_hash[:12]}_seed{cfg.seed}_val{cfg.val_fraction:.3f}.npz"
    )
    report_path = cfg.report_dir / "split_report.json"
    if split_path.exists() and not cfg.overwrite_cache:
        try:
            data = np.load(split_path)
            train_idx = data["train_idx"].astype(np.int64)
            val_idx = data["val_idx"].astype(np.int64)
            all_idx = np.concatenate([train_idx, val_idx])
            valid = (
                len(train_idx) > 0
                and len(val_idx) > 0
                and len(all_idx) == len(train_df)
                and np.array_equal(np.sort(all_idx), np.arange(len(train_df)))
                and len(np.unique(all_idx)) == len(train_df)
                and not set(canonical[train_idx].tolist()).intersection(
                    set(canonical[val_idx].tolist())
                )
            )
            if valid:
                LOGGER.info("Wczytano i zweryfikowano split z cache: %s", split_path)
                val_positive = labels[val_idx].sum(axis=0)
                train_positive = labels[train_idx].sum(axis=0)
                report = {
                    "seed": cfg.seed,
                    "val_fraction_requested": cfg.val_fraction,
                    "train_records": len(train_idx),
                    "validation_records": len(val_idx),
                    "validation_fraction_actual": len(val_idx) / len(train_df),
                    "group_count": int(len(np.unique(canonical))),
                    "canonical_overlap": 0,
                    "classes_without_positive_in_validation": [
                        CLASS_NAMES[i] for i in np.flatnonzero(val_positive == 0)
                    ],
                    "train_positive_per_class": train_positive.tolist(),
                    "validation_positive_per_class": val_positive.tolist(),
                    "loaded_from_cache": True,
                }
                write_json(report_path, report)
                return train_idx, val_idx, report
            LOGGER.warning("Split z cache jest niespójny — przebudowuję.")
        except (OSError, KeyError, ValueError, IndexError) as exc:
            LOGGER.warning("Nie można zweryfikować splitu z cache (%s) — przebudowuję.", exc)

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_idx, canon in enumerate(canonical):
        grouped[str(canon)].append(row_idx)
    group_keys = list(grouped.keys())
    group_indices = [np.asarray(grouped[key], dtype=np.int64) for key in group_keys]
    group_sizes = np.asarray([len(idx) for idx in group_indices], dtype=np.int64)
    group_label_counts = np.stack([labels[idx].sum(axis=0) for idx in group_indices])

    train_groups: np.ndarray
    val_groups: np.ndarray
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        # Dla prawie wszystkich grup rozmiar=1; po splitowaniu grup robimy małą
        # korektę tylko w custom fallbacku. Ta ścieżka jest bardzo dobra, gdy
        # iterative-stratification jest dostępne.
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=cfg.val_fraction, random_state=cfg.seed
        )
        group_presence = (group_label_counts > 0).astype(np.uint8)
        train_groups, val_groups = next(
            splitter.split(np.zeros(len(group_keys)), group_presence)
        )
        LOGGER.info("Split utworzony przez iterative-stratification na grupach.")
    except ImportError:
        LOGGER.info(
            "Brak iterative-stratification — używam wbudowanego algorytmu greedy."
        )
        train_groups, val_groups = _custom_group_multilabel_split(
            group_label_counts, group_sizes, cfg.val_fraction, cfg.seed
        )

    train_idx = np.concatenate([group_indices[i] for i in train_groups])
    val_idx = np.concatenate([group_indices[i] for i in val_groups])
    train_idx.sort()
    val_idx.sort()

    train_canon = set(canonical[train_idx].tolist())
    val_canon = set(canonical[val_idx].tolist())
    overlap = train_canon.intersection(val_canon)
    if overlap:
        raise AssertionError("Split rozdzielił tę samą kanoniczną strukturę.")

    val_positive = labels[val_idx].sum(axis=0)
    train_positive = labels[train_idx].sum(axis=0)
    report = {
        "seed": cfg.seed,
        "val_fraction_requested": cfg.val_fraction,
        "train_records": len(train_idx),
        "validation_records": len(val_idx),
        "validation_fraction_actual": len(val_idx) / len(train_df),
        "group_count": len(group_keys),
        "canonical_overlap": len(overlap),
        "classes_without_positive_in_validation": [
            CLASS_NAMES[i] for i in np.flatnonzero(val_positive == 0)
        ],
        "train_positive_per_class": train_positive.tolist(),
        "validation_positive_per_class": val_positive.tolist(),
    }
    np.savez_compressed(split_path, train_idx=train_idx, val_idx=val_idx)
    write_json(report_path, report)
    return train_idx, val_idx, report


# ---------------------------------------------------------------------------
# Featuryzacja grafów
# ---------------------------------------------------------------------------


HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
    Chem.rdchem.HybridizationType.S: 1,
    Chem.rdchem.HybridizationType.SP: 2,
    Chem.rdchem.HybridizationType.SP2: 3,
    Chem.rdchem.HybridizationType.SP3: 4,
    Chem.rdchem.HybridizationType.SP3D: 5,
    Chem.rdchem.HybridizationType.SP3D2: 6,
}
CHIRAL_MAP = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
}
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}
STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREONONE: 0,
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ: 2,
    Chem.rdchem.BondStereo.STEREOE: 3,
    Chem.rdchem.BondStereo.STEREOCIS: 4,
    Chem.rdchem.BondStereo.STEREOTRANS: 5,
}
BOND_DIR_MAP = {
    Chem.rdchem.BondDir.NONE: 0,
    Chem.rdchem.BondDir.BEGINWEDGE: 1,
    Chem.rdchem.BondDir.BEGINDASH: 2,
    Chem.rdchem.BondDir.ENDDOWNRIGHT: 3,
    Chem.rdchem.BondDir.ENDUPRIGHT: 4,
    Chem.rdchem.BondDir.EITHERDOUBLE: 5,
}


def molecule_to_graph(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit nie może sparsować SMILES: {smiles!r}")

    atom_cats: list[list[int]] = []
    atom_nums: list[list[float]] = []
    for atom in mol.GetAtoms():
        atomic_num = int(np.clip(atom.GetAtomicNum(), 0, 118))
        degree = int(np.clip(atom.GetDegree(), 0, 10))
        formal_charge = int(np.clip(atom.GetFormalCharge(), -8, 8)) + 8
        hybrid = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 7)
        aromatic = int(atom.GetIsAromatic())
        explicit_h = int(np.clip(atom.GetNumExplicitHs(), 0, 8))
        implicit_h = int(np.clip(atom.GetNumImplicitHs(), 0, 8))
        in_ring = int(atom.IsInRing())
        chirality = CHIRAL_MAP.get(atom.GetChiralTag(), 3)
        atom_cats.append(
            [
                atomic_num,
                degree,
                formal_charge,
                hybrid,
                aromatic,
                explicit_h,
                implicit_h,
                in_ring,
                chirality,
            ]
        )
        atom_nums.append([float(atom.GetMass() / 200.0)])

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_cats: list[list[int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = [
            BOND_TYPE_MAP.get(bond.GetBondType(), 4),
            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            STEREO_MAP.get(bond.GetStereo(), 6),
            BOND_DIR_MAP.get(bond.GetBondDir(), 6),
        ]
        edge_src.extend([begin, end])
        edge_dst.extend([end, begin])
        edge_cats.extend([features, features])

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        raise ValueError(f"Cząsteczka bez atomów: {smiles!r}")

    if edge_src:
        edge_index = np.asarray([edge_src, edge_dst], dtype=np.int32)
        edge_cat = np.asarray(edge_cats, dtype=np.int16)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_cat = np.empty((0, len(EDGE_CARDINALITIES)), dtype=np.int16)

    return {
        "x_cat": np.asarray(atom_cats, dtype=np.int16),
        "x_num": np.asarray(atom_nums, dtype=np.float32),
        "edge_index": edge_index,
        "edge_cat": edge_cat,
        "num_nodes": n_atoms,
    }


def graph_cache_path(cfg: Config, combined_hash: str) -> Path:
    signature = stable_signature(
        {
            "version": GRAPH_FEATURE_VERSION,
            "rdkit": package_version("rdkit"),
            "atom_cardinalities": ATOM_CARDINALITIES,
            "edge_cardinalities": EDGE_CARDINALITIES,
        }
    )
    return cfg.cache_dir / f"graphs_{combined_hash[:12]}_{signature}.pt"


def build_or_load_graphs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    combined_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    path = graph_cache_path(cfg, combined_hash)
    if path.exists() and not cfg.overwrite_cache:
        start = time.perf_counter()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        train_graphs = payload.get("train_graphs", [])
        test_graphs = payload.get("test_graphs", [])
        feature_version = payload.get("feature_version")
        rdkit_version = payload.get("rdkit_version")
        if (
            feature_version == GRAPH_FEATURE_VERSION
            and rdkit_version == package_version("rdkit")
            and len(train_graphs) == len(train_df)
            and len(test_graphs) == len(test_df)
        ):
            elapsed = time.perf_counter() - start
            LOGGER.info("Wczytano cache grafów: %s (%.2f s)", path, elapsed)
            return train_graphs, test_graphs, elapsed
        LOGGER.warning("Cache grafów ma złą wersję lub rozmiar — przebudowuję.")

    start = time.perf_counter()
    limit_seconds = max(0.0, cfg.graph_preprocess_limit_minutes * 60.0)
    train_graphs: list[dict[str, Any]] = []
    test_graphs: list[dict[str, Any]] = []
    chunk_size = 256

    def check_time_limit(processed: int, total: int) -> None:
        elapsed = time.perf_counter() - start
        if (
            cfg.model == "auto"
            and limit_seconds > 0
            and elapsed > limit_seconds
        ):
            raise NeedECFP(
                "Preprocessing grafów przekroczył limit "
                f"{limit_seconds:.0f}s po {processed}/{total} cząsteczek."
            )

    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    try:
        if cfg.feature_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg.feature_workers,
                thread_name_prefix="rdkit-graph",
            )
        total_all = len(train_df) + len(test_df)
        processed_all = 0
        for name, dataframe, target in [
            ("train", train_df, train_graphs),
            ("test", test_df, test_graphs),
        ]:
            smiles_values = dataframe["SMILES"].astype(str).tolist()
            total = len(smiles_values)
            for chunk_start in range(0, total, chunk_size):
                chunk = smiles_values[chunk_start : chunk_start + chunk_size]
                if executor is None:
                    graphs = [molecule_to_graph(smiles) for smiles in chunk]
                else:
                    graphs = list(executor.map(molecule_to_graph, chunk))
                target.extend(graphs)
                processed_all += len(graphs)
                completed = min(chunk_start + len(chunk), total)
                if completed % 5000 < chunk_size or completed == total:
                    LOGGER.info("Grafy %s: %d/%d", name, completed, total)
                check_time_limit(processed_all, total_all)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    build_seconds = time.perf_counter() - start
    payload = {
        "feature_version": GRAPH_FEATURE_VERSION,
        "rdkit_version": package_version("rdkit"),
        "train_graphs": train_graphs,
        "test_graphs": test_graphs,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_start = time.perf_counter()
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    save_seconds = time.perf_counter() - save_start
    elapsed = time.perf_counter() - start

    write_json(
        cfg.report_dir / "graph_preprocessing.json",
        {
            "elapsed_seconds": elapsed,
            "build_seconds": build_seconds,
            "save_seconds": save_seconds,
            "feature_workers": cfg.feature_workers,
            "train_graphs": len(train_graphs),
            "test_graphs": len(test_graphs),
            "train_atoms": {
                "min": min(g["num_nodes"] for g in train_graphs),
                "median": float(np.median([g["num_nodes"] for g in train_graphs])),
                "max": max(g["num_nodes"] for g in train_graphs),
            },
        },
    )
    if cfg.model == "auto" and limit_seconds > 0 and elapsed > limit_seconds:
        raise NeedECFP(
            f"Budowa i zapis cache grafów trwały {elapsed:.1f}s, ponad limit "
            f"{limit_seconds:.1f}s. Cache został zachowany na kolejne uruchomienie."
        )
    return train_graphs, test_graphs, elapsed


class GraphDataset(Dataset):
    def __init__(
        self,
        graphs: Sequence[dict[str, Any]],
        indices: Sequence[int],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        self.graphs = graphs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.num_nodes = np.asarray(
            [graphs[int(i)]["num_nodes"] for i in self.indices], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[dict[str, Any], Optional[np.ndarray]]:
        source_idx = int(self.indices[position])
        label = None if self.labels is None else self.labels[source_idx]
        return self.graphs[source_idx], label


class AtomBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        num_nodes: np.ndarray,
        target_atoms: int,
        shuffle: bool,
        seed: int,
        bucket_size: int = 256,
    ) -> None:
        self.num_nodes = np.asarray(num_nodes, dtype=np.int64)
        self.target_atoms = max(1, int(target_atoms))
        self.shuffle = shuffle
        self.seed = int(seed)
        self.bucket_size = max(1, int(bucket_size))
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        if self.shuffle:
            order = np.argsort(self.num_nodes, kind="stable")
            buckets = [
                order[i : i + self.bucket_size].copy()
                for i in range(0, len(order), self.bucket_size)
            ]
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
            sequence = (
                np.concatenate(buckets)
                if buckets
                else np.empty(0, dtype=np.int64)
            )
        else:
            # Ewaluacja i test muszą zachować dokładną kolejność rekordów.
            sequence = np.arange(len(self.num_nodes), dtype=np.int64)

        batch: list[int] = []
        atom_count = 0
        for position in sequence:
            nodes = int(self.num_nodes[position])
            if batch and atom_count + nodes > self.target_atoms:
                yield batch
                batch = []
                atom_count = 0
            batch.append(int(position))
            atom_count += nodes
            if nodes >= self.target_atoms:
                yield batch
                batch = []
                atom_count = 0
        if batch:
            yield batch

    def __len__(self) -> int:
        total_atoms = int(self.num_nodes.sum())
        return max(1, math.ceil(total_atoms / self.target_atoms))


def collate_graphs(
    items: Sequence[tuple[dict[str, Any], Optional[np.ndarray]]]
) -> dict[str, torch.Tensor]:
    x_cat_parts: list[torch.Tensor] = []
    x_num_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_cat_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offset = 0

    for graph_idx, (graph, label) in enumerate(items):
        x_cat = torch.from_numpy(graph["x_cat"]).long()
        x_num = torch.from_numpy(graph["x_num"]).float()
        edge_index = torch.from_numpy(graph["edge_index"]).long()
        edge_cat = torch.from_numpy(graph["edge_cat"]).long()
        n_nodes = int(graph["num_nodes"])

        x_cat_parts.append(x_cat)
        x_num_parts.append(x_num)
        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + offset)
            edge_cat_parts.append(edge_cat)
        batch_parts.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        if label is not None:
            labels.append(torch.as_tensor(label, dtype=torch.float32))
        offset += n_nodes

    result = {
        "x_cat": torch.cat(x_cat_parts, dim=0),
        "x_num": torch.cat(x_num_parts, dim=0),
        "edge_index": (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_cat": (
            torch.cat(edge_cat_parts, dim=0)
            if edge_cat_parts
            else torch.empty((0, len(EDGE_CARDINALITIES)), dtype=torch.long)
        ),
        "batch": torch.cat(batch_parts, dim=0),
        "num_graphs": len(items),
    }
    if labels:
        result["y"] = torch.stack(labels, dim=0)
    return result


def make_graph_loader(
    dataset: GraphDataset,
    cfg: Config,
    training: bool,
    seed_offset: int = 0,
) -> DataLoader:
    target = (
        cfg.target_atoms_per_batch if training else cfg.eval_atoms_per_batch
    )
    sampler = AtomBudgetBatchSampler(
        dataset.num_nodes,
        target_atoms=target,
        shuffle=training,
        seed=cfg.seed + seed_offset,
        bucket_size=cfg.bucket_size,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_graphs,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# ---------------------------------------------------------------------------
# Model GINE bez PyTorch Geometric
# ---------------------------------------------------------------------------


class CategoricalEncoder(nn.Module):
    def __init__(self, cardinalities: Sequence[int], hidden_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in cardinalities]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.embeddings[0](values[:, 0])
        for col, embedding in enumerate(self.embeddings[1:], start=1):
            output = output + embedding(values[:, col])
        return output


class GINEMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(x)
        else:
            src, dst = edge_index[0], edge_index[1]
            messages = F.silu(x[src] + edge_attr)
            aggregated = torch.zeros_like(x)
            aggregated.index_add_(0, dst, messages)
        return self.mlp((1.0 + self.eps) * x + aggregated)


class GINEBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = GINEMessageLayer(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        update = self.conv(x, edge_index, edge_attr)
        x = self.norm(x + update)
        x = F.silu(x)
        return self.dropout(x)


def global_pool_all(
    x: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    hidden_dim = x.shape[1]
    sum_pool = torch.zeros((num_graphs, hidden_dim), dtype=x.dtype, device=x.device)
    sum_pool.index_add_(0, batch, x)
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(x.dtype)
    mean_pool = sum_pool / counts.unsqueeze(1)

    max_pool = torch.full(
        (num_graphs, hidden_dim), -torch.inf, dtype=x.dtype, device=x.device
    )
    expanded_index = batch.unsqueeze(1).expand(-1, hidden_dim)
    max_pool.scatter_reduce_(
        0, expanded_index, x, reduce="amax", include_self=True
    )
    max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
    return torch.cat([sum_pool, mean_pool, max_pool], dim=1)


class GINEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = CLASS_COUNT,
        num_layers: int = 4,
        hidden_dim: int = 192,
        head_hidden_dim: int = 384,
        gnn_dropout: float = 0.15,
        head_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.atom_encoder = CategoricalEncoder(ATOM_CARDINALITIES, hidden_dim)
        self.atom_numeric = nn.Linear(1, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.edge_encoder = CategoricalEncoder(EDGE_CARDINALITIES, hidden_dim)
        self.blocks = nn.ModuleList(
            [GINEBlock(hidden_dim, gnn_dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.atom_encoder(batch["x_cat"]) + self.atom_numeric(batch["x_num"])
        x = F.silu(self.input_norm(x))
        edge_attr = self.edge_encoder(batch["edge_cat"])
        layer_outputs: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, batch["edge_index"], edge_attr)
            layer_outputs.append(x)
        x = torch.stack(layer_outputs, dim=0).sum(dim=0)
        pooled = global_pool_all(x, batch["batch"], int(batch["num_graphs"]))
        return self.head(pooled)

    def architecture(self) -> dict[str, Any]:
        first_linear = self.head[1]
        return {
            "model": "gine",
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "head_hidden_dim": int(first_linear.out_features),
            "num_classes": self.num_classes,
        }


# ---------------------------------------------------------------------------
# Metryki, progi i DAG
# ---------------------------------------------------------------------------


def f1_per_class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = np.logical_and(truth, pred).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~truth, pred).sum(axis=0).astype(np.float64)
    fn = np.logical_and(truth, ~pred).sum(axis=0).astype(np.float64)
    denominator = 2.0 * tp + fp + fn
    return np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator > 0,
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_per_class(y_true, y_pred).mean())


def class_frequency_bucket(positive_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(positive_counts)
    return np.select(
        [counts < 200, counts < 500, counts < 1000, counts < 5000],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)


def optimize_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold_grid: Sequence[float],
    force_class0: bool,
    frequency_positive_counts: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float32)
    grid = np.asarray(threshold_grid, dtype=np.float32)
    scores = np.zeros((len(grid), y.shape[1]), dtype=np.float64)
    for i, threshold in enumerate(grid):
        pred = probs >= threshold
        if force_class0:
            pred[:, 0] = True
        scores[i] = f1_per_class(y, pred)

    best_idx = scores.argmax(axis=0)
    class_thresholds = grid[best_idx].astype(np.float32)
    validation_positives = y.sum(axis=0)
    frequency_counts = (
        validation_positives
        if frequency_positive_counts is None
        else np.asarray(frequency_positive_counts, dtype=np.int64)
    )
    if frequency_counts.shape != validation_positives.shape:
        raise ValueError("frequency_positive_counts ma nieprawidłowy kształt.")
    buckets = class_frequency_bucket(frequency_counts)
    bucket_thresholds = np.full(5, 0.30, dtype=np.float32)
    for bucket in range(5):
        class_mask = buckets == bucket
        if class_mask.any():
            mean_scores = scores[:, class_mask].mean(axis=1)
            bucket_thresholds[bucket] = grid[int(mean_scores.argmax())]

    regularized = class_thresholds.copy()
    for class_idx, validation_count in enumerate(validation_positives):
        frequency_count = int(frequency_counts[class_idx])
        bucket_threshold = bucket_thresholds[buckets[class_idx]]
        if validation_count == 0:
            regularized[class_idx] = bucket_threshold
        elif frequency_count < 200:
            regularized[class_idx] = (
                0.25 * class_thresholds[class_idx] + 0.75 * bucket_threshold
            )
        elif frequency_count <= 500:
            regularized[class_idx] = (
                0.50 * class_thresholds[class_idx] + 0.50 * bucket_threshold
            )
    if force_class0:
        regularized[0] = 0.0
        class_thresholds[0] = 0.0

    raw_pred = probs >= class_thresholds
    regularized_pred = probs >= regularized
    if force_class0:
        raw_pred[:, 0] = True
        regularized_pred[:, 0] = True
    return {
        "class_thresholds": class_thresholds,
        "regularized_thresholds": regularized,
        "bucket_thresholds": bucket_thresholds,
        "validation_positive_counts": validation_positives,
        "frequency_positive_counts": frequency_counts,
        "macro_f1_raw_thresholds": macro_f1(y, raw_pred),
        "macro_f1_regularized_thresholds": macro_f1(y, regularized_pred),
    }


def apply_thresholds(
    probabilities: np.ndarray, thresholds: np.ndarray, force_class0: bool
) -> np.ndarray:
    pred = (np.asarray(probabilities) >= np.asarray(thresholds)[None, :]).astype(
        np.uint8
    )
    if force_class0:
        pred[:, 0] = 1
    return pred


def evaluate_hierarchy_variants(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    frequency_positive_counts: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    raw_threshold_info = optimize_thresholds(
        y_true,
        raw_probabilities,
        cfg.threshold_grid,
        cfg.force_class0,
        frequency_positive_counts,
    )
    raw_thresholds = raw_threshold_info["regularized_thresholds"]
    monotonic_probs = ontology.monotonic_probabilities(raw_probabilities)
    mono_threshold_info = optimize_thresholds(
        y_true,
        monotonic_probs,
        cfg.threshold_grid,
        cfg.force_class0,
        frequency_positive_counts,
    )
    mono_thresholds = mono_threshold_info["regularized_thresholds"]

    variants: dict[str, dict[str, Any]] = {}
    definitions = {
        "A_raw": (raw_probabilities, raw_thresholds, False, "raw"),
        "B_raw_closure": (raw_probabilities, raw_thresholds, True, "raw"),
        "C_monotonic": (monotonic_probs, mono_thresholds, False, "monotonic"),
        "D_monotonic_closure": (
            monotonic_probs,
            mono_thresholds,
            True,
            "monotonic",
        ),
    }
    for name, (probs, thresholds, closure, probability_mode) in definitions.items():
        pred = apply_thresholds(probs, thresholds, cfg.force_class0)
        if closure:
            pred = ontology.binary_closure(pred)
        per_class = f1_per_class(y_true, pred)
        variants[name] = {
            "macro_f1": float(per_class.mean()),
            "per_class_f1": per_class.tolist(),
            "mean_positive_labels": float(pred.sum(axis=1).mean()),
            "binary_inconsistencies": ontology.inconsistency_counts(pred),
            "probability_inconsistencies": ontology.inconsistency_counts(probs, eps=1e-8),
            "closure": closure,
            "probability_mode": probability_mode,
            "thresholds": thresholds.tolist(),
        }

    best_score = max(item["macro_f1"] for item in variants.values())
    near_best = [
        (name, item)
        for name, item in variants.items()
        if best_score - item["macro_f1"] <= cfg.hierarchy_f1_tolerance
    ]
    near_best.sort(
        key=lambda pair: (
            pair[1]["probability_inconsistencies"]["mean_per_record"],
            pair[1]["binary_inconsistencies"]["mean_per_record"],
            0 if pair[1]["closure"] else 1,
        )
    )
    selected_name, selected = near_best[0]
    return {
        "selected_variant": selected_name,
        "selected": selected,
        "variants": variants,
        "raw_threshold_optimization": jsonable(raw_threshold_info),
        "monotonic_threshold_optimization": jsonable(mono_threshold_info),
    }


def candidate_checkpoint_path(base_path: Path, epoch: int) -> Path:
    """Return a unique path for a validation checkpoint candidate."""
    return base_path.with_name(
        f"{base_path.stem}_epoch{int(epoch):03d}{base_path.suffix}"
    )


def retain_checkpoint_candidate(
    candidates: list[dict[str, Any]],
    candidate: dict[str, Any],
    limit: int,
) -> None:
    """Keep only the strongest candidates according to the cheap online metric."""
    candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            -float(item["quick_metric"]),
            int(item["epoch"]),
        )
    )
    while len(candidates) > max(1, int(limit)):
        removed = candidates.pop()
        removed_path = Path(removed["checkpoint_path"])
        with contextlib.suppress(OSError):
            removed_path.unlink()


def select_best_checkpoint_candidate(
    result: dict[str, Any],
    ontology: Ontology,
    cfg: Config,
    frequency_positive_counts: np.ndarray,
) -> dict[str, Any]:
    """Select a checkpoint after full threshold and hierarchy calibration.

    Training still uses the inexpensive fixed-threshold metric for early stopping,
    but the final checkpoint is selected from a small shortlist using exactly the
    same threshold optimization and DAG variants as the final submission.
    """

    candidates = list(result.pop("checkpoint_candidates", []))
    if not candidates:
        return result

    summaries: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, float, float, float, int], dict[str, Any]]] = []
    for candidate in candidates:
        selection = evaluate_hierarchy_variants(
            result["val_labels"],
            candidate["val_probabilities"],
            ontology,
            cfg,
            frequency_positive_counts=frequency_positive_counts,
        )
        selected = selection["selected"]
        probability_inconsistency = float(
            selected["probability_inconsistencies"]["mean_per_record"]
        )
        binary_inconsistency = float(
            selected["binary_inconsistencies"]["mean_per_record"]
        )
        summary = {
            "epoch": int(candidate["epoch"]),
            "checkpoint_path": str(candidate["checkpoint_path"]),
            "quick_metric": float(candidate["quick_metric"]),
            "calibrated_macro_f1": float(selected["macro_f1"]),
            "selected_hierarchy_variant": selection["selected_variant"],
            "probability_inconsistencies_mean": probability_inconsistency,
            "binary_inconsistencies_mean": binary_inconsistency,
        }
        summaries.append(summary)
        rank_key = (
            -float(selected["macro_f1"]),
            probability_inconsistency,
            binary_inconsistency,
            -float(candidate["quick_metric"]),
            int(candidate["epoch"]),
        )
        ranked.append((rank_key, candidate))

    ranked.sort(key=lambda item: item[0])
    selected_candidate = ranked[0][1]
    selected_path = Path(selected_candidate["checkpoint_path"])
    for candidate in candidates:
        path = Path(candidate["checkpoint_path"])
        if path != selected_path:
            with contextlib.suppress(OSError):
                path.unlink()

    result.update(
        {
            "checkpoint_path": selected_path,
            "best_epoch": int(selected_candidate["epoch"]),
            "best_working_metric": float(selected_candidate["quick_metric"]),
            "val_probabilities": np.asarray(
                selected_candidate["val_probabilities"], dtype=np.float32
            ),
            "checkpoint_selection": {
                "selection_method": "calibrated_macro_f1_then_probability_consistency",
                "candidate_count": len(candidates),
                "selected_checkpoint": str(selected_path),
                "selected_epoch": int(selected_candidate["epoch"]),
                "candidates": summaries,
            },
        }
    )
    LOGGER.info(
        "Wybrano checkpoint z epoki %d po kalibracji progów (%d kandydatów).",
        result["best_epoch"],
        len(candidates),
    )
    return result


# ---------------------------------------------------------------------------
# Trening i inferencja GNN
# ---------------------------------------------------------------------------


def compute_pos_weight(labels: np.ndarray, clip_max: float) -> torch.Tensor:
    positives = labels.sum(axis=0).astype(np.float64)
    negatives = len(labels) - positives
    weights = np.sqrt((negatives + 1.0) / (positives + 1.0))
    weights = np.clip(weights, 1.0, clip_max).astype(np.float32)
    return torch.from_numpy(weights)


def train_graph_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, dict[str, float]]:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    step_count = 0
    atom_count = 0
    started = time.perf_counter()
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.use_bf16):
            logits = model(batch)
            loss = criterion(logits, batch["y"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Niefinity loss: {float(loss.detach().cpu())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("Niefinity gradient norm.")
        optimizer.step()
        batch_size = int(batch["y"].shape[0])
        loss_sum += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
        atom_count += int(batch["x_cat"].shape[0])
        step_count += 1
    elapsed = time.perf_counter() - started
    return loss_sum / max(sample_count, 1), {
        "seconds": elapsed,
        "steps": step_count,
        "samples": sample_count,
        "atoms": atom_count,
        "atoms_per_second": atom_count / max(elapsed, 1e-9),
    }


@torch.inference_mode()
def predict_graph_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    with_labels: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, cfg.use_bf16):
            logits = model(batch)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        if with_labels:
            labels.append(batch["y"].float().cpu().numpy())
    probs = np.concatenate(probabilities, axis=0)
    y = np.concatenate(labels, axis=0) if labels else None
    return probs, y


def save_model_checkpoint(
    path: Path,
    model: GINEClassifier,
    epoch: int,
    metric: float,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "architecture": model.architecture(),
        "epoch": int(epoch),
        "metric": float(metric),
        "extra": extra or {},
    }
    torch.save(payload, path)


def build_gine(
    cfg: Config,
    num_layers: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    head_hidden_dim: Optional[int] = None,
) -> GINEClassifier:
    return GINEClassifier(
        num_classes=CLASS_COUNT,
        num_layers=num_layers or cfg.num_layers,
        hidden_dim=hidden_dim or cfg.hidden_dim,
        head_hidden_dim=head_hidden_dim or cfg.head_hidden_dim,
        gnn_dropout=cfg.gnn_dropout,
        head_dropout=cfg.head_dropout,
    )


def fit_gnn_validation(
    graphs: Sequence[dict[str, Any]],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    architecture: tuple[int, int, int],
    auto_benchmark: bool,
    checkpoint_name: str,
    max_epochs: Optional[int] = None,
) -> dict[str, Any]:
    num_layers, hidden_dim, head_hidden_dim = architecture
    effective_max_epochs = max(1, int(max_epochs or cfg.max_epochs))
    model = build_gine(cfg, num_layers, hidden_dim, head_hidden_dim).to(device)
    train_ds = GraphDataset(graphs, train_idx, labels)
    val_ds = GraphDataset(graphs, val_idx, labels)
    train_loader = make_graph_loader(train_ds, cfg, training=True, seed_offset=11)
    val_loader = make_graph_loader(val_ds, cfg, training=False, seed_offset=12)

    pos_weight = compute_pos_weight(labels[train_idx], cfg.pos_weight_clip).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    checkpoint_path = cfg.checkpoint_dir / checkpoint_name
    checkpoint_candidates: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    no_improvement = 0
    epoch_times: list[float] = []
    last_val_y: Optional[np.ndarray] = None

    for epoch in range(1, effective_max_epochs + 1):
        if epoch > 2 and epoch_times:
            avg_epoch = float(np.mean(epoch_times[-3:]))
            estimated_final = avg_epoch * max(best_epoch, 2) * 1.30 + 180.0
            if budget.remaining < budget.buffer_seconds + estimated_final:
                LOGGER.warning(
                    "Kończę walidacyjny trening wcześniej, aby zachować budżet finalny."
                )
                break
        loss, timing = train_graph_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        epoch_times.append(timing["seconds"])
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss,
            **timing,
            "budget_remaining_seconds": budget.remaining,
        }
        LOGGER.info(
            "GNN epoka %d/%d | loss=%.5f | %.2f s | %.0f atomów/s",
            epoch,
            effective_max_epochs,
            loss,
            timing["seconds"],
            timing["atoms_per_second"],
        )

        should_evaluate = epoch % cfg.evaluation_every == 0 or epoch == effective_max_epochs
        if should_evaluate:
            probs, val_y = predict_graph_loader(
                model, val_loader, device, cfg, with_labels=True
            )
            assert val_y is not None
            last_val_y = val_y
            quick_pred = probs >= cfg.working_threshold
            if cfg.force_class0:
                quick_pred[:, 0] = True
            metric = macro_f1(val_y, quick_pred)
            record["validation_macro_f1_working_threshold"] = metric
            LOGGER.info("Walidacja macro-F1@%.3f = %.6f", cfg.working_threshold, metric)
            candidate_floor = min(
                (float(item["quick_metric"]) for item in checkpoint_candidates),
                default=-float("inf"),
            )
            if (
                len(checkpoint_candidates) < cfg.checkpoint_candidates
                or metric > candidate_floor + 1e-8
            ):
                candidate_path = candidate_checkpoint_path(checkpoint_path, epoch)
                save_model_checkpoint(
                    candidate_path,
                    model,
                    epoch,
                    metric,
                    extra={"architecture": architecture},
                )
                retain_checkpoint_candidate(
                    checkpoint_candidates,
                    {
                        "checkpoint_path": candidate_path,
                        "epoch": epoch,
                        "quick_metric": metric,
                        "val_probabilities": probs.astype(np.float32, copy=True),
                    },
                    cfg.checkpoint_candidates,
                )
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch
                no_improvement = 0
            else:
                no_improvement += 1
        history.append(record)

        if epoch == 2 and auto_benchmark:
            average = float(np.mean(epoch_times[:2]))
            LOGGER.info("Benchmark GNN: średnio %.2f s/epokę", average)
            if average > cfg.benchmark_slow_seconds:
                raise NeedECFP(
                    f"Epoka GNN trwa {average:.2f}s > {cfg.benchmark_slow_seconds:.2f}s."
                )
            if (
                average > cfg.benchmark_fast_seconds
                and (num_layers, hidden_dim) != (3, 128)
            ):
                raise NeedSmallGNN(
                    f"Epoka GNN trwa {average:.2f}s — przejście na 3x128."
                )
        if no_improvement >= cfg.patience:
            LOGGER.info("Early stopping po %d ocenach bez poprawy.", no_improvement)
            break

    if best_epoch == 0:
        probs, val_y = predict_graph_loader(model, val_loader, device, cfg, True)
        assert val_y is not None
        last_val_y = val_y
        quick_pred = probs >= cfg.working_threshold
        if cfg.force_class0:
            quick_pred[:, 0] = True
        best_metric = macro_f1(val_y, quick_pred)
        best_epoch = max(1, len(history))
        candidate_path = candidate_checkpoint_path(checkpoint_path, best_epoch)
        save_model_checkpoint(candidate_path, model, best_epoch, best_metric)
        retain_checkpoint_candidate(
            checkpoint_candidates,
            {
                "checkpoint_path": candidate_path,
                "epoch": best_epoch,
                "quick_metric": best_metric,
                "val_probabilities": probs.astype(np.float32, copy=True),
            },
            cfg.checkpoint_candidates,
        )

    if last_val_y is None or not checkpoint_candidates:
        raise TimeBudgetExceeded("Nie udało się utworzyć checkpointu walidacyjnego GNN.")
    del model
    clear_cuda_cache()
    return {
        "checkpoint_path": Path(checkpoint_candidates[0]["checkpoint_path"]),
        "best_epoch": best_epoch,
        "best_working_metric": best_metric,
        "history": history,
        "epoch_times": epoch_times,
        "val_probabilities": checkpoint_candidates[0]["val_probabilities"],
        "val_labels": last_val_y.astype(np.uint8),
        "architecture": architecture,
        "checkpoint_candidates": checkpoint_candidates,
    }


def estimate_full_training_seconds(
    validation_epoch_times: Sequence[float],
    epochs: int,
    val_fraction: float,
    safety_factor: float,
    inference_reserve_seconds: float,
) -> float:
    times = [float(value) for value in validation_epoch_times if value > 0]
    if not times:
        return float("inf")
    representative = float(np.median(times[-min(3, len(times)) :]))
    full_train_scale = 1.0 / max(1e-6, 1.0 - val_fraction)
    return (
        representative
        * full_train_scale
        * max(1, int(epochs))
        * max(1.0, safety_factor)
        + max(0.0, inference_reserve_seconds)
    )


def predict_gnn_validation_checkpoint(
    test_graphs: Sequence[dict[str, Any]],
    validation_result: dict[str, Any],
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer on test with the checkpoint whose thresholds were calibrated."""
    checkpoint_path = Path(validation_result["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture_payload = payload.get("architecture")
    if isinstance(architecture_payload, dict):
        architecture = (
            int(architecture_payload["num_layers"]),
            int(architecture_payload["hidden_dim"]),
            int(architecture_payload["head_hidden_dim"]),
        )
    else:
        architecture = tuple(
            int(value) for value in validation_result["architecture"]
        )
    model = build_gine(cfg, *architecture).to(device)
    model.load_state_dict(payload["model_state"])
    test_ds = GraphDataset(
        test_graphs,
        np.arange(len(test_graphs), dtype=np.int64),
        labels=None,
    )
    test_loader = make_graph_loader(test_ds, cfg, training=False, seed_offset=32)
    started = time.perf_counter()
    probabilities, _ = predict_graph_loader(model, test_loader, device, cfg, False)
    elapsed = time.perf_counter() - started
    del model
    clear_cuda_cache()
    return probabilities, {
        "completed_full_training": False,
        "source": "validation_checkpoint",
        "checkpoint": str(checkpoint_path),
        "epochs_completed": int(payload.get("epoch", validation_result["best_epoch"])),
        "architecture": list(architecture),
        "inference_seconds": elapsed,
    }


def train_gnn_full_and_predict(
    train_graphs: Sequence[dict[str, Any]],
    test_graphs: Sequence[dict[str, Any]],
    labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    architecture: tuple[int, int, int],
    epochs: int,
    validation_epoch_times: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    requested_epochs = max(1, int(epochs))
    estimated_seconds = estimate_full_training_seconds(
        validation_epoch_times,
        requested_epochs,
        cfg.val_fraction,
        cfg.final_time_safety_factor,
        cfg.final_inference_reserve_seconds,
    )
    if budget.usable_remaining < estimated_seconds:
        raise TimeBudgetExceeded(
            "Za mało czasu na pełny finalny trening GNN: "
            f"potrzeba około {estimated_seconds:.1f}s, dostępne "
            f"{budget.usable_remaining:.1f}s."
        )

    set_seed(cfg.seed + 1000)
    model = build_gine(cfg, *architecture).to(device)
    full_indices = np.arange(len(train_graphs), dtype=np.int64)
    train_ds = GraphDataset(train_graphs, full_indices, labels)
    test_ds = GraphDataset(
        test_graphs, np.arange(len(test_graphs), dtype=np.int64), labels=None
    )
    train_loader = make_graph_loader(train_ds, cfg, training=True, seed_offset=31)
    test_loader = make_graph_loader(test_ds, cfg, training=False, seed_offset=32)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(labels, cfg.pos_weight_clip).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history: list[dict[str, Any]] = []
    recent_epoch_times: list[float] = []
    for epoch in range(1, requested_epochs + 1):
        if recent_epoch_times:
            remaining_epochs = requested_epochs - epoch + 1
            representative = float(np.median(recent_epoch_times[-3:]))
            required_remaining = (
                representative
                * remaining_epochs
                * cfg.final_time_safety_factor
                + cfg.final_inference_reserve_seconds
            )
            if budget.usable_remaining < required_remaining:
                raise TimeBudgetExceeded(
                    "Rzeczywisty czas finalnego GNN jest dłuższy od estymacji; "
                    f"nie można bezpiecznie ukończyć wszystkich {requested_epochs} epok."
                )
        loss, timing = train_graph_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        recent_epoch_times.append(timing["seconds"])
        history.append({"epoch": epoch, "loss": loss, **timing})
        LOGGER.info(
            "Finalny GNN %d/%d | loss=%.5f | %.2f s",
            epoch,
            requested_epochs,
            loss,
            timing["seconds"],
        )

    final_path = cfg.checkpoint_dir / "gine_final.pt"
    save_model_checkpoint(
        final_path,
        model,
        requested_epochs,
        metric=float("nan"),
        extra={"completed_full_training": True},
    )
    test_probs, _ = predict_graph_loader(model, test_loader, device, cfg, False)
    return test_probs, {
        "epochs_requested": requested_epochs,
        "epochs_completed": requested_epochs,
        "completed_full_training": True,
        "estimated_seconds_before_start": estimated_seconds,
        "history": history,
        "checkpoint": str(final_path),
    }


# ---------------------------------------------------------------------------
# Fallback ECFP
# ---------------------------------------------------------------------------


def rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    return np.asarray(
        [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Crippen.MolLogP(mol),
            mol.GetNumAtoms(),
            Lipinski.NumHeteroatoms(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumRotatableBonds(mol),
            Chem.GetFormalCharge(mol),
            len(Chem.GetMolFrags(mol)),
        ],
        dtype=np.float32,
    )


def ecfp_feature_signature(cfg: Config) -> str:
    payload = {
        "version": ECFP_FEATURE_VERSION,
        "radius": int(cfg.ecfp_radius),
        "bits": int(cfg.ecfp_bits),
        "include_chirality": True,
        "descriptors": bool(cfg.ecfp_descriptors),
        "rdkit": package_version("rdkit"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def ecfp_cache_paths(cfg: Config, combined_hash: str) -> dict[str, Path]:
    signature = ecfp_feature_signature(cfg)
    prefix = cfg.cache_dir / (
        f"ecfp_{combined_hash[:12]}_r{cfg.ecfp_radius}_b{cfg.ecfp_bits}_"
        f"d{int(cfg.ecfp_descriptors)}_{signature}"
    )
    return {
        "train_bits": Path(str(prefix) + "_train_bits.npy"),
        "test_bits": Path(str(prefix) + "_test_bits.npy"),
        "train_desc": Path(str(prefix) + "_train_desc.npy"),
        "test_desc": Path(str(prefix) + "_test_desc.npy"),
        "metadata": Path(str(prefix) + "_metadata.json"),
    }


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def build_or_load_ecfp(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    combined_hash: str,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    paths = ecfp_cache_paths(cfg, combined_hash)
    required = [paths["train_bits"], paths["test_bits"], paths["metadata"]]
    if cfg.ecfp_descriptors:
        required.extend([paths["train_desc"], paths["test_desc"]])
    if all(path.exists() for path in required) and not cfg.overwrite_cache:
        train_bits = np.load(paths["train_bits"], mmap_mode="r", allow_pickle=False)
        test_bits = np.load(paths["test_bits"], mmap_mode="r", allow_pickle=False)
        expected_train_shape = (len(train_df), cfg.ecfp_bits)
        expected_test_shape = (len(test_df), cfg.ecfp_bits)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        cache_valid = (
            tuple(train_bits.shape) == expected_train_shape
            and tuple(test_bits.shape) == expected_test_shape
            and metadata.get("feature_signature") == ecfp_feature_signature(cfg)
            and metadata.get("combined_hash") == combined_hash
        )
        train_desc: Optional[np.ndarray] = None
        test_desc: Optional[np.ndarray] = None
        if cache_valid and cfg.ecfp_descriptors:
            train_desc = np.load(
                paths["train_desc"], mmap_mode="r", allow_pickle=False
            )
            test_desc = np.load(
                paths["test_desc"], mmap_mode="r", allow_pickle=False
            )
            cache_valid = (
                tuple(train_desc.shape) == (len(train_df), 9)
                and tuple(test_desc.shape) == (len(test_df), 9)
            )
        if cache_valid:
            LOGGER.info(
                "Wczytano cache ECFP: radius=%d, bits=%d, descriptors=%s.",
                cfg.ecfp_radius,
                cfg.ecfp_bits,
                cfg.ecfp_descriptors,
            )
            return train_bits, test_bits, train_desc, test_desc
        LOGGER.warning("Cache ECFP ma niezgodną konfigurację lub kształt — przebudowuję.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.ecfp_radius,
        fpSize=cfg.ecfp_bits,
        includeChirality=True,
    )

    def build(
        dataframe: pd.DataFrame, split_name: str
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        bits = np.empty((len(dataframe), cfg.ecfp_bits), dtype=np.uint8)
        descs = (
            np.empty((len(dataframe), 9), dtype=np.float32)
            if cfg.ecfp_descriptors
            else None
        )
        for i, smiles in enumerate(dataframe["SMILES"].astype(str)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Nieprawidłowy SMILES w {split_name}: {smiles!r}")
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, bits[i])
            if descs is not None:
                descs[i] = rdkit_descriptors(mol)
            if (i + 1) % 5000 == 0 or i + 1 == len(dataframe):
                LOGGER.info("ECFP %s: %d/%d", split_name, i + 1, len(dataframe))
        return bits, descs

    train_bits, train_desc = build(train_df, "train")
    test_bits, test_desc = build(test_df, "test")
    atomic_save_npy(paths["train_bits"], train_bits)
    atomic_save_npy(paths["test_bits"], test_bits)
    if train_desc is not None and test_desc is not None:
        atomic_save_npy(paths["train_desc"], train_desc)
        atomic_save_npy(paths["test_desc"], test_desc)
    write_json(
        paths["metadata"],
        {
            "feature_version": ECFP_FEATURE_VERSION,
            "feature_signature": ecfp_feature_signature(cfg),
            "combined_hash": combined_hash,
            "radius": cfg.ecfp_radius,
            "bits": cfg.ecfp_bits,
            "descriptors": cfg.ecfp_descriptors,
            "rdkit": package_version("rdkit"),
            "train_shape": list(train_bits.shape),
            "test_shape": list(test_bits.shape),
        },
    )
    return train_bits, test_bits, train_desc, test_desc


class ArrayDataset(Dataset):
    def __init__(
        self,
        bits: np.ndarray,
        indices: Sequence[int],
        labels: Optional[np.ndarray] = None,
        descriptors: Optional[np.ndarray] = None,
        descriptor_mean: Optional[np.ndarray] = None,
        descriptor_std: Optional[np.ndarray] = None,
    ) -> None:
        self.bits = bits
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.descriptors = descriptors
        self.descriptor_mean = descriptor_mean
        self.descriptor_std = descriptor_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[np.ndarray, Optional[np.ndarray]]:
        idx = int(self.indices[position])
        x = np.asarray(self.bits[idx], dtype=np.float32)
        if self.descriptors is not None:
            desc = np.asarray(self.descriptors[idx], dtype=np.float32)
            if self.descriptor_mean is not None and self.descriptor_std is not None:
                desc = (desc - self.descriptor_mean) / self.descriptor_std
            x = np.concatenate([x, desc], axis=0)
        y = None if self.labels is None else self.labels[idx].astype(np.float32)
        return x, y


def collate_arrays(
    items: Sequence[tuple[np.ndarray, Optional[np.ndarray]]]
) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(np.stack([item[0] for item in items])).float()
    result = {"x": x}
    if items[0][1] is not None:
        result["y"] = torch.from_numpy(np.stack([item[1] for item in items])).float()
    return result


class ECFPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.variant = "mlp"
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, CLASS_COUNT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ECFPLinear(nn.Module):
    """Liniowy klasyfikator wieloetykietowy, równoważny 500 modelom OvR."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 0
        self.variant = "linear"
        self.linear = nn.Linear(input_dim, CLASS_COUNT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_ecfp_model(input_dim: int, cfg: Config, variant: str) -> nn.Module:
    if variant == "mlp":
        return ECFPMLP(input_dim, cfg.ecfp_hidden_dim, cfg.head_dropout)
    if variant == "linear":
        return ECFPLinear(input_dim)
    raise ValueError(f"Nieznany wariant ECFP: {variant!r}")


def make_array_loader(
    dataset: ArrayDataset, batch_size: int, cfg: Config, shuffle: bool
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_arrays,
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    kwargs["generator"] = generator
    return DataLoader(**kwargs)


def train_ecfp_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    start = time.perf_counter()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.use_bf16):
            logits = model(x)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Niefinity loss ECFP.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(x)
        total_count += len(x)
    return total_loss / max(total_count, 1), time.perf_counter() - start


@torch.inference_mode()
def predict_ecfp(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    with_labels: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        with autocast_context(device, cfg.use_bf16):
            logits = model(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
        if with_labels:
            ys.append(batch["y"].numpy())
    return np.concatenate(probs), np.concatenate(ys) if ys else None


def predict_ecfp_validation_checkpoint(
    test_bits: np.ndarray,
    test_desc: Optional[np.ndarray],
    validation_result: dict[str, Any],
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Awaryjna inferencja modelem wytrenowanym na części treningowej splitu."""
    checkpoint_path = Path(validation_result["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = str(payload.get("variant", validation_result["model_variant"]))
    input_dim = int(payload.get("input_dim", validation_result["input_dim"]))
    model = build_ecfp_model(input_dim, cfg, variant)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    descriptor_mean = payload.get("descriptor_mean")
    descriptor_std = payload.get("descriptor_std")
    test_ds = ArrayDataset(
        test_bits,
        np.arange(len(test_bits), dtype=np.int64),
        labels=None,
        descriptors=test_desc,
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
    )
    test_loader = make_array_loader(test_ds, cfg.ecfp_eval_batch_size, cfg, False)
    started = time.perf_counter()
    probabilities, _ = predict_ecfp(model, test_loader, device, cfg, False)
    elapsed = time.perf_counter() - started
    del model
    clear_cuda_cache()
    return probabilities, {
        "model_variant": variant,
        "completed_full_training": False,
        "source": "validation_checkpoint",
        "checkpoint": str(checkpoint_path),
        "epochs_completed": int(payload.get("epoch", validation_result["best_epoch"])),
        "inference_seconds": elapsed,
    }


def _fit_ecfp_validation_variant(
    train_bits: np.ndarray,
    train_desc: Optional[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    variant: str,
) -> dict[str, Any]:
    descriptor_mean = descriptor_std = None
    if train_desc is not None:
        descriptor_mean = np.asarray(train_desc[train_idx].mean(axis=0), dtype=np.float32)
        descriptor_std = np.asarray(train_desc[train_idx].std(axis=0), dtype=np.float32)
        descriptor_std = np.where(descriptor_std < 1e-6, 1.0, descriptor_std)
    input_dim = cfg.ecfp_bits + (9 if train_desc is not None else 0)
    train_ds = ArrayDataset(
        train_bits,
        train_idx,
        labels,
        train_desc,
        descriptor_mean,
        descriptor_std,
    )
    val_ds = ArrayDataset(
        train_bits,
        val_idx,
        labels,
        train_desc,
        descriptor_mean,
        descriptor_std,
    )
    train_loader = make_array_loader(train_ds, cfg.ecfp_batch_size, cfg, True)
    val_loader = make_array_loader(val_ds, cfg.ecfp_eval_batch_size, cfg, False)
    model = build_ecfp_model(input_dim, cfg, variant).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(labels[train_idx], cfg.pos_weight_clip).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    checkpoint = cfg.checkpoint_dir / f"ecfp_{variant}_best_validation.pt"
    checkpoint_candidates: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    last_val_y: Optional[np.ndarray] = None

    for epoch in range(1, cfg.ecfp_max_epochs + 1):
        if epoch > 1 and budget.usable_remaining < 300:
            LOGGER.warning("Kończę ECFP validation z powodu limitu czasu.")
            break
        loss, seconds = train_ecfp_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        probs, val_y = predict_ecfp(model, val_loader, device, cfg, True)
        assert val_y is not None
        last_val_y = val_y
        pred = probs >= cfg.working_threshold
        if cfg.force_class0:
            pred[:, 0] = True
        metric = macro_f1(val_y, pred)
        LOGGER.info(
            "ECFP-%s epoka %d/%d | loss=%.5f | macro-F1=%.6f | %.2f s",
            variant,
            epoch,
            cfg.ecfp_max_epochs,
            loss,
            metric,
            seconds,
        )
        history.append(
            {"epoch": epoch, "loss": loss, "macro_f1": metric, "seconds": seconds}
        )
        candidate_floor = min(
            (float(item["quick_metric"]) for item in checkpoint_candidates),
            default=-float("inf"),
        )
        if (
            len(checkpoint_candidates) < cfg.checkpoint_candidates
            or metric > candidate_floor + 1e-8
        ):
            candidate_path = candidate_checkpoint_path(checkpoint, epoch)
            torch.save(
                {
                    "model_state": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "input_dim": input_dim,
                    "hidden_dim": cfg.ecfp_hidden_dim if variant == "mlp" else 0,
                    "variant": variant,
                    "epoch": epoch,
                    "metric": metric,
                    "descriptor_mean": descriptor_mean,
                    "descriptor_std": descriptor_std,
                },
                candidate_path,
            )
            retain_checkpoint_candidate(
                checkpoint_candidates,
                {
                    "checkpoint_path": candidate_path,
                    "epoch": epoch,
                    "quick_metric": metric,
                    "val_probabilities": probs.astype(np.float32, copy=True),
                },
                cfg.checkpoint_candidates,
            )
        if metric > best_metric + 1e-8:
            best_metric = metric
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= cfg.ecfp_patience:
            break

    if best_epoch == 0 or not checkpoint_candidates or last_val_y is None:
        raise TimeBudgetExceeded("Nie udało się ukończyć ani jednej epoki ECFP validation.")
    del model
    clear_cuda_cache()
    return {
        "model_variant": variant,
        "best_epoch": best_epoch,
        "best_working_metric": best_metric,
        "checkpoint_path": Path(checkpoint_candidates[0]["checkpoint_path"]),
        "val_probabilities": checkpoint_candidates[0]["val_probabilities"],
        "val_labels": last_val_y.astype(np.uint8),
        "descriptor_mean": descriptor_mean,
        "descriptor_std": descriptor_std,
        "history": history,
        "epoch_times": [float(item["seconds"]) for item in history],
        "input_dim": input_dim,
        "checkpoint_candidates": checkpoint_candidates,
    }


def fit_ecfp_validation(
    train_bits: np.ndarray,
    train_desc: Optional[np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
) -> dict[str, Any]:
    variants = [cfg.ecfp_variant] if cfg.ecfp_variant != "auto" else ["mlp", "linear"]
    last_error: Optional[BaseException] = None
    for index, variant in enumerate(variants):
        try:
            return _fit_ecfp_validation_variant(
                train_bits,
                train_desc,
                labels,
                train_idx,
                val_idx,
                cfg,
                device,
                budget,
                variant,
            )
        except (FloatingPointError, TimeBudgetExceeded) as exc:
            if index + 1 >= len(variants):
                raise
            message = str(exc)
            last_error = RuntimeError(message)
            LOGGER.warning(
                "ECFP-%s nie powiódł się (%s). Przechodzę na wariant %s.",
                variant,
                message,
                variants[index + 1],
            )
            del exc
            clear_cuda_cache()
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or index + 1 >= len(variants):
                raise
            message = str(exc)
            last_error = RuntimeError(message)
            LOGGER.warning(
                "ECFP-%s przekroczył pamięć GPU. Przechodzę na wariant %s.",
                variant,
                variants[index + 1],
            )
            del exc
            clear_cuda_cache()
    assert last_error is not None
    raise RuntimeError("Nie udało się wytrenować żadnego wariantu ECFP.") from last_error


def train_ecfp_full_and_predict(
    train_bits: np.ndarray,
    test_bits: np.ndarray,
    train_desc: Optional[np.ndarray],
    test_desc: Optional[np.ndarray],
    labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    epochs: int,
    descriptor_mean: Optional[np.ndarray],
    descriptor_std: Optional[np.ndarray],
    model_variant: str,
    validation_epoch_times: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    requested_epochs = max(1, int(epochs))
    estimated_seconds = estimate_full_training_seconds(
        validation_epoch_times,
        requested_epochs,
        cfg.val_fraction,
        cfg.final_time_safety_factor,
        cfg.final_inference_reserve_seconds,
    )
    if budget.usable_remaining < estimated_seconds:
        raise TimeBudgetExceeded(
            "Za mało czasu na pełny finalny trening ECFP: "
            f"potrzeba około {estimated_seconds:.1f}s, dostępne "
            f"{budget.usable_remaining:.1f}s."
        )

    set_seed(cfg.seed + 2000)
    input_dim = cfg.ecfp_bits + (9 if train_desc is not None else 0)
    model = build_ecfp_model(input_dim, cfg, model_variant).to(device)
    full_idx = np.arange(len(train_bits))
    test_idx = np.arange(len(test_bits))
    if train_desc is not None:
        descriptor_mean = np.asarray(train_desc.mean(axis=0), dtype=np.float32)
        descriptor_std = np.asarray(train_desc.std(axis=0), dtype=np.float32)
        descriptor_std = np.where(descriptor_std < 1e-6, 1.0, descriptor_std)
    train_ds = ArrayDataset(
        train_bits,
        full_idx,
        labels,
        train_desc,
        descriptor_mean,
        descriptor_std,
    )
    test_ds = ArrayDataset(
        test_bits,
        test_idx,
        None,
        test_desc,
        descriptor_mean,
        descriptor_std,
    )
    train_loader = make_array_loader(train_ds, cfg.ecfp_batch_size, cfg, True)
    test_loader = make_array_loader(test_ds, cfg.ecfp_eval_batch_size, cfg, False)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=compute_pos_weight(labels, cfg.pos_weight_clip).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    history: list[dict[str, Any]] = []
    recent_epoch_times: list[float] = []
    for epoch in range(1, requested_epochs + 1):
        if recent_epoch_times:
            remaining_epochs = requested_epochs - epoch + 1
            representative = float(np.median(recent_epoch_times[-3:]))
            required_remaining = (
                representative
                * remaining_epochs
                * cfg.final_time_safety_factor
                + cfg.final_inference_reserve_seconds
            )
            if budget.usable_remaining < required_remaining:
                raise TimeBudgetExceeded(
                    "Rzeczywisty czas finalnego ECFP jest dłuższy od estymacji; "
                    f"nie można ukończyć wszystkich {requested_epochs} epok."
                )
        loss, seconds = train_ecfp_epoch(
            model, train_loader, optimizer, criterion, device, cfg
        )
        recent_epoch_times.append(seconds)
        history.append({"epoch": epoch, "loss": loss, "seconds": seconds})
        LOGGER.info(
            "Finalny ECFP-%s %d/%d | loss=%.5f | %.2f s",
            model_variant,
            epoch,
            requested_epochs,
            loss,
            seconds,
        )
    final_path = cfg.checkpoint_dir / f"ecfp_{model_variant}_final.pt"
    torch.save(
        {
            "model_state": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "input_dim": input_dim,
            "hidden_dim": cfg.ecfp_hidden_dim if model_variant == "mlp" else 0,
            "variant": model_variant,
            "epochs_completed": requested_epochs,
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "completed_full_training": True,
        },
        final_path,
    )
    test_probs, _ = predict_ecfp(model, test_loader, device, cfg, False)
    return test_probs, {
        "model_variant": model_variant,
        "epochs_requested": requested_epochs,
        "epochs_completed": requested_epochs,
        "completed_full_training": True,
        "estimated_seconds_before_start": estimated_seconds,
        "history": history,
        "checkpoint": str(final_path),
    }


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def generate_and_validate_submission(
    test_df: pd.DataFrame,
    sample_df: Optional[pd.DataFrame],
    predictions: np.ndarray,
    ontology: Ontology,
    selected_variant: dict[str, Any],
    cfg: Config,
    output_path: Optional[Path] = None,
    report_filename: str = "submission_validation.json",
) -> dict[str, Any]:
    expected_columns = ["mol_id", "SMILES", *CLASS_NAMES]
    target_path = output_path or cfg.submission_path
    pred = np.asarray(predictions)
    if pred.shape != (len(test_df), CLASS_COUNT):
        raise ValueError(
            f"Predykcje mają kształt {pred.shape}, oczekiwano "
            f"({len(test_df)}, {CLASS_COUNT})."
        )
    label_frame = pd.DataFrame(
        pred.astype(np.int8, copy=False), columns=CLASS_NAMES, index=test_df.index
    )
    submission = pd.concat(
        [test_df[["mol_id", "SMILES"]].copy(), label_frame], axis=1
    )
    if sample_df is not None:
        submission = submission[list(sample_df.columns)]
    else:
        submission = submission[expected_columns]

    issues: list[str] = []
    if len(submission) != len(test_df):
        issues.append("Nieprawidłowa liczba wierszy.")
    if submission.shape[1] != 502:
        issues.append(f"Nieprawidłowa liczba kolumn: {submission.shape[1]} zamiast 502.")
    if list(submission.columns) != expected_columns:
        issues.append("Nieprawidłowe nazwy lub kolejność kolumn.")
    if not submission["mol_id"].equals(test_df["mol_id"]):
        issues.append("mol_id nie zgadzają się z test.")
    if not submission["SMILES"].equals(test_df["SMILES"]):
        issues.append("SMILES nie zgadzają się z test.")
    values = submission[CLASS_NAMES].to_numpy()
    if not all(
        pd.api.types.is_integer_dtype(submission[column].dtype) for column in CLASS_NAMES
    ):
        issues.append("Kolumny predykcji nie mają typu całkowitego.")
    if pd.isna(values).any() or not np.isfinite(values).all():
        issues.append("Predykcje zawierają NaN lub wartości nieskończone.")
    if not np.isin(values, [0, 1]).all():
        issues.append("Predykcje zawierają wartości inne niż 0/1.")
    if cfg.force_class0 and not (values[:, 0] == 1).all():
        issues.append("class_0 nie jest równa 1 dla wszystkich rekordów.")
    binary_inconsistencies = ontology.inconsistency_counts(values)
    if selected_variant["closure"] and binary_inconsistencies["total"] != 0:
        issues.append("Po closure pozostały naruszenia DAG.")

    if issues:
        raise ValueError("Walidacja submission nie powiodła się:\n- " + "\n- ".join(issues))
    write_parquet(submission, target_path)
    reloaded = read_parquet(target_path)
    reload_issues: list[str] = []
    if list(reloaded.columns) != expected_columns or reloaded.shape != submission.shape:
        reload_issues.append("Plik po ponownym wczytaniu ma zły schemat lub kształt.")
    elif not reloaded["mol_id"].equals(test_df["mol_id"]):
        reload_issues.append("Plik po ponownym wczytaniu ma inną kolejność mol_id.")
    elif not reloaded["SMILES"].equals(test_df["SMILES"]):
        reload_issues.append("Plik po ponownym wczytaniu ma inne SMILES.")
    else:
        reloaded_values = reloaded[CLASS_NAMES].to_numpy()
        if not np.array_equal(reloaded_values, values):
            reload_issues.append("Predykcje zmieniły się po zapisie i ponownym wczytaniu.")
        if not all(
            pd.api.types.is_integer_dtype(reloaded[column].dtype)
            for column in CLASS_NAMES
        ):
            reload_issues.append("Typy klas nie są całkowite po ponownym wczytaniu.")
    if reload_issues:
        raise ValueError("Walidacja zapisanego submission nie powiodła się: " + " ".join(reload_issues))

    report = {
        "path": str(target_path),
        "shape": list(submission.shape),
        "label_density": float(values.mean()),
        "mean_labels_per_record": float(values.sum(axis=1).mean()),
        "binary_inconsistencies": binary_inconsistencies,
        "selected_variant": selected_variant,
        "issues": [],
    }
    write_json(cfg.report_dir / report_filename, report)
    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def run_self_test() -> None:
    LOGGER.info("Uruchamiam self-test.")
    ontology = Ontology(
        ["class_0", "class_1", "class_2"],
        [("class_0", "class_1"), ("class_1", "class_2")],
    )
    pred = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    closed = ontology.binary_closure(pred)
    assert closed.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert np.array_equal(closed, ontology.binary_closure(closed))
    probs = np.array([[0.1, 0.2, 0.9]], dtype=np.float32)
    mono = ontology.monotonic_probabilities(probs)
    assert np.allclose(mono, [[0.9, 0.9, 0.9]])

    graph_a = molecule_to_graph("CCO")
    graph_b = molecule_to_graph("c1ccccc1")
    y_a = np.zeros(CLASS_COUNT, dtype=np.float32)
    y_b = np.zeros(CLASS_COUNT, dtype=np.float32)
    batch = collate_graphs([(graph_a, y_a), (graph_b, y_b)])
    model = GINEClassifier(num_layers=2, hidden_dim=32, head_hidden_dim=64)
    logits = model(batch)
    assert logits.shape == (2, CLASS_COUNT)
    loss = logits.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    LOGGER.info("Self-test zakończony powodzeniem.")


# ---------------------------------------------------------------------------
# Główny pipeline
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Config) -> None:
    validate_config(cfg)
    ensure_dirs(cfg)
    setup_logging(cfg.output_dir)
    resolved_config = jsonable(cfg)
    write_json(cfg.output_dir / "resolved_config.json", resolved_config)
    config_digest = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(
        cfg.report_dir / "config_hash.json",
        {"sha256": config_digest},
    )

    if cfg.self_test:
        run_self_test()
        return

    required_paths = [cfg.train, cfg.test, cfg.ontology]
    if cfg.sample_submission is not None:
        required_paths.append(cfg.sample_submission)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Brak plików wejściowych:\n- " + "\n- ".join(missing))
    if cfg.dry_run:
        LOGGER.info("Dry-run OK. Ścieżki istnieją, katalogi wynikowe utworzone.")
        return

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    timer = StageTimer()
    budget = TimeBudget(cfg.time_limit_minutes, cfg.buffer_minutes)
    write_json(cfg.report_dir / "runtime.json", runtime_report(device))
    LOGGER.info("Urządzenie: %s", device)

    with timer.stage("hashe plików"):
        train_hash = sha256_file(cfg.train)
        test_hash = sha256_file(cfg.test)
        ontology_hash = sha256_file(cfg.ontology)
        # Hash danych nie może zależeć od rodzaju cech. Wersje featuryzacji są
        # częścią osobnych kluczy cache grafów i ECFP.
        combined_file_hash = hashlib.sha256(
            f"{train_hash}|{test_hash}".encode()
        ).hexdigest()
        write_json(
            cfg.report_dir / "input_hashes.json",
            {
                "train_file": train_hash,
                "test_file": test_hash,
                "ontology": ontology_hash,
                "combined_files": combined_file_hash,
            },
        )

    with timer.stage("wczytanie i walidacja danych"):
        train_df, test_df, sample_df, _ = load_and_validate_data(cfg)
        labels = train_df[CLASS_NAMES].to_numpy(dtype=np.uint8, copy=True)
        effective_train_hash = effective_dataset_hash(
            train_hash, train_df, cfg, "train"
        )
        effective_test_hash = effective_dataset_hash(test_hash, test_df, cfg, "test")
        combined_hash = hashlib.sha256(
            f"{effective_train_hash}|{effective_test_hash}".encode()
        ).hexdigest()
        write_json(
            cfg.report_dir / "input_hashes.json",
            {
                "train_file": train_hash,
                "test_file": test_hash,
                "ontology": ontology_hash,
                "combined_files": combined_file_hash,
                "effective_train": effective_train_hash,
                "effective_test": effective_test_hash,
                "combined_effective_data": combined_hash,
            },
        )

    with timer.stage("ontologia"):
        ontology = load_ontology(cfg)
        training_violations = ontology.inconsistency_counts(labels)
        class0_universal = bool((labels[:, 0] == 1).all())
        class0_is_root = len(ontology.parents[0]) == 0
        if cfg.force_class0 and not class0_universal:
            raise ValueError(
                "--force-class0 jest włączone, ale class_0 nie jest dodatnia "
                "dla wszystkich rekordów treningowych."
            )
        if cfg.force_class0 and not class0_is_root:
            raise ValueError(
                "--force-class0 jest włączone, ale class_0 nie jest korzeniem ontologii."
            )
        write_json(
            cfg.report_dir / "training_hierarchy.json",
            {
                "binary_inconsistencies": training_violations,
                "class0_universal": class0_universal,
                "class0_is_root": class0_is_root,
                "force_class0_confirmed": bool(
                    cfg.force_class0 and class0_universal and class0_is_root
                ),
            },
        )

    with timer.stage("kanonikalizacja i split"):
        canonical = get_canonical_smiles(train_df, cfg, effective_train_hash)
        duplicate_report = duplicate_conflict_report(
            canonical, labels, train_df["mol_id"].astype(str).tolist()
        )
        write_json(cfg.report_dir / "duplicate_conflicts.json", duplicate_report)
        train_idx, val_idx, _ = make_or_load_split(
            train_df, labels, canonical, cfg, effective_train_hash
        )

    selected_model = cfg.model
    validation_result: dict[str, Any]
    test_probabilities: np.ndarray
    final_training_report: dict[str, Any]
    fallback_reasons: list[str] = []

    train_graphs: Optional[list[dict[str, Any]]] = None
    test_graphs: Optional[list[dict[str, Any]]] = None
    train_bits: Optional[np.ndarray] = None
    test_bits: Optional[np.ndarray] = None
    train_desc: Optional[np.ndarray] = None
    test_desc: Optional[np.ndarray] = None
    ecfp_fallback_result: Optional[dict[str, Any]] = None
    fallback_result: Optional[dict[str, Any]] = None
    fallback_selection: Optional[dict[str, Any]] = None
    fallback_variant: Optional[dict[str, Any]] = None
    fallback_thresholds: Optional[np.ndarray] = None

    def release_validation_model(result: dict[str, Any]) -> None:
        model = result.pop("model", None)
        if model is not None:
            del model
        clear_cuda_cache()

    def prepare_ecfp_validation(
        reason: Optional[str] = None,
        force_variant: Optional[str] = None,
    ) -> dict[str, Any]:
        nonlocal train_bits, test_bits, train_desc, test_desc, selected_model
        nonlocal ecfp_fallback_result
        if reason:
            fallback_reasons.append(reason)
            LOGGER.warning("Przejście na ECFP: %s", reason)
        if force_variant is None and ecfp_fallback_result is not None:
            selected_model = "ecfp"
            return ecfp_fallback_result
        clear_cuda_cache()
        if train_bits is None or test_bits is None:
            with timer.stage("cechy ECFP"):
                train_bits, test_bits, train_desc, test_desc = build_or_load_ecfp(
                    train_df, test_df, cfg, combined_hash
                )
        with timer.stage("trening walidacyjny ECFP"):
            if force_variant is None:
                result = fit_ecfp_validation(
                    train_bits,
                    train_desc,
                    labels,
                    train_idx,
                    val_idx,
                    cfg,
                    device,
                    budget,
                )
            else:
                result = _fit_ecfp_validation_variant(
                    train_bits,
                    train_desc,
                    labels,
                    train_idx,
                    val_idx,
                    cfg,
                    device,
                    budget,
                    force_variant,
                )
        result = select_best_checkpoint_candidate(
            result,
            ontology,
            cfg,
            frequency_positive_counts=labels[train_idx].sum(axis=0),
        )
        if force_variant is None and cfg.model == "auto" and cfg.prepare_ecfp_fallback:
            ecfp_fallback_result = result
        selected_model = "ecfp"
        return result

    def select_hierarchy(
        result: dict[str, Any],
        artifact_prefix: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
        stage_name = "progi i warianty DAG"
        if artifact_prefix:
            stage_name += f" ({artifact_prefix.rstrip('_')})"
        with timer.stage(stage_name):
            selection = evaluate_hierarchy_variants(
                result["val_labels"],
                result["val_probabilities"],
                ontology,
                cfg,
                frequency_positive_counts=labels[train_idx].sum(axis=0),
            )
            write_json(
                cfg.report_dir / f"{artifact_prefix}hierarchy_variants.json",
                selection,
            )
            variant = selection["selected"]
            selected_thresholds = np.asarray(
                variant["thresholds"], dtype=np.float32
            )
            write_json(
                cfg.checkpoint_dir / f"{artifact_prefix}thresholds.json",
                {
                    "selected_variant": selection["selected_variant"],
                    "thresholds": selected_thresholds.tolist(),
                    "force_class0": cfg.force_class0,
                    "hierarchy_f1_tolerance": cfg.hierarchy_f1_tolerance,
                    "frequency_counts_source": "training_split",
                },
            )
            if cfg.save_probabilities:
                np.save(
                    cfg.checkpoint_dir
                    / f"{artifact_prefix}validation_probabilities.npy",
                    result["val_probabilities"].astype(np.float32),
                )
        return selection, variant, selected_thresholds

    def run_ecfp_full_or_checkpoint(
        result: dict[str, Any],
        stage_name: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        nonlocal selected_model
        assert train_bits is not None and test_bits is not None
        if cfg.final_strategy == "validation_checkpoint":
            with timer.stage("inferencja skalibrowanego checkpointu ECFP"):
                return predict_ecfp_validation_checkpoint(
                    test_bits, test_desc, result, cfg, device
                )
        try:
            with timer.stage(stage_name):
                return train_ecfp_full_and_predict(
                    train_bits,
                    test_bits,
                    train_desc,
                    test_desc,
                    labels,
                    cfg,
                    device,
                    budget,
                    result["best_epoch"],
                    result["descriptor_mean"],
                    result["descriptor_std"],
                    result["model_variant"],
                    result["epoch_times"],
                )
        except TimeBudgetExceeded as exc:
            reason = (
                "Brak czasu na pełny trening ECFP; używam gotowego checkpointu "
                f"walidacyjnego: {exc}"
            )
            fallback_reasons.append(reason)
            LOGGER.warning("%s", reason)
            selected_model = "ecfp"
            with timer.stage("awaryjna inferencja checkpointu ECFP"):
                probabilities, report = predict_ecfp_validation_checkpoint(
                    test_bits, test_desc, result, cfg, device
                )
            report["fallback_reason"] = reason
            return probabilities, report

    def finalize_ecfp_result(
        result: dict[str, Any],
        selection: dict[str, Any],
        variant: dict[str, Any],
        selected_thresholds: np.ndarray,
        stage_name: str = "finalny trening i inferencja ECFP",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        nonlocal selected_model
        selected_model = "ecfp"
        try:
            probabilities, training_report = run_ecfp_full_or_checkpoint(
                result, stage_name
            )
            return (
                result,
                selection,
                variant,
                selected_thresholds,
                probabilities,
                training_report,
            )
        except (FloatingPointError, RuntimeError) as exc:
            can_use_linear = (
                cfg.ecfp_variant == "auto"
                and result.get("model_variant") == "mlp"
                and (isinstance(exc, FloatingPointError) or is_cuda_oom(exc))
            )
            if not can_use_linear:
                raise
            message = f"Finalny ECFP-MLP nie powiódł się: {exc}"
            clear_cuda_cache()
            linear_result = prepare_ecfp_validation(
                message,
                force_variant="linear",
            )
            linear_selection, linear_variant, linear_thresholds = select_hierarchy(
                linear_result, artifact_prefix="ecfp_linear_"
            )
            release_validation_model(linear_result)
            probabilities, training_report = run_ecfp_full_or_checkpoint(
                linear_result, "finalny trening i inferencja ECFP-linear"
            )
            return (
                linear_result,
                linear_selection,
                linear_variant,
                linear_thresholds,
                probabilities,
                training_report,
            )

    attempt_gnn = cfg.model in {"auto", "gine"}
    if cfg.model == "auto" and cfg.prepare_ecfp_fallback:
        LOGGER.info("Przygotowuję gotowy fallback ECFP przed uruchomieniem GNN.")
        fallback_result = prepare_ecfp_validation()
        fallback_selection, fallback_variant, fallback_thresholds = select_hierarchy(
            fallback_result, artifact_prefix="ecfp_fallback_"
        )
        assert test_bits is not None
        with timer.stage("awaryjna inferencja ECFP"):
            fallback_test_probabilities, fallback_inference = (
                predict_ecfp_validation_checkpoint(
                    test_bits, test_desc, fallback_result, cfg, device
                )
            )
            fallback_probs_for_thresholds = fallback_test_probabilities
            if fallback_variant["probability_mode"] == "monotonic":
                fallback_probs_for_thresholds = ontology.monotonic_probabilities(
                    fallback_test_probabilities
                )
            fallback_predictions = apply_thresholds(
                fallback_probs_for_thresholds,
                fallback_thresholds,
                cfg.force_class0,
            )
            if fallback_variant["closure"]:
                fallback_predictions = ontology.binary_closure(fallback_predictions)
            fallback_submission_path = cfg.output_dir / "submission_ecfp_fallback.parquet"
            fallback_submission_report = generate_and_validate_submission(
                test_df,
                sample_df,
                fallback_predictions,
                ontology,
                fallback_variant,
                cfg,
                output_path=fallback_submission_path,
                report_filename="submission_ecfp_fallback_validation.json",
            )
            write_json(
                cfg.report_dir / "ecfp_fallback_ready.json",
                {
                    "ready": True,
                    "submission": fallback_submission_report,
                    "inference": fallback_inference,
                    "selected_hierarchy_variant": fallback_selection[
                        "selected_variant"
                    ],
                    "validation_macro_f1": fallback_variant["macro_f1"],
                },
            )
        release_validation_model(fallback_result)
        selected_model = "auto"

        estimated_ecfp_final = (
            estimate_full_training_seconds(
                fallback_result["epoch_times"],
                fallback_result["best_epoch"],
                cfg.val_fraction,
                cfg.final_time_safety_factor,
                cfg.final_inference_reserve_seconds,
            )
            if cfg.final_strategy == "full_retrain"
            else cfg.final_inference_reserve_seconds
        )
        minimum_for_gnn_attempt = (
            2.0 * cfg.benchmark_slow_seconds + estimated_ecfp_final
        )
        if budget.usable_remaining < minimum_for_gnn_attempt:
            reason = (
                "Po przygotowaniu fallbacku ECFP pozostało za mało czasu na "
                "bezpieczny benchmark GNN i finalny fallback: "
                f"potrzeba około {minimum_for_gnn_attempt:.1f}s, dostępne "
                f"{budget.usable_remaining:.1f}s."
            )
            fallback_reasons.append(reason)
            LOGGER.warning("%s", reason)
            validation_result = fallback_result
            selected_model = "ecfp"
            attempt_gnn = False

    if attempt_gnn:
        try:
            with timer.stage("grafy molekularne"):
                train_graphs, test_graphs, _ = build_or_load_graphs(
                    train_df, test_df, cfg, combined_hash
                )

            full_arch = (cfg.num_layers, cfg.hidden_dim, cfg.head_hidden_dim)
            try:
                with timer.stage("trening walidacyjny GINE"):
                    validation_result = fit_gnn_validation(
                        train_graphs,
                        labels,
                        train_idx,
                        val_idx,
                        cfg,
                        device,
                        budget,
                        full_arch,
                        auto_benchmark=(cfg.model == "auto"),
                        checkpoint_name="gine_best_validation.pt",
                        max_epochs=cfg.max_epochs,
                    )
                    validation_result = select_best_checkpoint_candidate(
                        validation_result,
                        ontology,
                        cfg,
                        frequency_positive_counts=labels[train_idx].sum(axis=0),
                    )
            except NeedSmallGNN as exc:
                message = str(exc)
                LOGGER.warning("%s", message)
                del exc
                clear_cuda_cache()
                set_seed(cfg.seed + 1)
                small_arch = (3, 128, 256)
                with timer.stage("trening walidacyjny GINE 3x128"):
                    validation_result = fit_gnn_validation(
                        train_graphs,
                        labels,
                        train_idx,
                        val_idx,
                        cfg,
                        device,
                        budget,
                        small_arch,
                        auto_benchmark=(cfg.model == "auto"),
                        checkpoint_name="gine_small_best_validation.pt",
                        max_epochs=min(
                            cfg.max_epochs,
                            cfg.small_gnn_max_epochs,
                        ),
                    )
                    validation_result = select_best_checkpoint_candidate(
                        validation_result,
                        ontology,
                        cfg,
                        frequency_positive_counts=labels[train_idx].sum(axis=0),
                    )
            selected_model = "gine"

            if cfg.model == "auto" and cfg.final_strategy == "full_retrain":
                estimated_final = estimate_full_training_seconds(
                    validation_result["epoch_times"],
                    validation_result["best_epoch"],
                    cfg.val_fraction,
                    cfg.final_time_safety_factor,
                    cfg.final_inference_reserve_seconds,
                )
                if budget.usable_remaining < estimated_final:
                    release_validation_model(validation_result)
                    raise NeedECFP(
                        "Po walidacji nie ma bezpiecznego budżetu na pełny finalny "
                        f"GNN: potrzeba około {estimated_final:.1f}s, dostępne "
                        f"{budget.usable_remaining:.1f}s."
                    )
        except (NeedECFP, FloatingPointError, TimeBudgetExceeded) as exc:
            if cfg.model == "gine":
                raise
            message = str(exc)
            del exc
            clear_cuda_cache()
            validation_result = prepare_ecfp_validation(message)
        except RuntimeError as exc:
            if cfg.model == "gine" or not is_cuda_oom(exc):
                raise
            message = f"GNN przekroczył pamięć GPU: {exc}"
            del exc
            clear_cuda_cache()
            validation_result = prepare_ecfp_validation(message)
    else:
        validation_result = prepare_ecfp_validation()

    hierarchy_selection, selected_variant, thresholds = select_hierarchy(
        validation_result
    )

    if (
        cfg.model == "auto"
        and selected_model == "gine"
        and fallback_result is not None
        and fallback_selection is not None
        and fallback_variant is not None
        and fallback_thresholds is not None
    ):
        gnn_key = (
            -float(selected_variant["macro_f1"]),
            float(
                selected_variant["probability_inconsistencies"][
                    "mean_per_record"
                ]
            ),
            float(
                selected_variant["binary_inconsistencies"]["mean_per_record"]
            ),
        )
        ecfp_key = (
            -float(fallback_variant["macro_f1"]),
            float(
                fallback_variant["probability_inconsistencies"][
                    "mean_per_record"
                ]
            ),
            float(
                fallback_variant["binary_inconsistencies"]["mean_per_record"]
            ),
        )
        if ecfp_key < gnn_key:
            reason = (
                "Tryb auto wybrał ECFP na podstawie walidacji: "
                f"ECFP macro-F1={fallback_variant['macro_f1']:.6f}, "
                f"GNN macro-F1={selected_variant['macro_f1']:.6f}."
            )
            LOGGER.info("%s", reason)
            fallback_reasons.append(reason)
            validation_result = fallback_result
            hierarchy_selection = fallback_selection
            selected_variant = fallback_variant
            thresholds = fallback_thresholds
            selected_model = "ecfp"
        else:
            LOGGER.info(
                "Tryb auto wybrał GNN na podstawie walidacji: "
                "GNN macro-F1=%.6f, ECFP macro-F1=%.6f.",
                selected_variant["macro_f1"],
                fallback_variant["macro_f1"],
            )
    release_validation_model(validation_result)

    if selected_model == "gine":
        assert train_graphs is not None and test_graphs is not None
        gnn_failure: Optional[str] = None
        try:
            if cfg.final_strategy == "validation_checkpoint":
                with timer.stage("inferencja skalibrowanego checkpointu GINE"):
                    test_probabilities, final_training_report = (
                        predict_gnn_validation_checkpoint(
                            test_graphs,
                            validation_result,
                            cfg,
                            device,
                        )
                    )
            else:
                with timer.stage("finalny trening i inferencja GINE"):
                    test_probabilities, final_training_report = train_gnn_full_and_predict(
                        train_graphs,
                        test_graphs,
                        labels,
                        cfg,
                        device,
                        budget,
                        validation_result["architecture"],
                        validation_result["best_epoch"],
                        validation_result["epoch_times"],
                    )
        except (TimeBudgetExceeded, FloatingPointError) as exc:
            if cfg.model == "gine":
                raise
            gnn_failure = f"Finalny GNN nie może zostać poprawnie ukończony: {exc}"
        except RuntimeError as exc:
            if cfg.model == "gine" or not is_cuda_oom(exc):
                raise
            gnn_failure = f"Finalny GNN przekroczył pamięć GPU: {exc}"

        if gnn_failure is not None:
            clear_cuda_cache()
            validation_result = prepare_ecfp_validation(gnn_failure)
            hierarchy_selection, selected_variant, thresholds = select_hierarchy(
                validation_result
            )
            release_validation_model(validation_result)
            (
                validation_result,
                hierarchy_selection,
                selected_variant,
                thresholds,
                test_probabilities,
                final_training_report,
            ) = finalize_ecfp_result(
                validation_result,
                hierarchy_selection,
                selected_variant,
                thresholds,
            )
    else:
        (
            validation_result,
            hierarchy_selection,
            selected_variant,
            thresholds,
            test_probabilities,
            final_training_report,
        ) = finalize_ecfp_result(
            validation_result,
            hierarchy_selection,
            selected_variant,
            thresholds,
        )

    with timer.stage("postprocessing i submission"):
        probabilities_for_thresholds = test_probabilities
        if selected_variant["probability_mode"] == "monotonic":
            probabilities_for_thresholds = ontology.monotonic_probabilities(
                test_probabilities
            )
        predictions = apply_thresholds(
            probabilities_for_thresholds, thresholds, cfg.force_class0
        )
        if selected_variant["closure"]:
            predictions = ontology.binary_closure(predictions)
        if cfg.save_probabilities:
            np.save(
                cfg.checkpoint_dir / "test_probabilities_raw.npy",
                test_probabilities.astype(np.float32),
            )
        submission_report = generate_and_validate_submission(
            test_df,
            sample_df,
            predictions,
            ontology,
            selected_variant,
            cfg,
        )

    final_report = {
        "selected_model": selected_model,
        "ecfp_model_variant": validation_result.get("model_variant"),
        "final_strategy": cfg.final_strategy,
        "validation_best_epoch": validation_result["best_epoch"],
        "validation_working_macro_f1": validation_result["best_working_metric"],
        "checkpoint_selection": validation_result.get("checkpoint_selection"),
        "selected_hierarchy_variant": hierarchy_selection["selected_variant"],
        "selected_validation_macro_f1": selected_variant["macro_f1"],
        "fallback_reasons": fallback_reasons,
        "final_training": final_training_report,
        "submission": submission_report,
        "stage_seconds": timer.stages,
        "total_elapsed_seconds": timer.elapsed,
        "time_budget": budget.summary(),
        "config_sha256": config_digest,
    }
    write_json(cfg.report_dir / "final_report.json", final_report)
    LOGGER.info("Gotowe: %s", cfg.submission_path)
    LOGGER.info(
        "Model=%s | val macro-F1=%.6f | wariant=%s | czas=%.1f s",
        selected_model,
        selected_variant["macro_f1"],
        hierarchy_selection["selected_variant"],
        timer.elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description="GINE/ECFP pipeline dla hierarchicznej klasyfikacji ChEBI."
    )
    parser.add_argument("--train", type=Path, default=Path("chebi_dataset_train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("chebi_dataset_test_empty.parquet"))
    parser.add_argument("--ontology", type=Path, default=Path("chebi_classes.obo"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("chebi_submission_example.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--model", choices=["auto", "gine", "ecfp"], default="auto")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 lub cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=min(8, max(0, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--target-atoms-per-batch", type=int, default=12000)
    parser.add_argument("--eval-atoms-per-batch", type=int, default=16000)
    parser.add_argument("--bucket-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--head-hidden-dim", type=int, default=384)
    parser.add_argument("--gnn-dropout", type=float, default=0.15)
    parser.add_argument("--head-dropout", type=float, default=0.20)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--small-gnn-max-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--evaluation-every", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--pos-weight-clip", type=float, default=6.0)
    parser.add_argument("--working-threshold", type=float, default=0.30)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-limit-minutes", type=float, default=60.0)
    parser.add_argument("--buffer-minutes", type=float, default=5.0)
    parser.add_argument("--graph-preprocess-limit-minutes", type=float, default=10.0)
    parser.add_argument("--benchmark-fast-seconds", type=float, default=50.0)
    parser.add_argument("--benchmark-slow-seconds", type=float, default=70.0)
    parser.add_argument("--ecfp-bits", type=int, default=4096)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--ecfp-hidden-dim", type=int, default=1024)
    parser.add_argument("--ecfp-batch-size", type=int, default=512)
    parser.add_argument("--ecfp-eval-batch-size", type=int, default=1024)
    parser.add_argument("--ecfp-max-epochs", type=int, default=15)
    parser.add_argument("--ecfp-patience", type=int, default=3)
    parser.add_argument("--ecfp-descriptors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ecfp-variant", choices=["auto", "mlp", "linear"], default="auto")
    parser.add_argument("--force-class0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hierarchy-f1-tolerance", type=float, default=0.01)
    parser.add_argument("--final-time-safety-factor", type=float, default=1.35)
    parser.add_argument("--final-inference-reserve-seconds", type=float, default=180.0)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--debug-sample",
        type=int,
        default=0,
        help="Liczba rekordów train do szybkiego testu (0 = pełne dane).",
    )
    parser.add_argument("--save-probabilities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prepare-ecfp-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="W trybie auto przygotuj poprawny fallback ECFP przed GNN.",
    )
    parser.add_argument(
        "--checkpoint-candidates",
        type=int,
        default=3,
        help=(
            "Liczba najlepszych checkpointów wybieranych ponownie po pełnej "
            "kalibracji progów i wariantów DAG."
        ),
    )
    parser.add_argument(
        "--final-strategy",
        choices=["validation_checkpoint", "full_retrain"],
        default="validation_checkpoint",
        help=(
            "validation_checkpoint zachowuje zgodność kalibracji i jest szybsze; "
            "full_retrain odtwarza wcześniejsze trenowanie od zera na całym train."
        ),
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    submission_path = (
        args.submission.resolve()
        if args.submission is not None
        else output_dir / "submission.parquet"
    )
    sample_submission = args.sample_submission
    if sample_submission is not None and str(sample_submission).lower() in {"none", "null", "-"}:
        sample_submission = None

    return Config(
        train=args.train.resolve(),
        test=args.test.resolve(),
        ontology=args.ontology.resolve(),
        sample_submission=sample_submission.resolve() if sample_submission else None,
        output_dir=output_dir,
        submission_path=submission_path,
        model=args.model,
        device=args.device,
        seed=args.seed,
        val_fraction=args.val_fraction,
        num_workers=max(0, args.num_workers),
        feature_workers=max(1, args.feature_workers),
        target_atoms_per_batch=args.target_atoms_per_batch,
        eval_atoms_per_batch=args.eval_atoms_per_batch,
        bucket_size=args.bucket_size,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        gnn_dropout=args.gnn_dropout,
        head_dropout=args.head_dropout,
        max_epochs=args.max_epochs,
        small_gnn_max_epochs=args.small_gnn_max_epochs,
        patience=args.patience,
        evaluation_every=args.evaluation_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        pos_weight_clip=args.pos_weight_clip,
        working_threshold=args.working_threshold,
        use_bf16=args.use_bf16,
        time_limit_minutes=args.time_limit_minutes,
        buffer_minutes=args.buffer_minutes,
        graph_preprocess_limit_minutes=args.graph_preprocess_limit_minutes,
        benchmark_fast_seconds=args.benchmark_fast_seconds,
        benchmark_slow_seconds=args.benchmark_slow_seconds,
        ecfp_bits=args.ecfp_bits,
        ecfp_radius=args.ecfp_radius,
        ecfp_hidden_dim=args.ecfp_hidden_dim,
        ecfp_batch_size=args.ecfp_batch_size,
        ecfp_eval_batch_size=args.ecfp_eval_batch_size,
        ecfp_max_epochs=args.ecfp_max_epochs,
        ecfp_patience=args.ecfp_patience,
        ecfp_descriptors=args.ecfp_descriptors,
        ecfp_variant=args.ecfp_variant,
        force_class0=args.force_class0,
        hierarchy_f1_tolerance=max(0.0, args.hierarchy_f1_tolerance),
        final_time_safety_factor=max(1.0, args.final_time_safety_factor),
        final_inference_reserve_seconds=max(0.0, args.final_inference_reserve_seconds),
        overwrite_cache=args.overwrite_cache,
        dry_run=args.dry_run,
        self_test=args.self_test,
        debug_sample=args.debug_sample,
        save_probabilities=args.save_probabilities,
        prepare_ecfp_fallback=args.prepare_ecfp_fallback,
        checkpoint_candidates=args.checkpoint_candidates,
        final_strategy=args.final_strategy,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        run_pipeline(cfg)
        return 0
    except KeyboardInterrupt:
        print("Przerwano przez użytkownika.", file=sys.stderr)
        return 130
    except Exception as exc:
        if LOGGER.handlers:
            LOGGER.exception("Pipeline zakończony błędem: %s", exc)
        else:
            print(f"Błąd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
