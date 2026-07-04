#!/usr/bin/env python3
"""
gnn_v3.py — mocny pipeline OOF/ensemble do hierarchicznej klasyfikacji wieloetykietowej
cząsteczek ChEBI.

Skrypt realizuje kompletny przebieg:
  1. walidacja train/test/submission,
  2. parsowanie ontologii OBO i operacje na DAG,
  3. kanonikalizacja SMILES i 5-fold grouped multilabel OOF,
  4. GINE 4x256 z deskryptorami, virtual node, EMA i cosine schedule,
  5. ECFP binary/count radius 2/3 z deskryptorami,
  6. OOF kNN Tanimoto oraz ensemble w przestrzeni logitów,
  7. dokładne progi F1 per klasa i porównanie wariantów DAG,
  8. ensemble modeli foldowych na teście oraz exact-match override,
  9. zapis, raportowanie i pełna walidacja submission.parquet.

Domyślne uruchomienie:

    python gnn_v3.py \
      --train chebi_dataset_train.parquet \
      --test chebi_dataset_test_empty.parquet \
      --ontology chebi_classes.obo \
      --sample-submission chebi_submission_example.parquet \
      --output-dir artifacts_v3

Najważniejsze zależności:
    Python >= 3.10, numpy, pandas, pyarrow, torch, rdkit.

PyTorch Geometric NIE jest wymagany — GINE i batchowanie grafów są
zaimplementowane bezpośrednio w PyTorch, aby plik był samodzielny.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
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
from typing import Any, Iterator, Optional, Sequence

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
GRAPH_FEATURE_VERSION = "graph_v4_oof_descriptors_virtual_node"
ECFP_FEATURE_VERSION = "ecfp_v5_binary_count_r2_r3_descriptors"
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
    model: str = "ensemble"  # ensemble | auto | gine | ecfp
    device: str = "auto"
    seed: int = 42
    val_fraction: float = 0.20  # zachowane dla kompatybilności raportów V2
    num_folds: int = 5
    num_workers: int = 4
    feature_workers: int = 1
    target_atoms_per_batch: int = 8000
    eval_atoms_per_batch: int = 16000
    bucket_size: int = 256
    num_layers: int = 4
    hidden_dim: int = 256
    head_hidden_dim: int = 512
    gnn_dropout: float = 0.15
    head_dropout: float = 0.20
    max_epochs: int = 35
    small_gnn_max_epochs: int = 24
    patience: int = 7
    evaluation_every: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 2.0
    pos_weight_clip: float = 8.0
    working_threshold: float = 0.30
    use_bf16: bool = True
    time_limit_minutes: float = 180.0
    buffer_minutes: float = 10.0
    graph_preprocess_limit_minutes: float = 60.0
    benchmark_fast_seconds: float = 120.0
    benchmark_slow_seconds: float = 240.0
    warmup_epochs: float = 2.0
    min_lr_ratio: float = 0.02
    ema_decay: float = 0.999
    use_virtual_node: bool = True
    gnn_members: int = 2
    gnn_second_loss: str = "asl"  # bce | asl
    hierarchy_loss_weight: float = 0.03
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    ecfp_bits: int = 4096
    ecfp_radius: int = 2  # zachowane dla kompatybilności; V3 używa r=2 i r=3
    ecfp_count_clip: int = 15
    ecfp_hidden_dim: int = 1536
    ecfp_batch_size: int = 512
    ecfp_eval_batch_size: int = 1024
    ecfp_max_epochs: int = 25
    ecfp_patience: int = 5
    ecfp_descriptors: bool = True
    ecfp_variant: str = "mlp"  # auto | mlp | linear
    ecfp_members: int = 2
    ecfp_hierarchy_loss_weight: float = 0.01
    knn_enabled: bool = True
    knn_k: int = 25
    knn_power: float = 3.0
    knn_min_similarity: float = 0.20
    knn_prior_weight: float = 1.0
    knn_workers: int = 4
    exact_match_override: bool = True
    exact_match_strength: float = 0.98
    blend_bucket_regularization: float = 0.50
    threshold_shrinkage: bool = True
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
    checkpoint_candidates: int = 10
    final_strategy: str = "fold_ensemble"

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
    if cfg.final_strategy not in {"validation_checkpoint", "full_retrain", "fold_ensemble"}:
        issues.append(
            "final_strategy musi mieć wartość validation_checkpoint, full_retrain lub fold_ensemble."
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
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                jsonable(data),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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


def torch_load_cpu(path: Path) -> Any:
    """Wczytuje plik torch także na starszych wersjach bez weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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
                "bf16_supported": cuda_bf16_supported(device),
            }
        )
    return report


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("Wybrano CUDA, ale torch.cuda.is_available() == False.")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise RuntimeError(
            f"Nieprawidłowy indeks CUDA {index}; dostępne urządzenia: 0..{device_count - 1}."
        )
    return torch.device("cuda", index)


def cuda_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    try:
        with torch.cuda.device(index):
            return bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        return False


def autocast_context(device: torch.device, use_bf16: bool):
    if use_bf16 and cuda_bf16_supported(device):
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    except ImportError as exc:
        raise RuntimeError(
            "Do zapisu Parquet potrzebny jest pyarrow. Zainstaluj: pip install pyarrow"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)


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
        payload = torch_load_cpu(path)
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
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.bucket_size = max(1, int(bucket_size))
        self.epoch = 0

    def _sequence_for_epoch(self, epoch: int) -> np.ndarray:
        if not self.shuffle:
            return np.arange(len(self.num_nodes), dtype=np.int64)
        rng = np.random.default_rng(self.seed + epoch)
        order = np.argsort(self.num_nodes, kind="stable")
        buckets = [
            order[i : i + self.bucket_size].copy()
            for i in range(0, len(order), self.bucket_size)
        ]
        for bucket in buckets:
            rng.shuffle(bucket)
        rng.shuffle(buckets)
        return np.concatenate(buckets) if buckets else np.empty(0, dtype=np.int64)

    def _batches_for_epoch(self, epoch: int) -> list[list[int]]:
        batches: list[list[int]] = []
        batch: list[int] = []
        atom_count = 0
        for position in self._sequence_for_epoch(epoch):
            nodes = int(self.num_nodes[position])
            if batch and atom_count + nodes > self.target_atoms:
                batches.append(batch)
                batch = []
                atom_count = 0
            batch.append(int(position))
            atom_count += nodes
            if nodes >= self.target_atoms:
                batches.append(batch)
                batch = []
                atom_count = 0
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        self.epoch += 1
        yield from self._batches_for_epoch(epoch)

    def __len__(self) -> int:
        # Poprzednia implementacja używała ceil(sum(atomów)/budżet), co bywało
        # mniejsze od faktycznej liczby batchy i zniekształcało cosine schedule.
        return len(self._batches_for_epoch(self.epoch))


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
    payload = torch_load_cpu(checkpoint_path)
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
    payload = torch_load_cpu(checkpoint_path)
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
    try:
        numeric_pred = np.asarray(pred, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predykcje nie są macierzą liczbową 0/1.") from exc
    if not np.isfinite(numeric_pred).all():
        raise ValueError("Predykcje zawierają NaN lub wartości nieskończone.")
    if not np.isin(numeric_pred, [0.0, 1.0]).all():
        invalid = np.unique(numeric_pred[~np.isin(numeric_pred, [0.0, 1.0])])
        raise ValueError(
            "Predykcje muszą być binarne przed konwersją typu; "
            f"przykładowe nieprawidłowe wartości: {invalid[:10].tolist()}."
        )
    label_frame = pd.DataFrame(
        numeric_pred.astype(np.int8, copy=False),
        columns=CLASS_NAMES,
        index=test_df.index,
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


def _run_self_test_legacy_v2() -> None:
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


def _run_pipeline_legacy_v2(cfg: Config) -> None:
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
        _run_self_test_legacy_v2()
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
# V3: 5-fold OOF, bogate cechy, ensemble i dokładna kalibracja
# ---------------------------------------------------------------------------

V3_PIPELINE_REVISION = "gnn_v3_fix_20260704_1"
V3_GRAPH_FEATURE_VERSION = "graph_v4_descriptors_virtual_node_20260704_fix1"
V3_ECFP_FEATURE_VERSION = "ecfp_v5_binary_count_r2_r3_desc_20260704_fix1"
V3_DESCRIPTOR_NAMES = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomMolWt",
    "TPSA",
    "MolLogP",
    "MolMR",
    "NumAtoms",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumRotatableBonds",
    "NumHAcceptors",
    "NumHDonors",
    "FractionCSP3",
    "FormalCharge",
    "NumRadicalElectrons",
    "NumValenceElectrons",
    "NHOHCount",
    "NOCount",
    "NumFragments",
    "BertzCT",
]
V3_DESCRIPTOR_DIM = len(V3_DESCRIPTOR_NAMES)


def molecular_descriptors_v3(mol: Chem.Mol) -> np.ndarray:
    values = np.asarray(
        [
            Descriptors.MolWt(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.TPSA(mol),
            Crippen.MolLogP(mol),
            Crippen.MolMR(mol),
            mol.GetNumAtoms(),
            Lipinski.HeavyAtomCount(mol),
            Lipinski.NumHeteroatoms(mol),
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.FractionCSP3(mol),
            Chem.GetFormalCharge(mol),
            Descriptors.NumRadicalElectrons(mol),
            Descriptors.NumValenceElectrons(mol),
            Lipinski.NHOHCount(mol),
            Lipinski.NOCount(mol),
            len(Chem.GetMolFrags(mol)),
            Descriptors.BertzCT(mol),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)


def canonical_cache_path_v3(
    cfg: Config, dataset_hash: str, split_name: str
) -> Path:
    signature = stable_signature(
        {
            "version": "canonical_isomeric_v3",
            "rdkit": package_version("rdkit"),
            "split": split_name,
        }
    )
    return cfg.cache_dir / (
        f"canonical_{split_name}_{dataset_hash[:12]}_{signature}.npy"
    )


def get_canonical_smiles_v3(
    dataframe: pd.DataFrame,
    cfg: Config,
    dataset_hash: str,
    split_name: str,
) -> np.ndarray:
    path = canonical_cache_path_v3(cfg, dataset_hash, split_name)
    if path.exists() and not cfg.overwrite_cache:
        try:
            values = np.load(path, allow_pickle=True)
            if len(values) == len(dataframe):
                LOGGER.info(
                    "Wczytano cache kanonicznych SMILES %s: %s",
                    split_name,
                    path,
                )
                return values.astype(object)
            LOGGER.warning(
                "Cache kanonicznych SMILES %s ma zły rozmiar — przebudowuję.",
                split_name,
            )
        except (OSError, ValueError, EOFError) as exc:
            LOGGER.warning(
                "Nie można wczytać cache kanonicznych SMILES %s (%s) — przebudowuję.",
                split_name,
                exc,
            )
    values: list[str] = []
    for index, smiles in enumerate(dataframe["SMILES"].astype(str)):
        values.append(canonicalize_smiles(smiles))
        if (index + 1) % 5000 == 0 or index + 1 == len(dataframe):
            LOGGER.info(
                "Kanonikalizacja %s: %d/%d",
                split_name,
                index + 1,
                len(dataframe),
            )
    array = np.asarray(values, dtype=object)
    np.save(path, array, allow_pickle=True)
    return array


def _greedy_group_multilabel_folds(
    group_label_counts: np.ndarray,
    group_sizes: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_groups, n_classes = group_label_counts.shape
    if n_groups < n_splits:
        raise ValueError(
            f"Za mało grup ({n_groups}) na {n_splits} foldów."
        )
    target_size = float(group_sizes.sum()) / n_splits
    target_labels = group_label_counts.sum(axis=0).astype(np.float64) / n_splits
    label_denom = np.maximum(target_labels, 1.0)
    presence = (group_label_counts > 0).astype(np.float64)
    group_frequency = presence.sum(axis=0)
    rarity = presence @ (1.0 / np.maximum(group_frequency, 1.0))
    random_tie = rng.random(n_groups)
    order = np.lexsort((random_tie, -group_sizes, -rarity))

    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    fold_labels = np.zeros((n_splits, n_classes), dtype=np.float64)
    assignment = np.full(n_groups, -1, dtype=np.int16)

    for rank, group_idx in enumerate(order):
        if rank < n_splits:
            candidate_folds = np.asarray([rank], dtype=np.int64)
        else:
            candidate_folds = rng.permutation(n_splits)
        best_fold = -1
        best_score = float("inf")
        for fold in candidate_folds:
            new_size = fold_sizes[fold] + float(group_sizes[group_idx])
            new_labels = fold_labels[fold] + group_label_counts[group_idx]
            size_error = abs(new_size - target_size) / max(target_size, 1.0)
            label_error = np.mean(np.abs(new_labels - target_labels) / label_denom)
            overload = max(0.0, new_size - target_size * 1.08) / max(target_size, 1.0)
            score = 1.75 * label_error + size_error + 3.0 * overload
            if score < best_score:
                best_score = score
                best_fold = int(fold)
        assignment[group_idx] = best_fold
        fold_sizes[best_fold] += float(group_sizes[group_idx])
        fold_labels[best_fold] += group_label_counts[group_idx]

    if np.any(assignment < 0):
        raise RuntimeError("Nie przypisano wszystkich grup do foldów.")
    return assignment


def make_or_load_folds_v3(
    train_df: pd.DataFrame,
    labels: np.ndarray,
    canonical: np.ndarray,
    cfg: Config,
    train_hash: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_splits = int(cfg.num_folds)
    signature = stable_signature(
        {
            "version": V3_PIPELINE_REVISION + "_group_multilabel_kfold",
            "train_hash": train_hash,
            "seed": cfg.seed,
            "n_splits": n_splits,
        }
    )
    path = cfg.cache_dir / f"folds_{train_hash[:12]}_k{n_splits}_{signature}.npz"
    report_path = cfg.report_dir / "folds_report.json"
    if path.exists() and not cfg.overwrite_cache:
        try:
            fold_ids = np.load(path)["fold_ids"].astype(np.int16)
            valid = (
                fold_ids.shape == (len(train_df),)
                and np.all((fold_ids >= 0) & (fold_ids < n_splits))
                and len(np.unique(fold_ids)) == n_splits
            )
            if valid:
                canonical_fold: dict[str, int] = {}
                for row_idx, canon in enumerate(canonical):
                    key = str(canon)
                    fold = int(fold_ids[row_idx])
                    previous = canonical_fold.setdefault(key, fold)
                    if previous != fold:
                        valid = False
                        break
            if valid:
                LOGGER.info("Wczytano %d-fold OOF z cache: %s", n_splits, path)
                report = _fold_report_v3(labels, canonical, fold_ids, cfg)
                report["loaded_from_cache"] = True
                write_json(report_path, report)
                return fold_ids, report
        except (OSError, KeyError, ValueError, IndexError) as exc:
            LOGGER.warning("Nieprawidłowy cache foldów (%s) — przebudowuję.", exc)

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_idx, canon in enumerate(canonical):
        grouped[str(canon)].append(row_idx)
    group_indices = [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]
    group_sizes = np.asarray([len(indices) for indices in group_indices], dtype=np.int64)
    group_label_counts = np.stack(
        [labels[indices].sum(axis=0) for indices in group_indices]
    )
    if len(group_indices) < n_splits:
        raise ValueError(
            f"Za mało unikalnych grup kanonicznych ({len(group_indices)}) "
            f"na {n_splits} foldów."
        )

    group_fold_ids: np.ndarray
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=cfg.seed
        )
        group_presence = (group_label_counts > 0).astype(np.uint8)
        group_fold_ids = np.full(len(group_indices), -1, dtype=np.int16)
        for fold, (_, validation_groups) in enumerate(
            splitter.split(np.zeros(len(group_indices)), group_presence)
        ):
            group_fold_ids[validation_groups] = fold
        if np.any(group_fold_ids < 0):
            raise RuntimeError("Iterative stratification nie przypisała wszystkich grup.")
        fold_counts = np.bincount(group_fold_ids, minlength=n_splits)
        if np.any(fold_counts == 0):
            raise RuntimeError("Iterative stratification utworzyła pusty fold.")
        LOGGER.info("Foldy utworzone przez MultilabelStratifiedKFold na grupach.")
    except ImportError:
        LOGGER.info("Brak iterative-stratification — używam greedy group k-fold.")
        group_fold_ids = _greedy_group_multilabel_folds(
            group_label_counts, group_sizes, n_splits, cfg.seed
        )
    except (ValueError, RuntimeError) as exc:
        LOGGER.warning(
            "Iterative stratification nie powiodła się (%s) — używam greedy group k-fold.",
            exc,
        )
        group_fold_ids = _greedy_group_multilabel_folds(
            group_label_counts, group_sizes, n_splits, cfg.seed
        )

    fold_ids = np.full(len(train_df), -1, dtype=np.int16)
    for group_idx, rows in enumerate(group_indices):
        fold_ids[rows] = group_fold_ids[group_idx]
    if np.any(fold_ids < 0):
        raise RuntimeError("Nie przypisano wszystkich rekordów do foldów.")
    record_fold_counts = np.bincount(fold_ids.astype(np.int64), minlength=n_splits)
    if np.any(record_fold_counts == 0):
        raise RuntimeError(
            f"Utworzono pusty fold; liczebności rekordów: {record_fold_counts.tolist()}."
        )
    for rows in group_indices:
        if len(np.unique(fold_ids[rows])) != 1:
            raise AssertionError("Ta sama struktura kanoniczna trafiła do wielu foldów.")
    np.savez_compressed(path, fold_ids=fold_ids)
    report = _fold_report_v3(labels, canonical, fold_ids, cfg)
    write_json(report_path, report)
    return fold_ids, report


def _fold_report_v3(
    labels: np.ndarray,
    canonical: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    total_positive = labels.sum(axis=0)
    for fold in range(cfg.num_folds):
        validation_idx = np.flatnonzero(fold_ids == fold)
        training_idx = np.flatnonzero(fold_ids != fold)
        validation_positive = labels[validation_idx].sum(axis=0)
        folds.append(
            {
                "fold": fold,
                "train_records": len(training_idx),
                "validation_records": len(validation_idx),
                "validation_fraction": len(validation_idx) / len(labels),
                "classes_without_positive": [
                    CLASS_NAMES[i] for i in np.flatnonzero(validation_positive == 0)
                ],
                "validation_positive_per_class": validation_positive.tolist(),
            }
        )
    canonical_folds: dict[str, set[int]] = defaultdict(set)
    for row_idx, canon in enumerate(canonical):
        canonical_folds[str(canon)].add(int(fold_ids[row_idx]))
    canonical_fold_counts = [len(values) for values in canonical_folds.values()]
    return {
        "seed": cfg.seed,
        "num_folds": cfg.num_folds,
        "records": len(labels),
        "group_count": int(len(np.unique(canonical))),
        "max_folds_per_canonical_group": int(max(canonical_fold_counts, default=0)),
        "total_positive_per_class": total_positive.tolist(),
        "folds": folds,
    }


def probability_matrix_is_valid(
    values: np.ndarray, expected_shape: tuple[int, int]
) -> bool:
    array = np.asarray(values)
    return bool(
        array.shape == expected_shape
        and np.issubdtype(array.dtype, np.number)
        and np.isfinite(array).all()
        and np.all((array >= 0.0) & (array <= 1.0))
    )


def probability_to_logit(probabilities: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float32), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def logit_to_probability(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    positive = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def exact_f1_thresholds_v3(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    force_class0: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float32)
    n_samples, n_classes = y.shape
    thresholds = np.ones(n_classes, dtype=np.float32)
    best_scores = np.zeros(n_classes, dtype=np.float64)
    for class_idx in range(n_classes):
        truth = y[:, class_idx].astype(np.int64, copy=False)
        positive_count = int(truth.sum())
        if positive_count == 0:
            thresholds[class_idx] = 1.0
            continue
        order = np.argsort(-probs[:, class_idx], kind="stable")
        sorted_probs = probs[order, class_idx]
        cumulative_tp = np.cumsum(truth[order], dtype=np.int64)
        # Progi należy oceniać po całych grupach remisów, bo predykcja używa >=.
        unique_ends = np.flatnonzero(
            np.r_[sorted_probs[:-1] != sorted_probs[1:], True]
        )
        k = unique_ends + 1
        f1 = 2.0 * cumulative_tp[unique_ends] / (k + positive_count)
        best_position = int(np.argmax(f1))
        best_end = int(unique_ends[best_position])
        thresholds[class_idx] = float(sorted_probs[best_end])
        best_scores[class_idx] = float(f1[best_position])
    if force_class0:
        thresholds[0] = 0.0
        best_scores[0] = 1.0 if np.all(y[:, 0] == 1) else best_scores[0]
    return thresholds, best_scores


def _bucket_targets_from_thresholds_v3(
    thresholds: np.ndarray,
    frequency_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    buckets = class_frequency_bucket(frequency_counts)
    threshold_logits = probability_to_logit(np.clip(thresholds, 1e-4, 1 - 1e-4))
    bucket_logits = np.zeros(5, dtype=np.float32)
    for bucket in range(5):
        mask = buckets == bucket
        if mask.any():
            bucket_logits[bucket] = float(np.median(threshold_logits[mask]))
        else:
            bucket_logits[bucket] = probability_to_logit(np.asarray([0.30]))[0]
    return buckets, bucket_logits


def optimize_thresholds_exact_v3(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    force_class0: bool,
    frequency_positive_counts: Optional[np.ndarray] = None,
    shrinkage: bool = True,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float32)
    class_thresholds, exact_scores = exact_f1_thresholds_v3(y, probs, force_class0)
    validation_positive = y.sum(axis=0).astype(np.int64)
    frequency_counts = (
        validation_positive
        if frequency_positive_counts is None
        else np.asarray(frequency_positive_counts, dtype=np.int64)
    )
    buckets, bucket_logits = _bucket_targets_from_thresholds_v3(
        class_thresholds, frequency_counts
    )
    class_logits = probability_to_logit(np.clip(class_thresholds, 1e-4, 1 - 1e-4))

    best_thresholds = class_thresholds.copy()
    best_score = macro_f1(y, apply_thresholds(probs, best_thresholds, force_class0))
    best_alpha_rare = 0.0
    best_alpha_mid = 0.0
    if shrinkage:
        alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
        for alpha_rare in alpha_grid:
            for alpha_mid in alpha_grid:
                candidate_logits = class_logits.copy()
                rare_mask = frequency_counts < 200
                mid_mask = (frequency_counts >= 200) & (frequency_counts <= 500)
                candidate_logits[rare_mask] = (
                    (1.0 - alpha_rare) * class_logits[rare_mask]
                    + alpha_rare * bucket_logits[buckets[rare_mask]]
                )
                candidate_logits[mid_mask] = (
                    (1.0 - alpha_mid) * class_logits[mid_mask]
                    + alpha_mid * bucket_logits[buckets[mid_mask]]
                )
                no_positive = validation_positive == 0
                candidate_logits[no_positive] = bucket_logits[buckets[no_positive]]
                candidate = logit_to_probability(candidate_logits)
                if force_class0:
                    candidate[0] = 0.0
                score = macro_f1(y, apply_thresholds(probs, candidate, force_class0))
                if score > best_score + 1e-12:
                    best_score = score
                    best_thresholds = candidate.astype(np.float32)
                    best_alpha_rare = float(alpha_rare)
                    best_alpha_mid = float(alpha_mid)
    if force_class0:
        class_thresholds[0] = 0.0
        best_thresholds[0] = 0.0
    return {
        "class_thresholds": class_thresholds,
        "regularized_thresholds": best_thresholds,
        "exact_per_class_f1": exact_scores,
        "bucket_thresholds": logit_to_probability(bucket_logits),
        "validation_positive_counts": validation_positive,
        "frequency_positive_counts": frequency_counts,
        "shrinkage_alpha_rare": best_alpha_rare,
        "shrinkage_alpha_mid": best_alpha_mid,
        "macro_f1_raw_thresholds": macro_f1(
            y, apply_thresholds(probs, class_thresholds, force_class0)
        ),
        "macro_f1_regularized_thresholds": best_score,
    }


def quick_calibrated_macro_f1_v3(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    force_class0: bool,
    class_mask: Optional[np.ndarray] = None,
) -> float:
    y = np.asarray(y_true, dtype=np.uint8)
    probs = np.asarray(probabilities, dtype=np.float32)
    grid = np.asarray(
        [
            0.02,
            0.03,
            0.04,
            0.05,
            0.075,
            0.10,
            0.125,
            0.15,
            0.175,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
        ],
        dtype=np.float32,
    )
    best = np.zeros(y.shape[1], dtype=np.float64)
    for threshold in grid:
        pred = probs >= threshold
        if force_class0:
            pred[:, 0] = True
        best = np.maximum(best, f1_per_class(y, pred))
    if class_mask is not None:
        mask = np.asarray(class_mask, dtype=bool)
        return float(best[mask].mean()) if mask.any() else 0.0
    return float(best.mean())


def evaluate_hierarchy_variants_v3(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    frequency_positive_counts: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    raw_threshold_info = optimize_thresholds_exact_v3(
        y_true,
        raw_probabilities,
        cfg.force_class0,
        frequency_positive_counts,
        cfg.threshold_shrinkage,
    )
    raw_thresholds = np.asarray(
        raw_threshold_info["regularized_thresholds"], dtype=np.float32
    )
    monotonic_probs = ontology.monotonic_probabilities(raw_probabilities)
    mono_threshold_info = optimize_thresholds_exact_v3(
        y_true,
        monotonic_probs,
        cfg.force_class0,
        frequency_positive_counts,
        cfg.threshold_shrinkage,
    )
    mono_thresholds = np.asarray(
        mono_threshold_info["regularized_thresholds"], dtype=np.float32
    )
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
    variants: dict[str, dict[str, Any]] = {}
    for name, (probs, thresholds, closure, mode) in definitions.items():
        pred = apply_thresholds(probs, thresholds, cfg.force_class0)
        if closure:
            pred = ontology.binary_closure(pred)
        per_class = f1_per_class(y_true, pred)
        variants[name] = {
            "macro_f1": float(per_class.mean()),
            "per_class_f1": per_class.tolist(),
            "mean_positive_labels": float(pred.sum(axis=1).mean()),
            "binary_inconsistencies": ontology.inconsistency_counts(pred),
            "probability_inconsistencies": ontology.inconsistency_counts(
                probs, eps=1e-8
            ),
            "closure": closure,
            "probability_mode": mode,
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
            -pair[1]["macro_f1"],
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

# ---------------------------------------------------------------------------
# V3: grafy z deskryptorami, virtual node, EMA i cosine schedule
# ---------------------------------------------------------------------------


def molecule_to_graph_v3(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit nie może sparsować SMILES: {smiles!r}")
    atom_cats: list[list[int]] = []
    atom_nums: list[list[float]] = []
    for atom in mol.GetAtoms():
        atom_cats.append(
            [
                int(np.clip(atom.GetAtomicNum(), 0, 118)),
                int(np.clip(atom.GetDegree(), 0, 10)),
                int(np.clip(atom.GetFormalCharge(), -8, 8)) + 8,
                HYBRIDIZATION_MAP.get(atom.GetHybridization(), 7),
                int(atom.GetIsAromatic()),
                int(np.clip(atom.GetNumExplicitHs(), 0, 8)),
                int(np.clip(atom.GetNumImplicitHs(), 0, 8)),
                int(atom.IsInRing()),
                CHIRAL_MAP.get(atom.GetChiralTag(), 3),
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
    if n_atoms <= 0:
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
        "graph_desc": molecular_descriptors_v3(mol),
    }


def graph_cache_path_v3(cfg: Config, combined_hash: str) -> Path:
    signature = stable_signature(
        {
            "version": V3_GRAPH_FEATURE_VERSION,
            "rdkit": package_version("rdkit"),
            "atom_cardinalities": ATOM_CARDINALITIES,
            "edge_cardinalities": EDGE_CARDINALITIES,
            "descriptors": V3_DESCRIPTOR_NAMES,
        },
        length=20,
    )
    return cfg.cache_dir / f"graphs_v3_{combined_hash[:12]}_{signature}.pt"


def _graph_record_is_valid_v3(graph: dict[str, Any]) -> bool:
    try:
        num_nodes = int(graph["num_nodes"])
        x_cat = np.asarray(graph["x_cat"])
        x_num = np.asarray(graph["x_num"])
        edge_index = np.asarray(graph["edge_index"])
        edge_cat = np.asarray(graph["edge_cat"])
        graph_desc = np.asarray(graph["graph_desc"])
    except (KeyError, TypeError, ValueError):
        return False
    if num_nodes <= 0:
        return False
    if x_cat.shape != (num_nodes, len(ATOM_CARDINALITIES)):
        return False
    if x_num.shape != (num_nodes, 1):
        return False
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return False
    if edge_cat.shape != (edge_index.shape[1], len(EDGE_CARDINALITIES)):
        return False
    if graph_desc.shape != (V3_DESCRIPTOR_DIM,) or not np.isfinite(graph_desc).all():
        return False
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
        return False
    return True


def build_or_load_graphs_v3(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    combined_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    path = graph_cache_path_v3(cfg, combined_hash)
    if path.exists() and not cfg.overwrite_cache:
        started = time.perf_counter()
        try:
            payload = torch_load_cpu(path)
            train_graphs = payload.get("train_graphs", [])
            test_graphs = payload.get("test_graphs", [])
            valid = (
                payload.get("feature_version") == V3_GRAPH_FEATURE_VERSION
                and payload.get("rdkit_version") == package_version("rdkit")
                and payload.get("combined_hash") == combined_hash
                and len(train_graphs) == len(train_df)
                and len(test_graphs) == len(test_df)
                and all(_graph_record_is_valid_v3(graph) for graph in train_graphs)
                and all(_graph_record_is_valid_v3(graph) for graph in test_graphs)
            )
            if valid:
                elapsed = time.perf_counter() - started
                LOGGER.info("Wczytano cache grafów V3: %s (%.2f s)", path, elapsed)
                return train_graphs, test_graphs, elapsed
            LOGGER.warning("Cache grafów V3 jest niezgodny — przebudowuję.")
        except (OSError, RuntimeError, EOFError, TypeError, AttributeError) as exc:
            LOGGER.warning("Nie można wczytać cache grafów V3 (%s) — przebudowuję.", exc)

    started = time.perf_counter()
    train_graphs: list[dict[str, Any]] = []
    test_graphs: list[dict[str, Any]] = []
    executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    chunk_size = 256
    try:
        if cfg.feature_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=cfg.feature_workers,
                thread_name_prefix="rdkit-v3-graph",
            )
        for split_name, dataframe, target in [
            ("train", train_df, train_graphs),
            ("test", test_df, test_graphs),
        ]:
            smiles_values = dataframe["SMILES"].astype(str).tolist()
            for chunk_start in range(0, len(smiles_values), chunk_size):
                chunk = smiles_values[chunk_start : chunk_start + chunk_size]
                if executor is None:
                    graphs = [molecule_to_graph_v3(value) for value in chunk]
                else:
                    graphs = list(executor.map(molecule_to_graph_v3, chunk))
                target.extend(graphs)
                completed = chunk_start + len(chunk)
                if completed % 5000 < chunk_size or completed == len(smiles_values):
                    LOGGER.info(
                        "Grafy V3 %s: %d/%d",
                        split_name,
                        completed,
                        len(smiles_values),
                    )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    payload = {
        "feature_version": V3_GRAPH_FEATURE_VERSION,
        "rdkit_version": package_version("rdkit"),
        "combined_hash": combined_hash,
        "descriptor_names": V3_DESCRIPTOR_NAMES,
        "train_graphs": train_graphs,
        "test_graphs": test_graphs,
    }
    atomic_torch_save(path, payload)
    elapsed = time.perf_counter() - started
    write_json(
        cfg.report_dir / "graph_preprocessing_v3.json",
        {
            "elapsed_seconds": elapsed,
            "feature_workers": cfg.feature_workers,
            "train_graphs": len(train_graphs),
            "test_graphs": len(test_graphs),
            "descriptor_dim": V3_DESCRIPTOR_DIM,
            "descriptor_names": V3_DESCRIPTOR_NAMES,
            "train_atoms": {
                "min": min(graph["num_nodes"] for graph in train_graphs),
                "median": float(
                    np.median([graph["num_nodes"] for graph in train_graphs])
                ),
                "max": max(graph["num_nodes"] for graph in train_graphs),
            },
        },
    )
    return train_graphs, test_graphs, elapsed


class V3GraphDataset(Dataset):
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

    def __getitem__(
        self, position: int
    ) -> tuple[dict[str, Any], Optional[np.ndarray]]:
        source_idx = int(self.indices[position])
        label = None if self.labels is None else self.labels[source_idx]
        return self.graphs[source_idx], label


def collate_graphs_v3(
    items: Sequence[tuple[dict[str, Any], Optional[np.ndarray]]]
) -> dict[str, torch.Tensor]:
    x_cat_parts: list[torch.Tensor] = []
    x_num_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_cat_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    graph_desc: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offset = 0
    for graph_idx, (graph, label) in enumerate(items):
        x_cat = torch.from_numpy(np.asarray(graph["x_cat"])).long()
        x_num = torch.from_numpy(np.asarray(graph["x_num"])).float()
        edge_index = torch.from_numpy(np.asarray(graph["edge_index"])).long()
        edge_cat = torch.from_numpy(np.asarray(graph["edge_cat"])).long()
        n_nodes = int(graph["num_nodes"])
        x_cat_parts.append(x_cat)
        x_num_parts.append(x_num)
        if edge_index.numel() > 0:
            edge_index_parts.append(edge_index + offset)
            edge_cat_parts.append(edge_cat)
        batch_parts.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
        graph_desc.append(
            torch.from_numpy(np.asarray(graph["graph_desc"], dtype=np.float32))
        )
        if label is not None:
            labels.append(torch.as_tensor(label, dtype=torch.float32))
        offset += n_nodes
    result: dict[str, Any] = {
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
        "graph_desc": torch.stack(graph_desc, dim=0),
        "num_graphs": len(items),
    }
    if labels:
        result["y"] = torch.stack(labels, dim=0)
    return result


def make_graph_loader_v3(
    dataset: V3GraphDataset,
    cfg: Config,
    training: bool,
    seed: int,
) -> DataLoader:
    target = cfg.target_atoms_per_batch if training else cfg.eval_atoms_per_batch
    sampler = AtomBudgetBatchSampler(
        dataset.num_nodes,
        target_atoms=target,
        shuffle=training,
        seed=seed,
        bucket_size=cfg.bucket_size,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "collate_fn": collate_graphs_v3,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if cfg.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(**kwargs)


class V3GINEClassifier(nn.Module):
    def __init__(
        self,
        descriptor_mean: np.ndarray,
        descriptor_std: np.ndarray,
        num_classes: int = CLASS_COUNT,
        num_layers: int = 4,
        hidden_dim: int = 256,
        head_hidden_dim: int = 512,
        gnn_dropout: float = 0.15,
        head_dropout: float = 0.20,
        use_virtual_node: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.use_virtual_node = bool(use_virtual_node)
        self.atom_encoder = CategoricalEncoder(ATOM_CARDINALITIES, hidden_dim)
        self.atom_numeric = nn.Linear(1, hidden_dim, bias=False)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.edge_encoder = CategoricalEncoder(EDGE_CARDINALITIES, hidden_dim)
        self.blocks = nn.ModuleList(
            [GINEBlock(hidden_dim, gnn_dropout) for _ in range(num_layers)]
        )
        if self.use_virtual_node:
            self.virtual_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
            self.virtual_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(gnn_dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.register_parameter("virtual_embedding", None)
            self.virtual_mlps = nn.ModuleList()
        mean = np.asarray(descriptor_mean, dtype=np.float32)
        std = np.asarray(descriptor_std, dtype=np.float32)
        self.register_buffer("descriptor_mean", torch.from_numpy(mean))
        self.register_buffer("descriptor_std", torch.from_numpy(std))
        head_input = 3 * hidden_dim + V3_DESCRIPTOR_DIM
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Linear(head_input, head_hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.atom_encoder(batch["x_cat"]) + self.atom_numeric(batch["x_num"])
        x = F.silu(self.input_norm(x))
        edge_attr = self.edge_encoder(batch["edge_cat"])
        num_graphs = int(batch["num_graphs"])
        virtual: Optional[torch.Tensor] = None
        if self.use_virtual_node:
            virtual = self.virtual_embedding.expand(num_graphs, -1)
        layer_outputs: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            layer_input = x
            if virtual is not None:
                layer_input = layer_input + virtual[batch["batch"]]
            x = block(layer_input, batch["edge_index"], edge_attr)
            layer_outputs.append(x)
            if virtual is not None:
                pooled = torch.zeros(
                    (num_graphs, self.hidden_dim), dtype=x.dtype, device=x.device
                )
                pooled.index_add_(0, batch["batch"], x)
                counts = torch.bincount(
                    batch["batch"], minlength=num_graphs
                ).clamp_min(1).to(x.dtype)
                pooled = pooled / counts.sqrt().unsqueeze(1)
                virtual = virtual + self.virtual_mlps[layer_idx](pooled)
        x = torch.stack(layer_outputs, dim=0).sum(dim=0)
        pooled = global_pool_all(x, batch["batch"], num_graphs)
        desc = (batch["graph_desc"] - self.descriptor_mean) / self.descriptor_std
        desc = torch.nan_to_num(desc, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.head(torch.cat([pooled, desc], dim=1))

    def architecture(self) -> dict[str, Any]:
        return {
            "model": "gine_v3",
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "head_hidden_dim": self.head_hidden_dim,
            "num_classes": self.num_classes,
            "descriptor_dim": V3_DESCRIPTOR_DIM,
            "use_virtual_node": self.use_virtual_node,
        }


class ModelEMAV3:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            source = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source)


class AsymmetricLossV3(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits.float())
        positive = probabilities.clamp(min=1e-8, max=1.0 - 1e-8)
        negative = (1.0 - probabilities).clamp(min=1e-8, max=1.0)
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)
        positive_loss = targets * torch.log(positive) * self.pos_weight
        negative_loss = (1.0 - targets) * torch.log(negative)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = positive * targets + negative * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weight = torch.pow(1.0 - pt, gamma)
            positive_loss = positive_loss * weight
            negative_loss = negative_loss * weight
        return -(positive_loss + negative_loss).mean()


def learning_rate_factor_v3(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-4, float(step + 1) / warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(
        1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps))
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def set_optimizer_lr_v3(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    factor = learning_rate_factor_v3(
        step, total_steps, warmup_steps, min_lr_ratio
    )
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def hierarchy_penalty_v3(
    logits: torch.Tensor,
    parent_indices: torch.Tensor,
    child_indices: torch.Tensor,
) -> torch.Tensor:
    if parent_indices.numel() == 0:
        return logits.new_zeros(())
    difference = logits[:, child_indices] - logits[:, parent_indices]
    return F.softplus(difference * 5.0).mean() / 5.0


@torch.inference_mode()
def predict_gnn_v3(
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
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(labels, axis=0) if labels else None,
    )


def _cpu_state_dict_v3(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_gnn_fold_v3(
    train_graphs: Sequence[dict[str, Any]],
    test_graphs: Sequence[dict[str, Any]],
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    fold: int,
    member: int,
    loss_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = cfg.seed + fold * 1009 + member * 100_003
    set_seed(seed)
    descriptor_matrix = np.stack(
        [train_graphs[int(index)]["graph_desc"] for index in train_idx]
    ).astype(np.float32)
    descriptor_mean = descriptor_matrix.mean(axis=0).astype(np.float32)
    descriptor_std = descriptor_matrix.std(axis=0).astype(np.float32)
    descriptor_std = np.where(descriptor_std < 1e-6, 1.0, descriptor_std)

    train_ds = V3GraphDataset(train_graphs, train_idx, labels)
    val_ds = V3GraphDataset(train_graphs, val_idx, labels)
    test_ds = V3GraphDataset(
        test_graphs, np.arange(len(test_graphs), dtype=np.int64), None
    )
    train_loader = make_graph_loader_v3(train_ds, cfg, True, seed + 11)
    val_loader = make_graph_loader_v3(val_ds, cfg, False, seed + 12)
    test_loader = make_graph_loader_v3(test_ds, cfg, False, seed + 13)

    model = V3GINEClassifier(
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
        num_layers=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        gnn_dropout=cfg.gnn_dropout,
        head_dropout=cfg.head_dropout,
        use_virtual_node=cfg.use_virtual_node,
    ).to(device)
    ema = ModelEMAV3(model, cfg.ema_decay)
    pos_weight = compute_pos_weight(labels[train_idx], cfg.pos_weight_clip).to(device)
    if loss_mode == "asl":
        criterion: nn.Module = AsymmetricLossV3(
            pos_weight,
            gamma_neg=cfg.asl_gamma_neg,
            gamma_pos=cfg.asl_gamma_pos,
            clip=cfg.asl_clip,
        )
    elif loss_mode == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Nieznany loss GNN: {loss_mode}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    parent_indices = torch.tensor(
        [parent for parent, _ in ontology.edges], dtype=torch.long, device=device
    )
    child_indices = torch.tensor(
        [child for _, child in ontology.edges], dtype=torch.long, device=device
    )
    total_steps = max(1, cfg.max_epochs * len(train_loader))
    warmup_steps = int(round(cfg.warmup_epochs * len(train_loader)))
    global_step = 0
    best_metric = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_probs: Optional[np.ndarray] = None
    no_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        if budget.usable_remaining <= 0:
            raise TimeBudgetExceeded("Budżet czasu skończył się w trakcie GNN OOF.")
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        base_loss_sum = 0.0
        hierarchy_loss_sum = 0.0
        sample_count = 0
        last_lr = cfg.learning_rate
        for batch in train_loader:
            batch = move_batch(batch, device)
            last_lr = set_optimizer_lr_v3(
                optimizer,
                cfg.learning_rate,
                global_step,
                total_steps,
                warmup_steps,
                cfg.min_lr_ratio,
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.use_bf16):
                logits = model(batch)
                base_loss = criterion(logits, batch["y"])
                hierarchy_loss = hierarchy_penalty_v3(
                    logits, parent_indices, child_indices
                )
                loss = base_loss + cfg.hierarchy_loss_weight * hierarchy_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Niefinity loss GNN V3.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            ema.update(model)
            batch_size = int(batch["y"].shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            base_loss_sum += float(base_loss.detach().cpu()) * batch_size
            hierarchy_loss_sum += float(hierarchy_loss.detach().cpu()) * batch_size
            sample_count += batch_size
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": loss_sum / max(1, sample_count),
            "base_loss": base_loss_sum / max(1, sample_count),
            "hierarchy_loss": hierarchy_loss_sum / max(1, sample_count),
            "seconds": elapsed,
            "lr": last_lr,
            "budget_remaining_seconds": budget.remaining,
        }
        if epoch % cfg.evaluation_every == 0 or epoch == cfg.max_epochs:
            val_probs, val_y = predict_gnn_v3(
                ema.model, val_loader, device, cfg, with_labels=True
            )
            assert val_y is not None
            metric = quick_calibrated_macro_f1_v3(
                val_y, val_probs, cfg.force_class0
            )
            record["validation_quick_calibrated_macro_f1"] = metric
            LOGGER.info(
                "GNN V3 fold=%d member=%d loss=%s epoka=%d/%d "
                "loss=%.5f quick-F1=%.6f %.2fs",
                fold,
                member,
                loss_mode,
                epoch,
                cfg.max_epochs,
                record["loss"],
                metric,
                elapsed,
            )
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch
                best_state = _cpu_state_dict_v3(ema.model)
                best_val_probs = val_probs.astype(np.float32, copy=True)
                no_improvement = 0
            else:
                no_improvement += 1
        history.append(record)
        if no_improvement >= cfg.patience:
            LOGGER.info(
                "Early stopping GNN V3 fold=%d member=%d po %d ocenach.",
                fold,
                member,
                no_improvement,
            )
            break

    if best_state is None or best_val_probs is None:
        raise RuntimeError("Nie udało się wybrać checkpointu GNN V3.")
    model.load_state_dict(best_state)
    test_probs, _ = predict_gnn_v3(model, test_loader, device, cfg, False)
    checkpoint_path = cfg.checkpoint_dir / (
        f"gine_v3_fold{fold}_member{member}_{loss_mode}.pt"
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "model_state": best_state,
            "architecture": model.architecture(),
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "fold": fold,
            "member": member,
            "loss_mode": loss_mode,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "seed": seed,
            "pipeline_revision": V3_PIPELINE_REVISION,
        },
    )
    report = {
        "fold": fold,
        "member": member,
        "loss_mode": loss_mode,
        "best_epoch": best_epoch,
        "best_quick_calibrated_macro_f1": best_metric,
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "history": history,
    }
    del model, ema
    clear_cuda_cache()
    return best_val_probs, test_probs.astype(np.float32), report


def gnn_member_loss_mode_v3(cfg: Config, member: int) -> str:
    if member == 0:
        return "bce"
    return cfg.gnn_second_loss if member % 2 == 1 else "bce"


def run_gnn_oof_v3(
    train_graphs: Sequence[dict[str, Any]],
    test_graphs: Sequence[dict[str, Any]],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    data_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signature = stable_signature(
        {
            "version": V3_PIPELINE_REVISION + "_gnn_oof",
            "graph_feature_version": V3_GRAPH_FEATURE_VERSION,
            "data_hash": data_hash,
            "ontology_edges_hash": hashlib.sha256(
                np.asarray(ontology.edges, dtype=np.int16).tobytes()
            ).hexdigest(),
            "fold_hash": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
            "num_folds": cfg.num_folds,
            "num_layers": cfg.num_layers,
            "hidden_dim": cfg.hidden_dim,
            "head_hidden_dim": cfg.head_hidden_dim,
            "gnn_dropout": cfg.gnn_dropout,
            "head_dropout": cfg.head_dropout,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "evaluation_every": cfg.evaluation_every,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "gradient_clip": cfg.gradient_clip,
            "pos_weight_clip": cfg.pos_weight_clip,
            "ema_decay": cfg.ema_decay,
            "warmup_epochs": cfg.warmup_epochs,
            "min_lr_ratio": cfg.min_lr_ratio,
            "virtual_node": cfg.use_virtual_node,
            "hierarchy_loss_weight": cfg.hierarchy_loss_weight,
            "members": cfg.gnn_members,
            "second_loss": cfg.gnn_second_loss,
            "asl_gamma_neg": cfg.asl_gamma_neg,
            "asl_gamma_pos": cfg.asl_gamma_pos,
            "asl_clip": cfg.asl_clip,
            "target_atoms_per_batch": cfg.target_atoms_per_batch,
            "eval_atoms_per_batch": cfg.eval_atoms_per_batch,
            "bucket_size": cfg.bucket_size,
            "use_bf16": cfg.use_bf16,
            "force_class0": cfg.force_class0,
            "seed": cfg.seed,
        },
        length=20,
    )
    oof = np.zeros((len(labels), CLASS_COUNT), dtype=np.float32)
    test_sum = np.zeros((len(test_graphs), CLASS_COUNT), dtype=np.float64)
    model_count = 0
    reports: list[dict[str, Any]] = []
    for fold in range(cfg.num_folds):
        train_idx = np.flatnonzero(fold_ids != fold)
        val_idx = np.flatnonzero(fold_ids == fold)
        fold_val_sum = np.zeros((len(val_idx), CLASS_COUNT), dtype=np.float64)
        for member in range(cfg.gnn_members):
            loss_mode = gnn_member_loss_mode_v3(cfg, member)
            prefix = cfg.cache_dir / (
                f"gnn_v3_{signature}_fold{fold}_member{member}_{loss_mode}"
            )
            val_path = Path(str(prefix) + "_val.npy")
            test_path = Path(str(prefix) + "_test.npy")
            meta_path = Path(str(prefix) + "_meta.json")
            loaded = False
            if (
                val_path.exists()
                and test_path.exists()
                and meta_path.exists()
                and not cfg.overwrite_cache
            ):
                try:
                    val_probs = np.load(val_path, allow_pickle=False)
                    test_probs = np.load(test_path, allow_pickle=False)
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    loaded = (
                        probability_matrix_is_valid(
                            val_probs, (len(val_idx), CLASS_COUNT)
                        )
                        and probability_matrix_is_valid(
                            test_probs, (len(test_graphs), CLASS_COUNT)
                        )
                        and meta.get("signature") == signature
                        and meta.get("val_indices_hash")
                        == hashlib.sha256(val_idx.tobytes()).hexdigest()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    loaded = False
            if loaded:
                LOGGER.info(
                    "Wczytano GNN OOF fold=%d member=%d z cache.", fold, member
                )
                report = dict(meta.get("report", {}))
                report["loaded_from_cache"] = True
            else:
                val_probs, test_probs, report = train_gnn_fold_v3(
                    train_graphs,
                    test_graphs,
                    labels,
                    train_idx,
                    val_idx,
                    ontology,
                    cfg,
                    device,
                    budget,
                    fold,
                    member,
                    loss_mode,
                )
                atomic_save_npy(val_path, val_probs.astype(np.float32))
                atomic_save_npy(test_path, test_probs.astype(np.float32))
                write_json(
                    meta_path,
                    {
                        "signature": signature,
                        "val_indices_hash": hashlib.sha256(
                            val_idx.tobytes()
                        ).hexdigest(),
                        "report": report,
                    },
                )
            fold_val_sum += np.asarray(val_probs, dtype=np.float64)
            test_sum += np.asarray(test_probs, dtype=np.float64)
            model_count += 1
            reports.append(report)
        oof[val_idx] = (fold_val_sum / max(1, cfg.gnn_members)).astype(np.float32)
    if np.any(~np.isfinite(oof)) or np.any(~np.isfinite(test_sum)):
        raise FloatingPointError("GNN OOF zawiera NaN/inf.")
    test_probs = (test_sum / max(1, model_count)).astype(np.float32)
    if not probability_matrix_is_valid(oof, (len(labels), CLASS_COUNT)) or not probability_matrix_is_valid(
        test_probs, (len(test_graphs), CLASS_COUNT)
    ):
        raise FloatingPointError("GNN OOF zawiera wartości spoza [0, 1].")
    report = {
        "signature": signature,
        "num_models": model_count,
        "num_folds": cfg.num_folds,
        "members_per_fold": cfg.gnn_members,
        "oof_quick_calibrated_macro_f1": quick_calibrated_macro_f1_v3(
            labels, oof, cfg.force_class0
        ),
        "models": reports,
    }
    write_json(cfg.report_dir / "gnn_oof_report.json", report)
    return oof, test_probs, report

# ---------------------------------------------------------------------------
# V3: binary/count Morgan r=2/r=3 + deskryptory, OOF MLP/linear
# ---------------------------------------------------------------------------


def ecfp_v3_signature(cfg: Config) -> str:
    return stable_signature(
        {
            "version": V3_ECFP_FEATURE_VERSION,
            "bits": cfg.ecfp_bits,
            "count_clip": cfg.ecfp_count_clip,
            "include_chirality": True,
            "descriptors_enabled": cfg.ecfp_descriptors,
            "descriptors": V3_DESCRIPTOR_NAMES if cfg.ecfp_descriptors else [],
            "rdkit": package_version("rdkit"),
        },
        length=20,
    )


def ecfp_descriptor_dim_v3(cfg: Config) -> int:
    return V3_DESCRIPTOR_DIM if cfg.ecfp_descriptors else 0


def ecfp_v3_cache_paths(cfg: Config, combined_hash: str) -> dict[str, Path]:
    signature = ecfp_v3_signature(cfg)
    prefix = cfg.cache_dir / f"ecfp_v3_{combined_hash[:12]}_{signature}"
    return {
        "train_features": Path(str(prefix) + "_train.npy"),
        "test_features": Path(str(prefix) + "_test.npy"),
        "train_desc": Path(str(prefix) + "_train_desc.npy"),
        "test_desc": Path(str(prefix) + "_test_desc.npy"),
        "metadata": Path(str(prefix) + "_metadata.json"),
    }


def build_or_load_ecfp_v3(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    combined_hash: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = ecfp_v3_cache_paths(cfg, combined_hash)
    feature_dim = cfg.ecfp_bits * 3
    required = list(paths.values())
    if all(path.exists() for path in required) and not cfg.overwrite_cache:
        try:
            train_features = np.load(
                paths["train_features"], mmap_mode="r", allow_pickle=False
            )
            test_features = np.load(
                paths["test_features"], mmap_mode="r", allow_pickle=False
            )
            train_desc = np.load(
                paths["train_desc"], mmap_mode="r", allow_pickle=False
            )
            test_desc = np.load(
                paths["test_desc"], mmap_mode="r", allow_pickle=False
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            valid = (
                train_features.shape == (len(train_df), feature_dim)
                and test_features.shape == (len(test_df), feature_dim)
                and train_features.dtype == np.uint8
                and test_features.dtype == np.uint8
                and train_desc.shape == (len(train_df), V3_DESCRIPTOR_DIM)
                and test_desc.shape == (len(test_df), V3_DESCRIPTOR_DIM)
                and np.issubdtype(train_desc.dtype, np.floating)
                and np.issubdtype(test_desc.dtype, np.floating)
                and np.isfinite(train_desc).all()
                and np.isfinite(test_desc).all()
                and metadata.get("signature") == ecfp_v3_signature(cfg)
                and metadata.get("combined_hash") == combined_hash
                and metadata.get("descriptors_enabled") == cfg.ecfp_descriptors
            )
            if valid:
                LOGGER.info("Wczytano cache ECFP V3: %s", paths["metadata"])
                return train_features, test_features, train_desc, test_desc
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        LOGGER.warning("Cache ECFP V3 jest niezgodny — przebudowuję.")

    binary_r2 = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=cfg.ecfp_bits, includeChirality=True
    )
    count_r2 = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=cfg.ecfp_bits, includeChirality=True
    )
    count_r3 = rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=cfg.ecfp_bits, includeChirality=True
    )

    def build(
        dataframe: pd.DataFrame, split_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        features = np.empty((len(dataframe), feature_dim), dtype=np.uint8)
        descriptors = np.empty(
            (len(dataframe), V3_DESCRIPTOR_DIM), dtype=np.float32
        )
        binary_buffer = np.zeros(cfg.ecfp_bits, dtype=np.uint8)
        count_buffer = np.zeros(cfg.ecfp_bits, dtype=np.uint32)
        for row_idx, smiles in enumerate(dataframe["SMILES"].astype(str)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Nieprawidłowy SMILES w {split_name}: {smiles!r}")
            binary_buffer.fill(0)
            DataStructs.ConvertToNumpyArray(
                binary_r2.GetFingerprint(mol), binary_buffer
            )
            features[row_idx, : cfg.ecfp_bits] = binary_buffer
            count_buffer.fill(0)
            DataStructs.ConvertToNumpyArray(
                count_r2.GetCountFingerprint(mol), count_buffer
            )
            features[
                row_idx, cfg.ecfp_bits : 2 * cfg.ecfp_bits
            ] = np.minimum(count_buffer, cfg.ecfp_count_clip).astype(np.uint8)
            count_buffer.fill(0)
            DataStructs.ConvertToNumpyArray(
                count_r3.GetCountFingerprint(mol), count_buffer
            )
            features[
                row_idx, 2 * cfg.ecfp_bits : 3 * cfg.ecfp_bits
            ] = np.minimum(count_buffer, cfg.ecfp_count_clip).astype(np.uint8)
            descriptors[row_idx] = molecular_descriptors_v3(mol)
            if (row_idx + 1) % 5000 == 0 or row_idx + 1 == len(dataframe):
                LOGGER.info(
                    "ECFP V3 %s: %d/%d",
                    split_name,
                    row_idx + 1,
                    len(dataframe),
                )
        return features, descriptors

    train_features, train_desc = build(train_df, "train")
    test_features, test_desc = build(test_df, "test")
    atomic_save_npy(paths["train_features"], train_features)
    atomic_save_npy(paths["test_features"], test_features)
    atomic_save_npy(paths["train_desc"], train_desc)
    atomic_save_npy(paths["test_desc"], test_desc)
    write_json(
        paths["metadata"],
        {
            "signature": ecfp_v3_signature(cfg),
            "combined_hash": combined_hash,
            "feature_version": V3_ECFP_FEATURE_VERSION,
            "bits_per_block": cfg.ecfp_bits,
            "blocks": ["binary_r2", "count_r2", "count_r3"],
            "count_clip": cfg.ecfp_count_clip,
            "feature_dim": feature_dim,
            "descriptor_names": V3_DESCRIPTOR_NAMES,
            "descriptors_enabled": cfg.ecfp_descriptors,
            "train_shape": list(train_features.shape),
            "test_shape": list(test_features.shape),
        },
    )
    return (
        np.load(paths["train_features"], mmap_mode="r", allow_pickle=False),
        np.load(paths["test_features"], mmap_mode="r", allow_pickle=False),
        np.load(paths["train_desc"], mmap_mode="r", allow_pickle=False),
        np.load(paths["test_desc"], mmap_mode="r", allow_pickle=False),
    )


class V3FingerprintDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        descriptors: np.ndarray,
        indices: Sequence[int],
        labels: Optional[np.ndarray],
    ) -> None:
        self.features = features
        self.descriptors = descriptors
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, position: int
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        index = int(self.indices[position])
        label = None if self.labels is None else self.labels[index]
        return self.features[index], self.descriptors[index], label


class ECFPCollatorV3:
    """Picklowalny collator, zgodny z DataLoader spawn (Windows/macOS)."""

    def __init__(
        self,
        bits: int,
        descriptor_mean: np.ndarray,
        descriptor_std: np.ndarray,
        use_descriptors: bool,
    ) -> None:
        self.bits = int(bits)
        self.descriptor_mean = np.asarray(descriptor_mean, dtype=np.float32)
        self.descriptor_std = np.asarray(descriptor_std, dtype=np.float32)
        self.use_descriptors = bool(use_descriptors)

    def __call__(
        self,
        items: Sequence[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    ) -> dict[str, torch.Tensor]:
        if not items:
            raise ValueError("Nie można zbudować pustego batcha ECFP.")
        feature_array = np.stack(
            [np.asarray(item[0], dtype=np.uint8) for item in items]
        )
        x = torch.from_numpy(feature_array).float()
        x[:, self.bits :] = torch.log1p(x[:, self.bits :])
        if self.use_descriptors:
            descriptor_array = np.stack(
                [np.asarray(item[1], dtype=np.float32) for item in items]
            )
            desc = torch.from_numpy(descriptor_array).float()
            mean = torch.from_numpy(self.descriptor_mean)
            std = torch.from_numpy(self.descriptor_std)
            desc = (desc - mean) / std
            desc = torch.nan_to_num(desc, nan=0.0, posinf=10.0, neginf=-10.0)
            x = torch.cat([x, desc], dim=1)
        result = {"x": x}
        if items[0][2] is not None:
            if any(item[2] is None for item in items):
                raise ValueError("Batch ECFP miesza rekordy z etykietami i bez etykiet.")
            result["y"] = torch.from_numpy(
                np.stack(
                    [np.asarray(item[2], dtype=np.float32) for item in items]
                )
            ).float()
        return result


def make_ecfp_collate_v3(
    bits: int,
    descriptor_mean: np.ndarray,
    descriptor_std: np.ndarray,
    use_descriptors: bool,
) -> ECFPCollatorV3:
    return ECFPCollatorV3(
        bits,
        descriptor_mean,
        descriptor_std,
        use_descriptors,
    )


def make_ecfp_loader_v3(
    dataset: V3FingerprintDataset,
    cfg: Config,
    batch_size: int,
    shuffle: bool,
    seed: int,
    descriptor_mean: np.ndarray,
    descriptor_std: np.ndarray,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": make_ecfp_collate_v3(
            cfg.ecfp_bits,
            descriptor_mean,
            descriptor_std,
            cfg.ecfp_descriptors,
        ),
        "drop_last": False,
    }
    if cfg.num_workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs["generator"] = generator
    return DataLoader(**kwargs)


class V3ECFPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if variant == "mlp" else 0
        if variant == "mlp":
            second_hidden = max(512, hidden_dim // 2)
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, second_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(second_hidden, CLASS_COUNT),
            )
        elif variant == "linear":
            self.network = nn.Linear(input_dim, CLASS_COUNT)
        else:
            raise ValueError(f"Nieznany wariant ECFP V3: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@torch.inference_mode()
def predict_ecfp_v3(
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
        x = batch["x"].to(device, non_blocking=True)
        with autocast_context(device, cfg.use_bf16):
            logits = model(x)
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        if with_labels:
            labels.append(batch["y"].numpy())
    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(labels, axis=0) if labels else None,
    )


def train_ecfp_fold_v3(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_desc: np.ndarray,
    test_desc: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    fold: int,
    member: int,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = cfg.seed + 500_000 + fold * 1013 + member * 100_019
    set_seed(seed)
    if cfg.ecfp_descriptors:
        descriptor_mean = np.asarray(
            train_desc[train_idx].mean(axis=0), dtype=np.float32
        )
        descriptor_std = np.asarray(
            train_desc[train_idx].std(axis=0), dtype=np.float32
        )
        descriptor_std = np.where(descriptor_std < 1e-6, 1.0, descriptor_std)
    else:
        descriptor_mean = np.zeros(V3_DESCRIPTOR_DIM, dtype=np.float32)
        descriptor_std = np.ones(V3_DESCRIPTOR_DIM, dtype=np.float32)
    train_ds = V3FingerprintDataset(
        train_features, train_desc, train_idx, labels
    )
    val_ds = V3FingerprintDataset(train_features, train_desc, val_idx, labels)
    test_ds = V3FingerprintDataset(
        test_features,
        test_desc,
        np.arange(len(test_features), dtype=np.int64),
        None,
    )
    train_loader = make_ecfp_loader_v3(
        train_ds,
        cfg,
        cfg.ecfp_batch_size,
        True,
        seed + 21,
        descriptor_mean,
        descriptor_std,
    )
    val_loader = make_ecfp_loader_v3(
        val_ds,
        cfg,
        cfg.ecfp_eval_batch_size,
        False,
        seed + 22,
        descriptor_mean,
        descriptor_std,
    )
    test_loader = make_ecfp_loader_v3(
        test_ds,
        cfg,
        cfg.ecfp_eval_batch_size,
        False,
        seed + 23,
        descriptor_mean,
        descriptor_std,
    )
    input_dim = cfg.ecfp_bits * 3 + ecfp_descriptor_dim_v3(cfg)
    model = V3ECFPClassifier(
        input_dim,
        cfg.ecfp_hidden_dim,
        cfg.head_dropout,
        variant,
    ).to(device)
    ema = ModelEMAV3(model, cfg.ema_decay)
    pos_weight = compute_pos_weight(labels[train_idx], cfg.pos_weight_clip).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    parent_indices = torch.tensor(
        [parent for parent, _ in ontology.edges], dtype=torch.long, device=device
    )
    child_indices = torch.tensor(
        [child for _, child in ontology.edges], dtype=torch.long, device=device
    )
    max_epochs = cfg.ecfp_max_epochs
    total_steps = max(1, max_epochs * len(train_loader))
    warmup_steps = int(round(min(cfg.warmup_epochs, 1.0) * len(train_loader)))
    global_step = 0
    best_metric = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_probs: Optional[np.ndarray] = None
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        if budget.usable_remaining <= 0:
            raise TimeBudgetExceeded("Budżet czasu skończył się w trakcie ECFP OOF.")
        model.train()
        started = time.perf_counter()
        loss_sum = 0.0
        sample_count = 0
        last_lr = cfg.learning_rate
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            last_lr = set_optimizer_lr_v3(
                optimizer,
                cfg.learning_rate,
                global_step,
                total_steps,
                warmup_steps,
                cfg.min_lr_ratio,
            )
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg.use_bf16):
                logits = model(x)
                base_loss = criterion(logits, y)
                hierarchy_loss = hierarchy_penalty_v3(
                    logits, parent_indices, child_indices
                )
                loss = base_loss + cfg.ecfp_hierarchy_loss_weight * hierarchy_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Niefinity loss ECFP V3.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
            ema.update(model)
            loss_sum += float(loss.detach().cpu()) * len(x)
            sample_count += len(x)
        elapsed = time.perf_counter() - started
        val_probs, val_y = predict_ecfp_v3(
            ema.model, val_loader, device, cfg, True
        )
        assert val_y is not None
        metric = quick_calibrated_macro_f1_v3(
            val_y, val_probs, cfg.force_class0
        )
        record = {
            "epoch": epoch,
            "loss": loss_sum / max(1, sample_count),
            "validation_quick_calibrated_macro_f1": metric,
            "seconds": elapsed,
            "lr": last_lr,
        }
        history.append(record)
        LOGGER.info(
            "ECFP V3 fold=%d member=%d variant=%s epoka=%d/%d "
            "loss=%.5f quick-F1=%.6f %.2fs",
            fold,
            member,
            variant,
            epoch,
            max_epochs,
            record["loss"],
            metric,
            elapsed,
        )
        if metric > best_metric + 1e-8:
            best_metric = metric
            best_epoch = epoch
            best_state = _cpu_state_dict_v3(ema.model)
            best_val_probs = val_probs.astype(np.float32, copy=True)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= cfg.ecfp_patience:
            break
    if best_state is None or best_val_probs is None:
        raise RuntimeError("Nie udało się wybrać checkpointu ECFP V3.")
    model.load_state_dict(best_state)
    test_probs, _ = predict_ecfp_v3(model, test_loader, device, cfg, False)
    checkpoint_path = cfg.checkpoint_dir / (
        f"ecfp_v3_fold{fold}_member{member}_{variant}.pt"
    )
    atomic_torch_save(
        checkpoint_path,
        {
            "model_state": best_state,
            "variant": variant,
            "input_dim": input_dim,
            "hidden_dim": cfg.ecfp_hidden_dim if variant == "mlp" else 0,
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "descriptors_enabled": cfg.ecfp_descriptors,
            "fold": fold,
            "member": member,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "seed": seed,
            "pipeline_revision": V3_PIPELINE_REVISION,
        },
    )
    report = {
        "fold": fold,
        "member": member,
        "variant": variant,
        "best_epoch": best_epoch,
        "best_quick_calibrated_macro_f1": best_metric,
        "checkpoint": str(checkpoint_path),
        "descriptors_enabled": cfg.ecfp_descriptors,
        "seed": seed,
        "history": history,
    }
    del model, ema
    clear_cuda_cache()
    return best_val_probs, test_probs.astype(np.float32), report


def ecfp_member_variant_v3(cfg: Config, member: int) -> str:
    if cfg.ecfp_variant in {"mlp", "linear"}:
        return cfg.ecfp_variant
    return "mlp" if member % 2 == 0 else "linear"


def run_ecfp_oof_v3(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_desc: np.ndarray,
    test_desc: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    ontology: Ontology,
    cfg: Config,
    device: torch.device,
    budget: TimeBudget,
    data_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signature = stable_signature(
        {
            "version": V3_PIPELINE_REVISION + "_ecfp_oof",
            "data_hash": data_hash,
            "feature_signature": ecfp_v3_signature(cfg),
            "ontology_edges_hash": hashlib.sha256(
                np.asarray(ontology.edges, dtype=np.int16).tobytes()
            ).hexdigest(),
            "fold_hash": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
            "num_folds": cfg.num_folds,
            "members": cfg.ecfp_members,
            "variant": cfg.ecfp_variant,
            "descriptors_enabled": cfg.ecfp_descriptors,
            "hidden_dim": cfg.ecfp_hidden_dim,
            "max_epochs": cfg.ecfp_max_epochs,
            "patience": cfg.ecfp_patience,
            "batch_size": cfg.ecfp_batch_size,
            "eval_batch_size": cfg.ecfp_eval_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "gradient_clip": cfg.gradient_clip,
            "pos_weight_clip": cfg.pos_weight_clip,
            "ema_decay": cfg.ema_decay,
            "warmup_epochs": min(cfg.warmup_epochs, 1.0),
            "min_lr_ratio": cfg.min_lr_ratio,
            "hierarchy_loss_weight": cfg.ecfp_hierarchy_loss_weight,
            "head_dropout": cfg.head_dropout,
            "use_bf16": cfg.use_bf16,
            "force_class0": cfg.force_class0,
            "seed": cfg.seed,
        },
        length=20,
    )
    oof = np.zeros((len(labels), CLASS_COUNT), dtype=np.float32)
    test_sum = np.zeros((len(test_features), CLASS_COUNT), dtype=np.float64)
    model_count = 0
    reports: list[dict[str, Any]] = []
    for fold in range(cfg.num_folds):
        train_idx = np.flatnonzero(fold_ids != fold)
        val_idx = np.flatnonzero(fold_ids == fold)
        fold_val_sum = np.zeros((len(val_idx), CLASS_COUNT), dtype=np.float64)
        for member in range(cfg.ecfp_members):
            variant = ecfp_member_variant_v3(cfg, member)
            prefix = cfg.cache_dir / (
                f"ecfp_model_v3_{signature}_fold{fold}_member{member}_{variant}"
            )
            val_path = Path(str(prefix) + "_val.npy")
            test_path = Path(str(prefix) + "_test.npy")
            meta_path = Path(str(prefix) + "_meta.json")
            loaded = False
            if (
                val_path.exists()
                and test_path.exists()
                and meta_path.exists()
                and not cfg.overwrite_cache
            ):
                try:
                    val_probs = np.load(val_path, allow_pickle=False)
                    test_probs = np.load(test_path, allow_pickle=False)
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    loaded = (
                        probability_matrix_is_valid(
                            val_probs, (len(val_idx), CLASS_COUNT)
                        )
                        and probability_matrix_is_valid(
                            test_probs, (len(test_features), CLASS_COUNT)
                        )
                        and meta.get("signature") == signature
                        and meta.get("val_indices_hash")
                        == hashlib.sha256(val_idx.tobytes()).hexdigest()
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    loaded = False
            if loaded:
                LOGGER.info(
                    "Wczytano ECFP OOF fold=%d member=%d z cache.", fold, member
                )
                report = dict(meta.get("report", {}))
                report["loaded_from_cache"] = True
            else:
                val_probs, test_probs, report = train_ecfp_fold_v3(
                    train_features,
                    test_features,
                    train_desc,
                    test_desc,
                    labels,
                    train_idx,
                    val_idx,
                    ontology,
                    cfg,
                    device,
                    budget,
                    fold,
                    member,
                    variant,
                )
                atomic_save_npy(val_path, val_probs.astype(np.float32))
                atomic_save_npy(test_path, test_probs.astype(np.float32))
                write_json(
                    meta_path,
                    {
                        "signature": signature,
                        "val_indices_hash": hashlib.sha256(
                            val_idx.tobytes()
                        ).hexdigest(),
                        "report": report,
                    },
                )
            fold_val_sum += np.asarray(val_probs, dtype=np.float64)
            test_sum += np.asarray(test_probs, dtype=np.float64)
            model_count += 1
            reports.append(report)
        oof[val_idx] = (fold_val_sum / max(1, cfg.ecfp_members)).astype(np.float32)
    if np.any(~np.isfinite(oof)) or np.any(~np.isfinite(test_sum)):
        raise FloatingPointError("ECFP OOF zawiera NaN/inf.")
    test_probs = (test_sum / max(1, model_count)).astype(np.float32)
    if not probability_matrix_is_valid(oof, (len(labels), CLASS_COUNT)) or not probability_matrix_is_valid(
        test_probs, (len(test_features), CLASS_COUNT)
    ):
        raise FloatingPointError("ECFP OOF zawiera wartości spoza [0, 1].")
    report = {
        "signature": signature,
        "num_models": model_count,
        "num_folds": cfg.num_folds,
        "members_per_fold": cfg.ecfp_members,
        "oof_quick_calibrated_macro_f1": quick_calibrated_macro_f1_v3(
            labels, oof, cfg.force_class0
        ),
        "models": reports,
    }
    write_json(cfg.report_dir / "ecfp_oof_report.json", report)
    return oof, test_probs, report

# ---------------------------------------------------------------------------
# V3: kNN Tanimoto, blend logitów i exact-match override
# ---------------------------------------------------------------------------


def numpy_bits_to_rdkit_fps_v3(bits_matrix: np.ndarray) -> list[Any]:
    fingerprints: list[Any] = []
    for row_idx in range(len(bits_matrix)):
        row = np.asarray(bits_matrix[row_idx], dtype=np.uint8)
        packed = np.packbits(row, bitorder="little").tobytes()
        fingerprints.append(DataStructs.CreateFromBinaryText(packed))
        if (row_idx + 1) % 10000 == 0 or row_idx + 1 == len(bits_matrix):
            LOGGER.info("Konwersja fingerprintów RDKit: %d/%d", row_idx + 1, len(bits_matrix))
    return fingerprints


def knn_predict_queries_v3(
    query_fps: Sequence[Any],
    reference_fps: Sequence[Any],
    reference_labels: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    labels_float = np.asarray(reference_labels, dtype=np.float32)
    prior = labels_float.mean(axis=0)
    k = min(max(1, cfg.knn_k), len(reference_fps))

    def predict_one(query_fp: Any) -> tuple[np.ndarray, float]:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps),
            dtype=np.float32,
        )
        if k < len(similarities):
            top_indices = np.argpartition(similarities, -k)[-k:]
        else:
            top_indices = np.arange(len(similarities), dtype=np.int64)
        top_similarities = similarities[top_indices]
        order = np.argsort(-top_similarities, kind="stable")
        top_indices = top_indices[order]
        top_similarities = top_similarities[order]
        weights = np.maximum(
            top_similarities - float(cfg.knn_min_similarity), 0.0
        ) ** float(cfg.knn_power)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            weights = np.maximum(top_similarities, 0.0) ** float(cfg.knn_power)
            weight_sum = float(weights.sum())
        if weight_sum <= 1e-12:
            # Dla zapytania bez żadnego podobnego sąsiada zwracamy prior zamiast
            # niemal zerowego wektora (szczególnie gdy knn_prior_weight == 0).
            return prior.astype(np.float32, copy=True), float(top_similarities[0])
        prior_weight = float(cfg.knn_prior_weight)
        denominator = weight_sum + prior_weight
        numerator = (
            weights.astype(np.float32) @ labels_float[top_indices]
            + prior_weight * prior
        )
        prediction = numerator / denominator
        return prediction.astype(np.float32), float(top_similarities[0])

    predictions = np.empty((len(query_fps), CLASS_COUNT), dtype=np.float32)
    maximum_similarities = np.empty(len(query_fps), dtype=np.float32)
    workers = max(1, int(cfg.knn_workers))
    if workers == 1:
        iterator = map(predict_one, query_fps)
        executor = None
    else:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="tanimoto-knn"
        )
        iterator = executor.map(predict_one, query_fps)
    try:
        for row_idx, (prediction, max_similarity) in enumerate(iterator):
            predictions[row_idx] = prediction
            maximum_similarities[row_idx] = max_similarity
            if (row_idx + 1) % 2000 == 0 or row_idx + 1 == len(query_fps):
                LOGGER.info("kNN Tanimoto: %d/%d", row_idx + 1, len(query_fps))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return predictions, maximum_similarities


def run_knn_oof_v3(
    train_features: np.ndarray,
    test_features: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
    data_hash: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    signature = stable_signature(
        {
            "version": V3_PIPELINE_REVISION + "_knn_oof",
            "data_hash": data_hash,
            "feature_signature": ecfp_v3_signature(cfg),
            "fold_hash": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
            "k": cfg.knn_k,
            "power": cfg.knn_power,
            "min_similarity": cfg.knn_min_similarity,
            "prior_weight": cfg.knn_prior_weight,
        },
        length=20,
    )
    oof_path = cfg.cache_dir / f"knn_v3_{signature}_oof.npy"
    test_path = cfg.cache_dir / f"knn_v3_{signature}_test.npy"
    report_path = cfg.cache_dir / f"knn_v3_{signature}_report.json"
    if (
        oof_path.exists()
        and test_path.exists()
        and report_path.exists()
        and not cfg.overwrite_cache
    ):
        try:
            oof = np.load(oof_path, allow_pickle=False)
            test_probs = np.load(test_path, allow_pickle=False)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                probability_matrix_is_valid(oof, (len(labels), CLASS_COUNT))
                and probability_matrix_is_valid(
                    test_probs, (len(test_features), CLASS_COUNT)
                )
                and report.get("signature") == signature
            ):
                LOGGER.info("Wczytano kNN OOF z cache.")
                report["loaded_from_cache"] = True
                return oof, test_probs, report
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    train_fps = numpy_bits_to_rdkit_fps_v3(
        train_features[:, : cfg.ecfp_bits]
    )
    test_fps = numpy_bits_to_rdkit_fps_v3(test_features[:, : cfg.ecfp_bits])
    oof = np.zeros((len(labels), CLASS_COUNT), dtype=np.float32)
    fold_similarity: list[dict[str, Any]] = []
    for fold in range(cfg.num_folds):
        train_idx = np.flatnonzero(fold_ids != fold)
        val_idx = np.flatnonzero(fold_ids == fold)
        reference_fps = [train_fps[int(index)] for index in train_idx]
        query_fps = [train_fps[int(index)] for index in val_idx]
        fold_probs, max_similarity = knn_predict_queries_v3(
            query_fps, reference_fps, labels[train_idx], cfg
        )
        oof[val_idx] = fold_probs
        fold_similarity.append(
            {
                "fold": fold,
                "mean_max_similarity": float(max_similarity.mean()),
                "median_max_similarity": float(np.median(max_similarity)),
                "p95_max_similarity": float(np.quantile(max_similarity, 0.95)),
            }
        )
    test_probs, test_max_similarity = knn_predict_queries_v3(
        test_fps, train_fps, labels, cfg
    )
    report = {
        "signature": signature,
        "k": cfg.knn_k,
        "power": cfg.knn_power,
        "min_similarity": cfg.knn_min_similarity,
        "prior_weight": cfg.knn_prior_weight,
        "workers": cfg.knn_workers,
        "elapsed_seconds": time.perf_counter() - started,
        "oof_quick_calibrated_macro_f1": quick_calibrated_macro_f1_v3(
            labels, oof, cfg.force_class0
        ),
        "fold_similarity": fold_similarity,
        "test_similarity": {
            "mean_max_similarity": float(test_max_similarity.mean()),
            "median_max_similarity": float(np.median(test_max_similarity)),
            "p95_max_similarity": float(np.quantile(test_max_similarity, 0.95)),
            "max": float(test_max_similarity.max()),
        },
    }
    atomic_save_npy(oof_path, oof)
    atomic_save_npy(test_path, test_probs)
    write_json(report_path, report)
    write_json(cfg.report_dir / "knn_oof_report.json", report)
    return oof, test_probs, report


def _search_two_component_blend_v3(
    labels: np.ndarray,
    first_oof: np.ndarray,
    second_oof: np.ndarray,
    first_test: np.ndarray,
    second_test: np.ndarray,
    frequency_counts: np.ndarray,
    cfg: Config,
    name_first: str,
    name_second: str,
    max_second_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    first_logits = probability_to_logit(first_oof)
    second_logits = probability_to_logit(second_oof)
    first_test_logits = probability_to_logit(first_test)
    second_test_logits = probability_to_logit(second_test)
    global_grid = np.linspace(0.0, max_second_weight, 21, dtype=np.float32)
    global_scores: list[float] = []
    for second_weight in global_grid:
        logits = (1.0 - second_weight) * first_logits + second_weight * second_logits
        score = quick_calibrated_macro_f1_v3(
            labels, logit_to_probability(logits), cfg.force_class0
        )
        global_scores.append(score)
    best_global_index = int(np.argmax(global_scores))
    best_global_weight = float(global_grid[best_global_index])

    buckets = class_frequency_bucket(frequency_counts)
    class_second_weights = np.full(CLASS_COUNT, best_global_weight, dtype=np.float32)
    bucket_details: list[dict[str, Any]] = []
    bucket_grid = np.linspace(0.0, max_second_weight, 11, dtype=np.float32)
    regularization = float(np.clip(cfg.blend_bucket_regularization, 0.0, 1.0))
    for bucket in range(5):
        mask = buckets == bucket
        if not mask.any():
            continue
        bucket_scores: list[float] = []
        for second_weight in bucket_grid:
            logits = (
                (1.0 - second_weight) * first_logits[:, mask]
                + second_weight * second_logits[:, mask]
            )
            score = quick_calibrated_macro_f1_v3(
                labels[:, mask],
                logit_to_probability(logits),
                False,
            )
            bucket_scores.append(score)
        raw_bucket_weight = float(bucket_grid[int(np.argmax(bucket_scores))])
        regularized_weight = (
            regularization * best_global_weight
            + (1.0 - regularization) * raw_bucket_weight
        )
        class_second_weights[mask] = regularized_weight
        bucket_details.append(
            {
                "bucket": bucket,
                "class_count": int(mask.sum()),
                "raw_best_second_weight": raw_bucket_weight,
                "regularized_second_weight": regularized_weight,
                "best_score": float(max(bucket_scores)),
            }
        )
    oof_logits = (
        (1.0 - class_second_weights[None, :]) * first_logits
        + class_second_weights[None, :] * second_logits
    )
    test_logits = (
        (1.0 - class_second_weights[None, :]) * first_test_logits
        + class_second_weights[None, :] * second_test_logits
    )
    report = {
        "first_component": name_first,
        "second_component": name_second,
        "global_second_weight": best_global_weight,
        "global_search": [
            {"second_weight": float(weight), "score": float(score)}
            for weight, score in zip(global_grid, global_scores)
        ],
        "bucket_weights": bucket_details,
        "class_second_weights": class_second_weights.tolist(),
        "result_quick_calibrated_macro_f1": quick_calibrated_macro_f1_v3(
            labels, logit_to_probability(oof_logits), cfg.force_class0
        ),
    }
    return logit_to_probability(oof_logits), logit_to_probability(test_logits), report


def blend_components_v3(
    labels: np.ndarray,
    oof_components: dict[str, np.ndarray],
    test_components: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not oof_components:
        raise RuntimeError("Brak komponentów do ensemble.")
    frequency_counts = labels.sum(axis=0).astype(np.int64)
    reports: list[dict[str, Any]] = []

    preferred_order = [name for name in ["gnn", "ecfp", "knn"] if name in oof_components]
    preferred_order.extend(
        name for name in oof_components if name not in preferred_order
    )
    current_name = preferred_order[0]
    current_oof = np.asarray(oof_components[current_name], dtype=np.float32)
    current_test = np.asarray(test_components[current_name], dtype=np.float32)
    for next_name in preferred_order[1:]:
        max_weight = 0.40 if next_name == "knn" else 1.0
        current_oof, current_test, blend_report = _search_two_component_blend_v3(
            labels,
            current_oof,
            oof_components[next_name],
            current_test,
            test_components[next_name],
            frequency_counts,
            cfg,
            current_name,
            next_name,
            max_second_weight=max_weight,
        )
        reports.append(blend_report)
        current_name = f"blend({current_name},{next_name})"
    component_scores = {
        name: quick_calibrated_macro_f1_v3(
            labels, probabilities, cfg.force_class0
        )
        for name, probabilities in oof_components.items()
    }
    report = {
        "component_scores_quick": component_scores,
        "blend_steps": reports,
        "selected_expression": current_name,
        "final_quick_calibrated_macro_f1": quick_calibrated_macro_f1_v3(
            labels, current_oof, cfg.force_class0
        ),
    }
    return current_oof, current_test, report


def apply_exact_match_override_v3(
    probabilities: np.ndarray,
    train_canonical: np.ndarray,
    test_canonical: np.ndarray,
    labels: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = np.asarray(probabilities, dtype=np.float32).copy()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, canonical in enumerate(train_canonical):
        groups[str(canonical)].append(index)
    matched_rows: list[int] = []
    conflicting_matches = 0
    strength = float(np.clip(cfg.exact_match_strength, 0.0, 1.0))
    for test_idx, canonical in enumerate(test_canonical):
        matching = groups.get(str(canonical))
        if not matching:
            continue
        matching_labels = labels[np.asarray(matching, dtype=np.int64)]
        target = matching_labels.mean(axis=0).astype(np.float32)
        if len(np.unique(matching_labels, axis=0)) > 1:
            conflicting_matches += 1
        # Lekki Laplace smoothing chroni przed dokładnymi 0/1 w blendzie logitów.
        target = np.clip(target, 0.001, 0.999)
        output[test_idx] = (
            strength * target + (1.0 - strength) * output[test_idx]
        )
        matched_rows.append(test_idx)
    report = {
        "enabled": cfg.exact_match_override,
        "strength": strength,
        "matched_test_records": len(matched_rows),
        "conflicting_train_groups_used": conflicting_matches,
        "matched_test_indices": matched_rows,
    }
    return output, report


# ---------------------------------------------------------------------------
# V3: walidacja konfiguracji, self-test i główny pipeline
# ---------------------------------------------------------------------------


def validate_config_v3(cfg: Config) -> None:
    validate_config(cfg)
    issues: list[str] = []
    if cfg.model not in {"ensemble", "auto", "gine", "ecfp"}:
        issues.append("model musi być ensemble, auto, gine albo ecfp.")
    if cfg.num_folds < 2:
        issues.append("num_folds musi być >= 2.")
    if cfg.num_workers < 0 or cfg.feature_workers < 1:
        issues.append("num_workers musi być >= 0, a feature_workers >= 1.")
    if cfg.debug_sample < 0:
        issues.append("debug_sample nie może być ujemny.")
    if cfg.gnn_members < 1 or cfg.ecfp_members < 1:
        issues.append("Liczba członków ensemble musi być >= 1.")
    if cfg.gnn_second_loss not in {"bce", "asl"}:
        issues.append("gnn_second_loss musi być bce albo asl.")
    if not 0.0 <= cfg.ema_decay < 1.0:
        issues.append("ema_decay musi należeć do [0, 1).")
    if cfg.warmup_epochs < 0 or not 0 < cfg.min_lr_ratio <= 1:
        issues.append("Nieprawidłowy warmup/min_lr_ratio.")
    if cfg.hierarchy_loss_weight < 0 or cfg.ecfp_hierarchy_loss_weight < 0:
        issues.append("Wagi straty hierarchicznej nie mogą być ujemne.")
    if not 1 <= cfg.ecfp_count_clip <= 255:
        issues.append(
            "ecfp_count_clip musi należeć do [1, 255], ponieważ cechy są zapisywane jako uint8."
        )
    if cfg.ecfp_variant not in {"auto", "mlp", "linear"}:
        issues.append("ecfp_variant musi być auto, mlp albo linear.")
    if cfg.asl_gamma_neg < 0 or cfg.asl_gamma_pos < 0:
        issues.append("Parametry gamma straty ASL nie mogą być ujemne.")
    if not 0.0 <= cfg.asl_clip < 1.0:
        issues.append("asl_clip musi należeć do [0, 1).")
    if cfg.knn_enabled and cfg.ecfp_bits % 8 != 0:
        issues.append("Przy włączonym kNN ecfp_bits musi być podzielne przez 8.")
    if cfg.knn_k < 1 or cfg.knn_power <= 0 or cfg.knn_workers < 1:
        issues.append("Nieprawidłowa konfiguracja kNN.")
    if cfg.knn_prior_weight < 0:
        issues.append("knn_prior_weight nie może być ujemne.")
    if not 0.0 <= cfg.knn_min_similarity <= 1.0:
        issues.append("knn_min_similarity musi należeć do [0, 1].")
    if not 0.0 <= cfg.exact_match_strength <= 1.0:
        issues.append("exact_match_strength musi należeć do [0, 1].")
    if not 0.0 <= cfg.blend_bucket_regularization <= 1.0:
        issues.append("blend_bucket_regularization musi należeć do [0, 1].")
    if cfg.hierarchy_f1_tolerance < 0.0:
        issues.append("hierarchy_f1_tolerance nie może być ujemne.")
    if issues:
        raise ValueError("Nieprawidłowa konfiguracja V3:\n- " + "\n- ".join(issues))


def component_flags_v3(cfg: Config) -> tuple[bool, bool, bool]:
    use_gnn = cfg.model in {"ensemble", "auto", "gine"}
    use_ecfp = cfg.model in {"ensemble", "auto", "ecfp"}
    use_knn = cfg.knn_enabled and cfg.model in {"ensemble", "auto"}
    return use_gnn, use_ecfp, use_knn


def run_self_test_v3() -> None:
    LOGGER.info("Uruchamiam self-test V3.")
    ontology = Ontology(
        ["class_0", "class_1", "class_2"],
        [("class_0", "class_1"), ("class_1", "class_2")],
    )
    pred = np.asarray([[0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    closed = ontology.binary_closure(pred)
    assert closed.tolist() == [[1, 1, 1], [1, 0, 0]]
    graph_a = molecule_to_graph_v3("CCO")
    graph_b = molecule_to_graph_v3("c1ccccc1")
    labels_a = np.zeros(CLASS_COUNT, dtype=np.float32)
    labels_b = np.zeros(CLASS_COUNT, dtype=np.float32)
    batch = collate_graphs_v3([(graph_a, labels_a), (graph_b, labels_b)])
    descriptor_matrix = np.stack([graph_a["graph_desc"], graph_b["graph_desc"]])
    descriptor_mean = descriptor_matrix.mean(axis=0)
    descriptor_std = np.where(descriptor_matrix.std(axis=0) < 1e-6, 1.0, descriptor_matrix.std(axis=0))
    model = V3GINEClassifier(
        descriptor_mean,
        descriptor_std,
        num_layers=2,
        hidden_dim=32,
        head_hidden_dim=64,
        use_virtual_node=True,
    )
    logits = model(batch)
    assert logits.shape == (2, CLASS_COUNT)
    logits.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    y = np.asarray([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8)
    p = np.asarray(
        [[0.9, 0.2, 0.1], [0.9, 0.8, 0.2], [0.9, 0.8, 0.7]],
        dtype=np.float32,
    )
    info = optimize_thresholds_exact_v3(y, p, True, y.sum(axis=0), True)
    assert np.asarray(info["regularized_thresholds"]).shape == (3,)
    group_counts = np.asarray([[1, 0], [1, 1], [0, 1], [1, 0]], dtype=np.int64)
    group_sizes = np.ones(4, dtype=np.int64)
    folds = _greedy_group_multilabel_folds(group_counts, group_sizes, 2, 42)
    assert set(folds.tolist()) == {0, 1}

    # Jawne ustawienie wariantu ECFP musi być respektowane dla każdego członka.
    base_cfg = Config(
        train=Path("train.parquet"),
        test=Path("test.parquet"),
        ontology=Path("ontology.obo"),
        sample_submission=None,
        output_dir=Path("artifacts"),
        submission_path=Path("submission.parquet"),
        ecfp_variant="linear",
        ecfp_members=2,
    )
    assert ecfp_member_variant_v3(base_cfg, 0) == "linear"
    assert ecfp_member_variant_v3(base_cfg, 1) == "linear"

    # --no-ecfp-descriptors ma realnie zmieniać wymiar wejścia.
    base_cfg.ecfp_descriptors = False
    assert ecfp_descriptor_dim_v3(base_cfg) == 0
    collate = make_ecfp_collate_v3(
        bits=8,
        descriptor_mean=np.zeros(V3_DESCRIPTOR_DIM, dtype=np.float32),
        descriptor_std=np.ones(V3_DESCRIPTOR_DIM, dtype=np.float32),
        use_descriptors=False,
    )
    feature = np.zeros(24, dtype=np.uint8)
    descriptor = np.zeros(V3_DESCRIPTOR_DIM, dtype=np.float32)
    ecfp_batch = collate([(feature, descriptor, labels_a)])
    assert ecfp_batch["x"].shape == (1, 24)

    # Collator musi działać z multiprocessing spawn.
    import pickle

    pickle.dumps(collate)

    # Liczba batchy ma odpowiadać rzeczywistemu pakowaniu atomów.
    sampler = AtomBudgetBatchSampler(
        np.asarray([6, 6, 6]), target_atoms=10, shuffle=False, seed=42
    )
    assert len(sampler) == 3
    assert len(list(iter(sampler))) == 3

    # Jawny model ECFP nie może po cichu uruchamiać osobnego komponentu kNN.
    base_cfg.model = "ecfp"
    base_cfg.knn_enabled = True
    assert component_flags_v3(base_cfg) == (False, True, False)

    # Przy zerowym podobieństwie kNN wraca do prior, także dla prior_weight=0.
    base_cfg.knn_k = 2
    base_cfg.knn_workers = 1
    base_cfg.knn_prior_weight = 0.0
    base_cfg.knn_min_similarity = 0.2
    query_fp = DataStructs.ExplicitBitVect(8)
    reference_fps = [DataStructs.ExplicitBitVect(8), DataStructs.ExplicitBitVect(8)]
    reference_labels = np.zeros((2, CLASS_COUNT), dtype=np.uint8)
    reference_labels[0, 0] = 1
    reference_labels[1, 1] = 1
    knn_prediction, _ = knn_predict_queries_v3(
        [query_fp], reference_fps, reference_labels, base_cfg
    )
    assert np.allclose(knn_prediction[0, :2], [0.5, 0.5])

    # Submission nie może po cichu obcinać prawdopodobieństw do int8.
    invalid_prediction = np.zeros((1, CLASS_COUNT), dtype=np.float32)
    invalid_prediction[0, 0] = 0.5
    tiny_test = pd.DataFrame({"mol_id": ["mol_0"], "SMILES": ["C"]})
    try:
        generate_and_validate_submission(
            tiny_test,
            None,
            invalid_prediction,
            Ontology(CLASS_NAMES, []),
            {"closure": False},
            base_cfg,
        )
    except ValueError as exc:
        assert "binarne" in str(exc)
    else:
        raise AssertionError("Niebinarny submission został zaakceptowany.")

    LOGGER.info("Self-test V3 zakończony powodzeniem.")


def run_pipeline(cfg: Config) -> None:
    validate_config_v3(cfg)
    ensure_dirs(cfg)
    setup_logging(cfg.output_dir)
    resolved_config = jsonable(cfg)
    write_json(cfg.output_dir / "resolved_config.json", resolved_config)
    config_digest = hashlib.sha256(
        json.dumps(resolved_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(cfg.report_dir / "config_hash.json", {"sha256": config_digest})

    if cfg.self_test:
        run_self_test_v3()
        return
    required_paths = [cfg.train, cfg.test, cfg.ontology]
    if cfg.sample_submission is not None:
        required_paths.append(cfg.sample_submission)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Brak plików wejściowych:\n- " + "\n- ".join(missing))
    if cfg.dry_run:
        LOGGER.info("Dry-run V3 OK. Ścieżki istnieją, katalogi utworzone.")
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

    with timer.stage("wczytanie i walidacja danych"):
        train_df, test_df, sample_df, data_report = load_and_validate_data(cfg)
        labels = train_df[CLASS_NAMES].to_numpy(dtype=np.uint8, copy=True)
        effective_train_hash = effective_dataset_hash(train_hash, train_df, cfg, "train")
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
                "effective_train": effective_train_hash,
                "effective_test": effective_test_hash,
                "combined_effective_data": combined_hash,
            },
        )

    with timer.stage("ontologia"):
        ontology = load_ontology(cfg)
        training_violations = ontology.inconsistency_counts(labels)
        if cfg.force_class0 and not bool((labels[:, 0] == 1).all()):
            raise ValueError("class_0 nie jest dodatnia dla wszystkich rekordów.")
        write_json(
            cfg.report_dir / "training_hierarchy.json",
            {
                "binary_inconsistencies": training_violations,
                "class0_universal": bool((labels[:, 0] == 1).all()),
                "class0_is_root": len(ontology.parents[0]) == 0,
            },
        )

    with timer.stage(f"kanonikalizacja i {cfg.num_folds}-fold OOF"):
        train_canonical = get_canonical_smiles_v3(
            train_df, cfg, effective_train_hash, "train"
        )
        test_canonical = get_canonical_smiles_v3(
            test_df, cfg, effective_test_hash, "test"
        )
        duplicate_report = duplicate_conflict_report(
            train_canonical, labels, train_df["mol_id"].astype(str).tolist()
        )
        write_json(cfg.report_dir / "duplicate_conflicts.json", duplicate_report)
        fold_ids, folds_report = make_or_load_folds_v3(
            train_df, labels, train_canonical, cfg, effective_train_hash
        )

    use_gnn, use_ecfp, use_knn = component_flags_v3(cfg)
    need_ecfp_features = use_ecfp or use_knn
    component_failures: list[str] = []
    component_reports: dict[str, Any] = {}
    oof_components: dict[str, np.ndarray] = {}
    test_components: dict[str, np.ndarray] = {}

    train_features: Optional[np.ndarray] = None
    test_features: Optional[np.ndarray] = None
    train_desc: Optional[np.ndarray] = None
    test_desc: Optional[np.ndarray] = None
    if need_ecfp_features:
        with timer.stage("ECFP V3 binary/count r2/r3 + deskryptory"):
            train_features, test_features, train_desc, test_desc = build_or_load_ecfp_v3(
                train_df, test_df, cfg, combined_hash
            )

    if use_ecfp:
        assert train_features is not None and test_features is not None
        assert train_desc is not None and test_desc is not None
        try:
            with timer.stage(f"{cfg.num_folds}-fold OOF ECFP V3"):
                ecfp_oof, ecfp_test, ecfp_report = run_ecfp_oof_v3(
                    train_features,
                    test_features,
                    train_desc,
                    test_desc,
                    labels,
                    fold_ids,
                    ontology,
                    cfg,
                    device,
                    budget,
                    combined_hash,
                )
            oof_components["ecfp"] = ecfp_oof
            test_components["ecfp"] = ecfp_test
            component_reports["ecfp"] = ecfp_report
        except Exception as exc:
            if cfg.model in {"ecfp", "ensemble"}:
                raise
            message = f"ECFP V3 nie powiódł się: {exc}"
            LOGGER.exception(message)
            component_failures.append(message)
            clear_cuda_cache()

    if use_gnn:
        try:
            with timer.stage("cache grafów V3"):
                train_graphs, test_graphs, _ = build_or_load_graphs_v3(
                    train_df, test_df, cfg, combined_hash
                )
            with timer.stage(f"{cfg.num_folds}-fold OOF GINE V3"):
                gnn_oof, gnn_test, gnn_report = run_gnn_oof_v3(
                    train_graphs,
                    test_graphs,
                    labels,
                    fold_ids,
                    ontology,
                    cfg,
                    device,
                    budget,
                    combined_hash,
                )
            oof_components["gnn"] = gnn_oof
            test_components["gnn"] = gnn_test
            component_reports["gnn"] = gnn_report
        except Exception as exc:
            if cfg.model in {"gine", "ensemble"}:
                raise
            message = f"GNN V3 nie powiódł się: {exc}"
            LOGGER.exception(message)
            component_failures.append(message)
            clear_cuda_cache()

    if use_knn:
        assert train_features is not None and test_features is not None
        try:
            with timer.stage(f"{cfg.num_folds}-fold OOF kNN Tanimoto"):
                knn_oof, knn_test, knn_report = run_knn_oof_v3(
                    train_features,
                    test_features,
                    labels,
                    fold_ids,
                    cfg,
                    combined_hash,
                )
            oof_components["knn"] = knn_oof
            test_components["knn"] = knn_test
            component_reports["knn"] = knn_report
        except Exception as exc:
            if cfg.model == "ensemble":
                raise
            message = f"kNN V3 nie powiódł się: {exc}"
            LOGGER.exception(message)
            component_failures.append(message)

    if not oof_components:
        raise RuntimeError(
            "Żaden komponent V3 nie ukończył treningu/inferencji. "
            + " | ".join(component_failures)
        )

    with timer.stage("OOF blend w przestrzeni logitów"):
        blended_oof, blended_test, blend_report = blend_components_v3(
            labels, oof_components, test_components, cfg
        )
        write_json(cfg.report_dir / "blend_report.json", blend_report)

    with timer.stage("dokładne progi i warianty DAG"):
        hierarchy_selection = evaluate_hierarchy_variants_v3(
            labels,
            blended_oof,
            ontology,
            cfg,
            frequency_positive_counts=labels.sum(axis=0),
        )
        write_json(
            cfg.report_dir / "hierarchy_variants_v3.json", hierarchy_selection
        )
        selected_variant = hierarchy_selection["selected"]
        thresholds = np.asarray(selected_variant["thresholds"], dtype=np.float32)
        write_json(
            cfg.checkpoint_dir / "thresholds_v3.json",
            {
                "selected_variant": hierarchy_selection["selected_variant"],
                "thresholds": thresholds.tolist(),
                "method": "exact_f1_per_class_with_logit_shrinkage",
                "force_class0": cfg.force_class0,
            },
        )

    exact_match_report = {
        "enabled": False,
        "matched_test_records": 0,
    }
    test_probabilities = blended_test
    if cfg.exact_match_override:
        with timer.stage("exact-match override"):
            test_probabilities, exact_match_report = apply_exact_match_override_v3(
                test_probabilities,
                train_canonical,
                test_canonical,
                labels,
                cfg,
            )
            write_json(
                cfg.report_dir / "exact_match_override.json", exact_match_report
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
            atomic_save_npy(
                cfg.checkpoint_dir / "oof_probabilities_blended_v3.npy",
                blended_oof.astype(np.float32),
            )
            atomic_save_npy(
                cfg.checkpoint_dir / "test_probabilities_blended_v3.npy",
                test_probabilities.astype(np.float32),
            )
            for name, probabilities in oof_components.items():
                atomic_save_npy(
                    cfg.checkpoint_dir / f"oof_probabilities_{name}_v3.npy",
                    probabilities.astype(np.float32),
                )
        submission_report = generate_and_validate_submission(
            test_df,
            sample_df,
            predictions,
            ontology,
            selected_variant,
            cfg,
            report_filename="submission_validation_v3.json",
        )

    component_exact_scores: dict[str, float] = {}
    for name, probabilities in oof_components.items():
        threshold_info = optimize_thresholds_exact_v3(
            labels,
            probabilities,
            cfg.force_class0,
            labels.sum(axis=0),
            cfg.threshold_shrinkage,
        )
        component_exact_scores[name] = float(
            threshold_info["macro_f1_regularized_thresholds"]
        )
    final_report = {
        "pipeline_version": V3_PIPELINE_REVISION,
        "model_request": cfg.model,
        "completed_components": list(oof_components),
        "component_failures": component_failures,
        "component_exact_oof_macro_f1": component_exact_scores,
        "component_reports": component_reports,
        "blend": blend_report,
        "selected_hierarchy_variant": hierarchy_selection["selected_variant"],
        "selected_oof_macro_f1": selected_variant["macro_f1"],
        "exact_match_override": exact_match_report,
        "folds": folds_report,
        "data": data_report,
        "submission": submission_report,
        "stage_seconds": timer.stages,
        "total_elapsed_seconds": timer.elapsed,
        "time_budget": budget.summary(),
        "config_sha256": config_digest,
    }
    write_json(cfg.report_dir / "final_report.json", final_report)
    LOGGER.info("Gotowe V3: %s", cfg.submission_path)
    LOGGER.info(
        "Komponenty=%s | OOF macro-F1=%.6f | wariant=%s | czas=%.1fs",
        ",".join(oof_components),
        selected_variant["macro_f1"],
        hierarchy_selection["selected_variant"],
        timer.elapsed,
    )



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "GNN V3: 5-fold grouped OOF, GINE+ECFP+kNN ensemble, "
            "dokładne progi i postprocessing DAG dla ChEBI."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("chebi_dataset_train.parquet"))
    parser.add_argument("--test", type=Path, default=Path("chebi_dataset_test_empty.parquet"))
    parser.add_argument("--ontology", type=Path, default=Path("chebi_classes.obo"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("chebi_submission_example.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_v3"))
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument(
        "--model",
        choices=["ensemble", "auto", "gine", "ecfp"],
        default="ensemble",
        help=(
            "ensemble uruchamia GINE, ECFP i kNN; auto toleruje awarie "
            "komponentów; ecfp uruchamia wyłącznie model fingerprintowy."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 lub cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=(
            0
            if os.name == "nt"
            else min(8, max(0, (os.cpu_count() or 4) // 2))
        ),
    )
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--target-atoms-per-batch", type=int, default=8000)
    parser.add_argument("--eval-atoms-per-batch", type=int, default=16000)
    parser.add_argument("--bucket-size", type=int, default=256)

    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--gnn-dropout", type=float, default=0.15)
    parser.add_argument("--head-dropout", type=float, default=0.20)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--small-gnn-max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--evaluation-every", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--pos-weight-clip", type=float, default=8.0)
    parser.add_argument("--working-threshold", type=float, default=0.30)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--use-virtual-node", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gnn-members", type=int, default=2)
    parser.add_argument("--gnn-second-loss", choices=["bce", "asl"], default="asl")
    parser.add_argument("--hierarchy-loss-weight", type=float, default=0.03)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)

    parser.add_argument("--ecfp-bits", type=int, default=4096)
    parser.add_argument("--ecfp-radius", type=int, default=2)
    parser.add_argument("--ecfp-count-clip", type=int, default=15)
    parser.add_argument("--ecfp-hidden-dim", type=int, default=1536)
    parser.add_argument("--ecfp-batch-size", type=int, default=512)
    parser.add_argument("--ecfp-eval-batch-size", type=int, default=1024)
    parser.add_argument("--ecfp-max-epochs", type=int, default=25)
    parser.add_argument("--ecfp-patience", type=int, default=5)
    parser.add_argument(
        "--ecfp-descriptors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ecfp-variant", choices=["auto", "mlp", "linear"], default="mlp"
    )
    parser.add_argument("--ecfp-members", type=int, default=2)
    parser.add_argument("--ecfp-hierarchy-loss-weight", type=float, default=0.01)

    parser.add_argument("--knn-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--knn-k", type=int, default=25)
    parser.add_argument("--knn-power", type=float, default=3.0)
    parser.add_argument("--knn-min-similarity", type=float, default=0.20)
    parser.add_argument("--knn-prior-weight", type=float, default=1.0)
    parser.add_argument(
        "--knn-workers",
        type=int,
        default=min(8, max(1, (os.cpu_count() or 4) // 2)),
    )
    parser.add_argument(
        "--exact-match-override", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--exact-match-strength", type=float, default=0.98)
    parser.add_argument("--blend-bucket-regularization", type=float, default=0.50)
    parser.add_argument(
        "--threshold-shrinkage", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--force-class0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hierarchy-f1-tolerance", type=float, default=0.01)
    parser.add_argument("--use-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-limit-minutes", type=float, default=180.0)
    parser.add_argument("--buffer-minutes", type=float, default=10.0)
    parser.add_argument("--graph-preprocess-limit-minutes", type=float, default=60.0)
    parser.add_argument("--benchmark-fast-seconds", type=float, default=120.0)
    parser.add_argument("--benchmark-slow-seconds", type=float, default=240.0)
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
    parser.add_argument(
        "--save-probabilities", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prepare-ecfp-fallback", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--checkpoint-candidates", type=int, default=10)
    parser.add_argument(
        "--final-strategy",
        choices=["fold_ensemble", "validation_checkpoint", "full_retrain"],
        default="fold_ensemble",
        help="V3 używa fold_ensemble; pozostałe wartości są zachowane dla kompatybilności.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    submission_path = (
        args.submission.resolve()
        if args.submission is not None
        else output_dir / "submission.parquet"
    )
    sample_submission = args.sample_submission
    if sample_submission is not None and str(sample_submission).lower() in {
        "none",
        "null",
        "-",
    }:
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
        num_folds=args.num_folds,
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
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay=args.ema_decay,
        use_virtual_node=args.use_virtual_node,
        gnn_members=args.gnn_members,
        gnn_second_loss=args.gnn_second_loss,
        hierarchy_loss_weight=args.hierarchy_loss_weight,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_clip=args.asl_clip,
        ecfp_bits=args.ecfp_bits,
        ecfp_radius=args.ecfp_radius,
        ecfp_count_clip=args.ecfp_count_clip,
        ecfp_hidden_dim=args.ecfp_hidden_dim,
        ecfp_batch_size=args.ecfp_batch_size,
        ecfp_eval_batch_size=args.ecfp_eval_batch_size,
        ecfp_max_epochs=args.ecfp_max_epochs,
        ecfp_patience=args.ecfp_patience,
        ecfp_descriptors=args.ecfp_descriptors,
        ecfp_variant=args.ecfp_variant,
        ecfp_members=args.ecfp_members,
        ecfp_hierarchy_loss_weight=args.ecfp_hierarchy_loss_weight,
        knn_enabled=args.knn_enabled,
        knn_k=args.knn_k,
        knn_power=args.knn_power,
        knn_min_similarity=args.knn_min_similarity,
        knn_prior_weight=args.knn_prior_weight,
        knn_workers=max(1, args.knn_workers),
        exact_match_override=args.exact_match_override,
        exact_match_strength=args.exact_match_strength,
        blend_bucket_regularization=args.blend_bucket_regularization,
        threshold_shrinkage=args.threshold_shrinkage,
        force_class0=args.force_class0,
        hierarchy_f1_tolerance=max(0.0, args.hierarchy_f1_tolerance),
        final_time_safety_factor=max(1.0, args.final_time_safety_factor),
        final_inference_reserve_seconds=max(
            0.0, args.final_inference_reserve_seconds
        ),
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
