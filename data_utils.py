# data_utils.py
#
# Add drop-in support for a dataset named "sparwious" (also accepts the alias "spawrious")
# WITHOUT reorganizing files on disk.
#
# It mirrors the Waterbirds integration:
#   - get_data("sparwious_train"/"sparwious_val"/"sparwious_test") -> Dataset
#   - get_targets_only("sparwious_*") -> list[int]
#   - get_filenames_only("sparwious_*") -> list[str]
#   - get_groups_only("sparwious_*") -> list[int] where group_id = y * 2 + place
#
# Assumptions for Sparwious:
#   * Root:   /dsi/scratch/users/eisenbr2/spawrious/all/
#   * Images: /dsi/scratch/users/eisenbr2/spawrious/all/images/...
#   * CSV:    /dsi/scratch/users/eisenbr2/spawrious/all/metadata.csv
#   * CSV columns: img_id, img_filename, y, split, place
#                  split_map = {0: train, 1: val, 2: test}
#
# Also keeps the Waterbirds support you already use (per-split dirs with filtered metadata.csv).

import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms, models


# ------------------------------
# Paths & label files
# ------------------------------

DATASET_ROOTS = {
    # Existing paths you already had:
    "imagenet_train": "YOUR_PATH/CLS-LOC/train/",
    "imagenet_val": "YOUR_PATH/ImageNet_val/",
    "cub_train": "data/CUB/train",
    "cub_val": "data/CUB/test",

    # Waterbirds: expects pre-made split folders with per-split metadata.csv
    "waterbirds_train": "/dsi/dsai-lab/Ran/cbm/waterbirds/splits/train",
    "waterbirds_val":   "/dsi/dsai-lab/Ran/cbm/waterbirds/splits/test",
    "waterbirds_test":  "/dsi/dsai-lab/Ran/cbm/waterbirds/splits/test",

    # Sparwious (a.k.a. Spawrious): single root; we filter by metadata split
    # Note: get_data handles the split; this is just the base directory.
    "sparwious_root": "/dsi/scratch/users/eisenbr2/spawrious/all/",
}

LABEL_FILES = {
    "places365": "data/categories_places365_clean.txt",
    "imagenet":  "data/imagenet_classes.txt",
    "cifar10":   "data/cifar10_classes.txt",
    "cifar100":  "data/cifar100_classes.txt",
    "cub":       "data/cub_classes.txt",
    "waterbirds":"data/waterbirds_classes.txt",
    # Provided by you:
    "sparwious": "/home/eng/eisenbr2/Label-free-CBM-2/data/sparwious_classes.txt",
    # Alias key so either spelling works when code does LABEL_FILES[args.dataset]
    "spawrious": "/home/eng/eisenbr2/Label-free-CBM-2/data/sparwious_classes.txt",
}

# Add a label-file entry so args.dataset='cifar10c' works
LABEL_FILES.update({
    "cifar10c": "data/cifar10_classes.txt",   # 10 lines with CIFAR-10 class names
})

# Default config for CIFAR-10-C; will be overwritten by set_cifar10c_options(...)
CIFAR10C_CONFIG = {
    "root": "/dsi/dsai-lab/Ran/CIFAR-10-C-P/CIFAR-10-C",  # your path
    "corruptions": ["gaussian_noise", "shot_noise"],      # can override from CLI
    "severities": [1, 2, 3, 4, 5],                       # can override from CLI
}
def set_cifar10c_options(root=None, corruptions=None, severities=None):
    if root is not None: CIFAR10C_CONFIG["root"] = root
    if corruptions is not None: CIFAR10C_CONFIG["corruptions"] = list(corruptions)
    if severities is not None: CIFAR10C_CONFIG["severities"] = list(severities)


SPLIT_MAP = {0: "train", 1: "val", 2: "test"}  # for reference


# ------------------------------
# Common helpers
# ------------------------------

def _canon_dataset_name(name: str) -> str:
    """Normalize dataset name spelling (sparwious <-> spawrious)."""
    if name.startswith("spawrious"):
        return name.replace("spawrious", "sparwious", 1)
    return name

def _lstrip_slash(p: str) -> str:
    return p[1:] if p.startswith("/") else p

def _sniffed_reader(fp):
    """CSV/TSV tolerant DictReader."""
    sample = fp.read(4096)
    fp.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.get_dialect("excel")
    return csv.DictReader(fp, dialect=dialect)


# ------------------------------
# Preprocess (used elsewhere)
# ------------------------------

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=target_mean, std=target_std),
    ])


# ------------------------------
# WATERBIRDS dataset (unchanged)
# ------------------------------

class WaterbirdsDataset(Dataset):
    """
    Expects split directories with per-split metadata.csv and images under <split>/<y>/<img_filename>.
    """
    classes = ["landbird", "waterbird"]
    class_to_idx = {"landbird": 0, "waterbird": 1}

    def __init__(self, root: str, transform=None, metadata_name: str = "metadata.csv"):
        self.root = Path(root)
        self.transform = transform
        self.meta = self.root / metadata_name
        if not self.meta.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.meta}")

        self.filenames: List[str] = []
        self.targets:   List[int] = []
        self.places:    List[int] = []
        self.groups:    List[int] = []  # (1 - y) * 2 + (1 - place)
        self.paths:     List[Path] = []

        with self.meta.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"img_filename", "y", "place"}
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"metadata.csv missing columns: {sorted(miss)}")

            for row in reader:
                try:
                    y = int(row["y"]); place = int(row["place"])
                except Exception:
                    continue
                rel = _lstrip_slash(row["img_filename"])
                abs_path = self.root / str(y) / rel

                self.filenames.append(row["img_filename"])
                self.targets.append(y)
                self.places.append(place)
                self.groups.append((1 - y) * 2 + (1 - place))
                self.paths.append(abs_path)

        assert len(self.paths) == len(self.targets) == len(self.filenames) == len(self.groups)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            if self.transform is not None:
                im = self.transform(im)
        return im, self.targets[idx]


# ------------------------------
# SPARWIOUS dataset (no reorg)
# ------------------------------

class SparwiousDataset(Dataset):
    """
    Loads directly from a single root directory using metadata.csv with split column.
    - root points to: /dsi/scratch/users/eisenbr2/spawrious/all/
    - metadata.csv there contains: img_filename, y, split, place
    - Images live at: root / <img_filename>  (e.g., 'images/008365.png')
    - Group ID (per your earlier code): group_id = y * 2 + place
    """
    def __init__(self, root: str, split_code: int, transform=None, metadata_name: str = "metadata.csv"):
        assert split_code in (0, 1, 2), "split_code must be 0(train),1(val),2(test)"
        self.root = Path(root)
        self.transform = transform
        self.meta = self.root / metadata_name
        if not self.meta.exists():
            raise FileNotFoundError(f"metadata.csv not found at {self.meta}")

        self.filenames: List[str] = []
        self.targets:   List[int] = []
        self.places:    List[int] = []
        self.groups:    List[int] = []  # y * 2 + place
        self.paths:     List[Path] = []

        with self.meta.open("r", newline="") as f:
            reader = _sniffed_reader(f)
            need = {"img_filename", "y", "split", "place"}
            miss = need - set(reader.fieldnames or [])
            if miss:
                raise ValueError(f"metadata.csv missing columns: {sorted(miss)}")

            for row in reader:
                try:
                    y = int(row["y"]); place = int(row["place"]); sp = int(row["split"])
                except Exception:
                    continue
                if sp != split_code:
                    continue
                rel = _lstrip_slash(row["img_filename"])  # e.g., 'images/000123.png'
                abs_path = self.root / rel

                self.filenames.append(row["img_filename"])
                self.targets.append(y)
                self.places.append(place)
                self.groups.append(y * 2 + place)
                self.paths.append(abs_path)

        assert len(self.paths) == len(self.targets) == len(self.filenames) == len(self.groups)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path).convert("RGB") as im:
            if self.transform is not None:
                im = self.transform(im)
        return im, self.targets[idx]

import numpy as np

class Cifar10CSubset(Dataset):
    """
    Loads a subset of CIFAR-10-C given a list of corruptions and severities.
    - root: directory that contains e.g. gaussian_noise.npy, shot_noise.npy, labels.npy
    - Each corruption file has shape (50000, 32, 32, 3) with 10k images per severity (1..5) in order.
    - We concatenate [ (corr, sev) for corr in corruptions for sev in severities ] in that order.
    Exposes:
      - .targets (List[int])
      - .filenames (synthetic, for compatibility)
      - .segments: List[dict{name, severity, start, end}]  inclusive-exclusive indices for each (corr,sev)
    """
    def __init__(self, root: str, corruptions, severities, transform=None):
        self.root = Path(root)
        self.corruptions = list(corruptions)
        self.severities = list(severities)
        self.transform = transform

        labels = np.load(self.root / "labels.npy")  # (50000,)
        self._segments = []   # book-keeping for metrics
        self._items = []      # list of (arr_ref, idx_in_arr)
        self.targets = []     # aligned with _items
        self.filenames = []   # optional: fake names for compatibility

        # Load each corruption mmap'ed to keep RAM small
        per_sev = 10000
        for corr in self.corruptions:
            arr = np.load(self.root / f"{corr}.npy", mmap_mode="r")  # (50000, 32,32,3), uint8
            for sev in self.severities:
                s = (sev - 1) * per_sev
                e = sev * per_sev
                start = len(self._items)
                for i in range(s, e):
                    self._items.append((arr, i))
                    self.targets.append(int(labels[i]))
                    # fabricate a filename-like id (purely informational)
                    self.filenames.append(f"{corr}/severity{sev}/{i - s:05d}.png")
                end = len(self._items)
                self._segments.append({"name": corr, "severity": sev, "start": start, "end": end})

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        arr, j = self._items[idx]
        img = Image.fromarray(arr[j])  # HWC uint8 -> PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

    # Expose for metrics
    @property
    def segments(self):
        return list(self._segments)

def get_cifar10c_segments(dataset_name: str):
    if dataset_name not in ("cifar10c_val", "cifar10c_test"):
        return None
    ds = _as_dataset(dataset_name)
    return getattr(ds, "segments", None)


# ------------------------------
# Factory & convenience accessors
# ------------------------------

def get_data(dataset_name: str, preprocess=None):
    """
    Returns dataset instance for:
      - waterbirds_{train,val,test} -> WaterbirdsDataset
      - sparwious_{train,val,test}  -> SparwiousDataset
      - (legacy alias) spawrious_*  -> SparwiousDataset
      - others in DATASET_ROOTS     -> ImageFolder fallback
    """
    dataset_name = _canon_dataset_name(dataset_name)

    # Waterbirds custom
    if dataset_name in ("waterbirds_train", "waterbirds_val", "waterbirds_test"):
        root = DATASET_ROOTS[dataset_name]
        return WaterbirdsDataset(root=root, transform=preprocess)

    # Sparwious custom (single root + split filter)
    if dataset_name in ("sparwious_train", "sparwious_val", "sparwious_test"):
        split_code = {"sparwious_train": 0, "sparwious_val": 1, "sparwious_test": 2}[dataset_name]
        root = DATASET_ROOTS["sparwious_root"]
        return SparwiousDataset(root=root, split_code=split_code, transform=preprocess)
    
        # CIFAR-10-C support
    if dataset_name == "cifar10c_train":
        # train on clean CIFAR-10
        return datasets.CIFAR10(
            root=os.path.expanduser("~/.cache"),
            download=True, train=True, transform=preprocess
        )

    if dataset_name in ("cifar10c_val", "cifar10c_test"):
        # evaluate on the chosen CIFAR-10-C subset
        cfg = CIFAR10C_CONFIG
        return Cifar10CSubset(
            root=cfg["root"],
            corruptions=cfg["corruptions"],
            severities=cfg["severities"],
            transform=preprocess
        )


    # Torchvision built-ins (if you still rely on them here)
    if dataset_name == "cifar100_train":
        return datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=preprocess)
    if dataset_name == "cifar100_val":
        return datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
    if dataset_name == "cifar10_train":
        return datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True, transform=preprocess)
    if dataset_name == "cifar10_val":
        return datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False, transform=preprocess)
    if dataset_name == "places365_train":
        try:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True, transform=preprocess)
        except RuntimeError:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False, transform=preprocess)
    if dataset_name == "places365_val":
        try:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True, transform=preprocess)
        except RuntimeError:
            return datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False, transform=preprocess)

    # Fallback ImageFolder for other custom names in DATASET_ROOTS
    if dataset_name in DATASET_ROOTS:
        return datasets.ImageFolder(DATASET_ROOTS[dataset_name], transform=preprocess)

    # Alias path: if someone calls spawrious_* explicitly
    alias = _canon_dataset_name(dataset_name)
    if alias != dataset_name:
        return get_data(alias, preprocess)

    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def _as_dataset(dataset_name: str):
    """Internal helper returning a dataset with a safe default preprocess."""
    preprocess = get_resnet_imagenet_preprocess()
    return get_data(dataset_name, preprocess=preprocess)


def get_targets_only(dataset_name: str) -> List[int]:
    ds = _as_dataset(dataset_name)
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "samples"):
        return [y for _, y in ds.samples]
    if hasattr(ds, "imgs"):
        return [y for _, y in ds.imgs]
    raise AttributeError(f"{dataset_name} dataset does not expose targets")


def get_filenames_only(dataset_name: str) -> List[str]:
    ds = _as_dataset(dataset_name)
    if hasattr(ds, "filenames"):
        return list(ds.filenames)
    if hasattr(ds, "samples"):
        root = Path(DATASET_ROOTS[dataset_name])
        return [str(Path(p).relative_to(root)) for p, _ in ds.samples]
    if hasattr(ds, "imgs"):
        root = Path(DATASET_ROOTS[dataset_name])
        return [str(Path(p).relative_to(root)) for p, _ in ds.imgs]
    raise AttributeError(f"{dataset_name} dataset does not expose filenames")


def get_groups_only(dataset_name: str) -> Optional[List[int]]:
    """
    Returns group IDs aligned with dataset order.
      * waterbirds_* : (1 - y) * 2 + (1 - place)
      * sparwious_*  : y * 2 + place
    """
    dataset_name = _canon_dataset_name(dataset_name)
    if dataset_name.startswith("waterbirds_") or dataset_name.startswith("sparwious_"):
        ds = _as_dataset(dataset_name)
        return list(getattr(ds, "groups", []))
    if dataset_name.startswith("spawrious_"):
        return get_groups_only(_canon_dataset_name(dataset_name))
    return None
