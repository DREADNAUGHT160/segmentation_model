"""
Download a sample (or full) RUGD dataset from the HuggingFace mirror.

Source: https://huggingface.co/datasets/WilliamBonilla62/RUGD
Mirror maintained by the community — all credit to the original RUGD authors:
  Wigness et al., "A RUGD Dataset for Autonomous Navigation and Visual
  Perception in Unstructured Outdoor Environments," IROS 2019.

Usage
-----
Download a small sample (default — fast, ~60 MB):
    python data/download_rugd.py --output data/RUGD

Download the full dataset (~3 GB):
    python data/download_rugd.py --output data/RUGD --full

Download specific sequences only:
    python data/download_rugd.py --output data/RUGD --sequences creek park-2

The downloaded structure matches the expected RUGD layout:
    data/RUGD/
    ├── RUGD_frames-with-annotations/
    │   ├── creek/    (images)
    │   └── park-2/   (images)
    └── RUGD_annotations/
        ├── creek/    (label masks — pixel value = class index)
        └── park-2/   (label masks)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # segmentation_model/

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

HF_REPO_ID = "WilliamBonilla62/RUGD"
HF_REPO_TYPE = "dataset"

# Default sample: small sequences that give representative coverage
# creek  → training split (836 images, use first N)
# trail-7 → val/test split (290 images, use first N)
DEFAULT_SAMPLE: dict[str, int] = {
    "creek": 30,      # train
    "trail-7": 15,    # val + test
    "trail-10": 30,   # train (smallest full sequence = 49 imgs, use most)
}


def list_sequence_files(seq: str) -> tuple[list[str], list[str]]:
    """Return sorted (image_paths, label_paths) for a given sequence."""
    all_files = list(list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE))
    prefix_img = f"RUGD_frames-with-annotations/{seq}/"
    prefix_lbl = f"RUGD_annotations/{seq}/"
    imgs = sorted(f for f in all_files if f.startswith(prefix_img) and f.endswith(".png"))
    lbls = sorted(f for f in all_files if f.startswith(prefix_lbl) and f.endswith(".png"))
    return imgs, lbls


def download_file(repo_path: str, local_root: Path) -> None:
    """Download a single file from the HF repo, preserving directory structure."""
    local_path = local_root / repo_path
    if local_path.exists():
        return  # already downloaded

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=repo_path,
        local_dir=str(local_root),
    )
    # hf_hub_download writes to local_dir/filename — nothing more to do


def download_sequence(
    seq: str,
    local_root: Path,
    max_files: Optional[int] = None,
) -> int:
    """Download images + labels for one RUGD sequence.

    Args:
        seq:        Sequence name, e.g. ``"creek"``.
        local_root: Root directory where RUGD will be saved.
        max_files:  Cap the number of image/label pairs downloaded.
                    ``None`` downloads all.

    Returns:
        Number of image+label pairs downloaded.
    """
    imgs, lbls = list_sequence_files(seq)

    # Align by filename stem (safety: ensure img/lbl pairs match)
    img_stems = {Path(p).stem: p for p in imgs}
    lbl_stems = {Path(p).stem: p for p in lbls}
    common = sorted(set(img_stems) & set(lbl_stems))

    if max_files is not None:
        common = common[:max_files]

    if not common:
        print(f"  WARNING: no matching image/label pairs found for '{seq}'")
        return 0

    pairs = [(img_stems[s], lbl_stems[s]) for s in common]
    print(f"  Downloading {len(pairs)} pairs from '{seq}'...")

    for img_path, lbl_path in tqdm(pairs, desc=seq, unit="pair"):
        download_file(img_path, local_root)
        download_file(lbl_path, local_root)

    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RUGD dataset from HuggingFace")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_DIR, "data", "RUGD"),
        help="Root directory to download RUGD into (default: <project>/data/RUGD)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full dataset instead of a sample (~3 GB, all sequences)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        metavar="SEQ",
        help="Download specific sequences only, e.g. --sequences creek park-2",
    )
    parser.add_argument(
        "--max_per_seq",
        type=int,
        default=None,
        help="Max images per sequence (overrides default sample sizes)",
    )
    args = parser.parse_args()

    local_root = Path(args.output)
    print(f"Download destination: {local_root.resolve()}")

    # --- Determine what to download ---
    if args.full:
        # All sequences, no cap
        all_files = list(list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE))
        seqs = sorted({
            p.split("/")[1]
            for p in all_files
            if p.startswith("RUGD_annotations/") and "/" in p[len("RUGD_annotations/"):]
        })
        plan = {seq: None for seq in seqs}
        print(f"Full download: {len(seqs)} sequences")

    elif args.sequences:
        cap = args.max_per_seq  # may be None = all
        plan = {seq: cap for seq in args.sequences}

    else:
        # Default sample
        plan = {seq: n for seq, n in DEFAULT_SAMPLE.items()}
        if args.max_per_seq is not None:
            plan = {seq: args.max_per_seq for seq in plan}
        total = sum(v for v in plan.values() if v)
        print(f"Sample download: {len(plan)} sequences, ~{total} image/label pairs")

    # --- Download ---
    total_pairs = 0
    for seq, max_n in plan.items():
        total_pairs += download_sequence(seq, local_root, max_files=max_n)

    print(f"\nDownloaded {total_pairs} image/label pairs to {local_root.resolve()}")
    print("\nNext: update configs/config.yaml → data.root_dir to point here, then train.")


if __name__ == "__main__":
    main()
