"""
Download encoder weights for offline HPC use.

Run this on your laptop (which has internet access), then copy the
resulting hf_cache/ folder to /home/phd_li/hf_cache/ via WinSCP.

Encoder used: resnet101  (config: model.encoder = "resnet101")
smp uses timm internally; timm fetches resnet101 ImageNet weights
from HuggingFace Hub at: timm/resnet101.a1h_in1k

Usage:
    pip install huggingface_hub
    python download_weights.py

Then transfer:
    WinSCP: upload ./hf_cache/  →  /home/phd_li/hf_cache/
"""

from huggingface_hub import snapshot_download

# timm's ResNet-101 ImageNet weights (used by smp for encoder="resnet101")
print("Downloading timm/resnet101.a1h_in1k ...")
snapshot_download(
    repo_id="timm/resnet101.a1h_in1k",
    repo_type="model",
    cache_dir="./hf_cache",
)
print("Done. Transfer ./hf_cache/ to /home/phd_li/hf_cache/ via WinSCP.")
