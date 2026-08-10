# Steel Defect Inspection

Python package foundation for a steel-surface defect segmentation service.

## Prerequisites

Use 64-bit PowerShell and Python 3.11 or 3.12. This project is pinned for
CUDA 12.1-compatible PyTorch.

## Setup and test

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -v
```

The `data/` and `artifacts/` directories contain local datasets and generated
outputs and are intentionally not tracked by Git.
