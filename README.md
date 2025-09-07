# yolov8-instance-segmentation

![Language](https://img.shields.io/badge/language-Python-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Release](https://img.shields.io/github/v/release/AlbertoMarquillas/gesture-mouse-control)
![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow)

## Overview
This project demonstrates **instance segmentation using YOLOv8** on custom datasets. The repository contains training and inference scripts, dataset configuration, and documentation tailored for reproducibility and portfolio presentation. It is designed to run smoothly on **Windows (PowerShell examples provided)** but can be adapted for Linux/macOS.  

Instance segmentation is a core computer vision task where each object instance in an image is not only detected but also segmented into pixel-level masks. This project highlights the power of YOLOv8 for segmentation and provides a clean, professional structure suitable for recruiters, collaborators, or research purposes.

---

## Repository Structure
```
yolov8-instance-segmentation/
├─ src/
│  ├─ train.py          # training entry script
│  └─ predict.py        # inference entry script
├─ configs/
│  └─ config.yaml       # dataset config (classes, paths, parameters)
├─ data/                # datasets (ignored; see data/README.md)
├─ models/              # weights (ignored; see models/README.md)
├─ notebooks/           # optional notebooks
├─ build/               # outputs and temporary files (ignored)
├─ docs/                # project documentation, figures, assets
├─ requirements.txt     # Python dependencies (excluding torch/torchvision)
└─ LICENSE
```

---

## Getting Started

### Prerequisites
- Python 3.11 (recommended)
- Windows 10/11 (PowerShell examples included)
- Git
- Virtual environment management (venv)

### 1) Clone the repository
```powershell
git clone https://github.com/AlbertoMarquillas/yolov8-instance-segmentation.git
cd yolov8-instance-segmentation
```

### 2) Create and activate environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install PyTorch (choose build for your hardware)
Select the correct command for your CPU/GPU from the official PyTorch site: https://pytorch.org/get-started/locally/

**CPU-only example:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA example (adjust cu121 to your CUDA version):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4) Install project dependencies
```powershell
pip install -r requirements.txt
```

---

## Dataset Format (YOLOv8 segmentation)
- **Structure:**
```
data/
└─ DATASET_NAME/
   ├─ images/
   │  ├─ train/
   │  └─ val/
   └─ labels/
      ├─ train/
      └─ val/
```

- **Labels:**
Each `.txt` file contains one line per object instance, formatted as polygon annotations:
```
class x1 y1 x2 y2 ... xn yn
```
Where `(x, y)` are normalized polygon coordinates in `[0, 1]`.

- Update `configs/config.yaml` with dataset paths and class names.

---

## Training

### Using the provided script
```powershell
python .\src\train.py `
  --data .\configs\config.yaml `
  --model yolov8n-seg.pt `
  --img 640 `
  --epochs 100 `
  --batch 8 `
  --workers 4
```

### Using Ultralytics CLI directly
```powershell
yolo task=segment mode=train `
  model=yolov8n-seg.pt `
  data=./configs/config.yaml `
  imgsz=640 epochs=100 batch=8 workers=4
```

Outputs (runs, checkpoints) will be stored in `runs/` by default.

---

## Inference / Prediction
```powershell
python .\src\predict.py `
  --weights .\models\best.pt `
  --source .\data\DATASET_NAME\images\val `
  --conf 0.25 `
  --save
```

Or directly with YOLOv8 CLI:
```powershell
yolo task=segment mode=predict `
  model=./models/best.pt `
  source=./data/DATASET_NAME/images/val `
  conf=0.25 save=True
```

---

## Exporting Models
Export trained weights to other formats (ONNX, TorchScript, TensorRT, CoreML, etc.):
```powershell
yolo task=segment mode=export model=./models/best.pt format=onnx
```

---

## Datasets & Models
- Datasets are **not included** in the repository. See [`data/README.md`](data/README.md).
- Pretrained and trained weights are **not included**. See [`models/README.md`](models/README.md).

---

## Features
- Professional structure suitable for a portfolio project.
- Clean separation of code (`src/`), configs (`configs/`), and assets (`docs/`).
- Training and inference using both scripts and Ultralytics CLI.
- Clear dataset and model handling (excluded from git).
- PowerShell examples for Windows users.
- Conventional Commits and semantic versioning.

---

## What I Learned
- Setting up reproducible deep learning environments.
- Organizing computer vision projects for clarity and reusability.
- Using YOLOv8 for instance segmentation on custom datasets.
- Working with polygon-based segmentation annotations.
- Creating recruiter-friendly repositories.

---

## Roadmap
- [ ] Add a unified `src/main.py` CLI entry point.
- [ ] Provide example notebooks for visualization and EDA.
- [ ] Add lightweight test cases in `test/` for CI.
- [ ] Include export scripts for multiple deployment targets.

---

## License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
