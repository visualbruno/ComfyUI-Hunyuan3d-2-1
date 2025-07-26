# 🌀 ComfyUI Wrapper for [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

> **ComfyUI integration** for Tencent's powerful **Hunyuan3D-2.1** model. Supports textured 3D generation with optional high-quality UV mapping.

---

## 📦 Repository & Models

* **GitHub:** [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)
* **Model Weights (HuggingFace):**
  👉 [Main page](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main)

### 🔧 Required Checkpoints

Place the following checkpoints into the corresponding folders under your `ComfyUI` directory:

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   └── hunyuan3d-dit-v2-1.ckpt
│   ├── vae/
│   │   └── hunyuan3d-vae-v2-1.ckpt
```

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.12** and **Torch >= 2.6.0 + cu126**. Compatible with the latest ComfyUI Portable release.

### 1. Install Python Dependencies

For a standard Python environment:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/requirements.txt
```

For **ComfyUI Portable**:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\requirements.txt
```

---

### 2. Install or Compile Texture Generation Modules

Two critical C++ extensions need to be installed: the **custom rasterizer** and the **differentiable renderer**.

#### Option A: Use Precompiled Wheels (Recommended)

For standard Python:

```bash
pip install wheels\custom_rasterizer-0.1-cp312-cp312-win_amd64.whl
```

For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\wheels\custom_rasterizer-0.1-cp312-cp312-win_amd64.whl
```

> If you are running **Torch 2.6.0 + cu126**, use the updated wheel instead:

```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Hunyuan3d-2-1\wheels\custom_rasterizer-0.1.0+torch260.cuda126-cp312-cp312-win_amd64.whl
```

---

#### Option B: Manual Compilation (for advanced users)

```bash
# Compile custom rasterizer
cd ComfyUI/custom_nodes/ComfyUI-Hunyuan3d-2-1/hy3dpaint/custom_rasterizer
python setup.py install

# Compile differentiable renderer
cd ../DifferentiableRenderer
python setup.py install
```

---

## 🩻 Optional: Fix UV Wrapping for High Poly Meshes (Patched Xatlas)

This upgrade improves UV unwrapping stability for complex meshes.

```bash
# Step 1: Uninstall existing xatlas
python_embeded\python.exe -m pip uninstall xatlas

# Step 2: Clone updated xatlas-python wrapper
cd ComfyUI_windows_portable
git clone --recursive https://github.com/mworchel/xatlas-python.git

# Step 3: Replace internal xatlas source
cd xatlas-python\extern
del /s /q xatlas
git clone --recursive https://github.com/jpcy/xatlas

# Step 4: Patch source file
# In xatlas-python/extern/xatlas/source/xatlas/xatlas.cpp:
#   Line 6774: change `#if 0` → `//#if 0`
#   Line 6778: change `#endif` → `//#endif`

# Step 5: Install patched xatlas wrapper
cd ../../..
python_embeded\python.exe -m pip install .\xatlas-python\
```

```python
python_embeded\python.exe -m pip uninstall -y xatlas; `
cd ComfyUI_windows_portable; `
if (Test-Path xatlas-python) { Remove-Item xatlas-python -Recurse -Force }; `
git clone --recursive https://github.com/mworchel/xatlas-python.git; `
cd xatlas-python\extern; `
if (Test-Path xatlas) { Remove-Item xatlas -Recurse -Force }; `
git clone --recursive https://github.com/jpcy/xatlas; `
(Get-Content .\xatlas\source\xatlas\xatlas.cpp) -replace '#if 0', '//#if 0' -replace '#endif', '//#endif' | Set-Content .\xatlas\source\xatlas\xatlas.cpp; `
cd ..\..\..; `
python_embeded\python.exe -m pip install .\xatlas-python\
```

---

## 📂 Directory Overview

```
ComfyUI/
├── custom_nodes/
│   └── ComfyUI-Hunyuan3DWrapper/
│       ├── hy3dpaint/
│       │   ├── custom_rasterizer/         # Custom rasterizer module
│       │   │   ├── setup.py
│       │   │   └── dist/
│       │   ├── DifferentiableRenderer/    # Differentiable renderer
│       │   │   ├── setup.py
│       │   │   └── dist/
│       ├── wheels/                        # Precompiled wheel files
├── models/
│   ├── diffusion_models/
│   │   └── [hunyuan3d-dit-v2-1.ckpt](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1)
│   └── vae/
│       └── [hunyuan3d-vae-v2-1.ckpt](https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-vae-v2-1)
├── xatlas-python/                         # Patched UV unwrapper (optional)
│   └── extern/
│       └── xatlas/
```

---

## 🙏 Acknowledgements

* **[kijai](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)** — Original wrapper developer
* TrueMike, Agee, Palindar, and the vibrant Discord community
* Tencent team for the incredible [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) model
