# 4CGT — 四彫画匠 (Shichō Gashō)

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?style=for-the-badge&logo=github)](https://github.com/sponsors/teenu)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge)](LICENSE)

**Master Artisan of Sculpted Art** — quality-engineered anime image generation with cross-platform determinism, DoRA anatomical accuracy, and integrated 2D→3D conversion. Built on NoobAI XL V-Pred 1.0.

## ⚠️ Critical: Supported Model Formats

**Supported Models**: This implementation supports two formats for lossless quality:
1. **`NoobAI-XL-Vpred-v1.0.safetensors`** (BF16 single file, 7GB)
2. **`NoobAI-XL-Vpred-v1.0-FP32/`** (FP32 directory format, 13GB)

**NOT Supported**: FP16 models are rejected due to lossy quantization.

**Why These Formats?**
- FP16 models were created via **lossy quantization** from BF16
- Even with FP32 upcast, FP16 model weights are **already degraded**
- BF16 and FP32 formats ensure developer-intended quality on ALL platforms

**Automatic Precision Handling**:
- **BF16 model** → Auto-upcast to FP32 on platforms without BF16 support
- **FP32 directory** → Used directly (pre-converted for determinism)
- **VAE**: Always runs in FP32 for lossless image decode
- **Result**: Identical quality and output hashes across all platforms

**Cross-Platform Determinism**: Same seed = **identical image hash** on macOS, Windows, and Linux.

## Features

- **Lossless Quality Pipeline**: FP32 VAE + intelligent precision (BF16 native or FP32 upcast)
- **Cross-Platform Parity**: Identical image hashes for same seed across all platforms
- **Deterministic Generation**: Reproducible outputs with enforced deterministic algorithms
- **CPU Offloading**: Automatic sequential CPU offloading for GPUs with <8GB VRAM
- **DoRA Adapters**: Support for Weight-Decomposed Low-Rank Adaptation stabilizers
- **Interactive GUI**: Gradio-based web interface with character/artist search
- **CLI Mode**: Command-line interface for batch processing

## System Requirements

### Minimum
- Python 3.11+
- 8GB System RAM
- GPU: NVIDIA GTX 1060 6GB or equivalent (CUDA support)
- ~15GB free disk space (7GB model + dependencies)

### Recommended
- Python 3.11 or 3.12
- 16GB System RAM
- GPU: NVIDIA RTX 3060 12GB or better
- CUDA 12.4+ (CUDA 12.8 for RTX 5090)

### GPU Compatibility
- **RTX 50xx series (Blackwell)**: Native BF16 support, requires CUDA 12.8+ and PyTorch 2.9+
- **RTX 30xx/40xx series (Ampere/Ada)**: Native BF16 support (optimal)
- **RTX 20xx series (Turing)**: BF16 model upcast to FP32 (slower but lossless)
- **Apple Silicon (M1/M2/M3/M4)**: Native BF16 via AMX (optimal)
- **<8GB VRAM**: Automatic CPU offloading enabled
- **≥8GB VRAM**: Full GPU loading

## Installation

### 1. Install PyTorch

Choose the appropriate PyTorch version for your GPU:

**RTX 5090 (Blackwell) - CUDA 12.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**RTX 30xx/40xx (Ampere/Ada) - CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**RTX 20xx (Turing) - CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `diffusers` - Stable Diffusion pipeline
- `transformers` - Model components
- `accelerate` - CPU offloading for low-VRAM GPUs
- `peft` - DoRA adapter support
- `gradio` - Web interface
- `pandas` - CSV data processing for character/artist search
- `safetensors` - Model format optimization

**Windows Users - Important:**

If you encounter "path too long" errors during installation:

1. **Enable Windows Long Paths (Run PowerShell as Administrator):**
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

2. **Restart your computer** (required)

3. **Then retry:** `pip install -r requirements.txt`

Alternatively, use a shorter installation path like `C:\noobai\` instead of deep nested folders.

### 3. Download Assets

All assets (model, DoRA adapters, ControlNet, style data) are hosted on Hugging Face:

```bash
pip install huggingface-hub
huggingface-cli download epigene/4cgt --local-dir assets
```

Then move them into place:

```bash
mv assets/NoobAI-XL-Vpred-v1.0.safetensors .
mv assets/dora/* dora/
mv assets/controlnet/* controlnet/
mv assets/style/* style/
mv assets/poses/* poses/
rm -rf assets
```

| Asset | Size | Required? |
|-------|------|-----------|
| `NoobAI-XL-Vpred-v1.0.safetensors` | 6.7 GB | Yes |
| `controlnet/openpose_fp32.safetensors` | 4.7 GB | Optional (pose control) |
| `dora/*.safetensors` | 108 MB | Optional (anatomical accuracy) |
| `style/*.csv` | 109 MB | Optional (character/artist search) |
| `poses/*.png` | 140 KB | Optional (example poses) |

**Model format notes:**
- BF16 model auto-upcasts to FP32 on platforms without BF16 support
- FP16 models are **rejected** (lossy quantization)
- The app auto-detects the model in the repo root, `./models/`, `~/Downloads/`, or `~/Models/`

## Usage

### GUI Mode (Default)

```bash
python main.py
```

The web interface will open automatically at `http://localhost:7860`

**GUI Features:**
- 🎨 **Positive Prompt Formatter**: Character/artist search with autocomplete
- 🎯 **DoRA Adapter Selection**: Dropdown menu for stabilizer adapters
- ⚙️ **Advanced Settings**: CFG scale, steps, resolution, seed control
- 📊 **Real-time Generation**: Progress tracking with ETA

### LAN Access Mode (Network Access)

Enable LAN access to use the GUI from any device on your local network:

```bash
python main.py --lan
```

**What happens:**
- Server binds to `0.0.0.0` (all network interfaces)
- Browser doesn't open automatically
- Gradio displays both local and network URLs

**Example output:**
```
🌐 LAN Access Mode: Enabled
   Interface will be accessible from any device on your local network
   Server will bind to: 0.0.0.0:7860

Running on local URL:  http://127.0.0.1:7860
Running on public URL: http://192.168.1.100:7860
```

**Use cases:**
- **Windows PC → iPhone/iPad**: Run on Windows, access from mobile Safari
- **Linux Server → Multiple Clients**: Headless server, access from any device
- **Mac → Android Tablet**: Run on Mac, access from tablet browser

**Custom port:**
```bash
python main.py --lan --port 8080
```

**Security Note:** LAN mode is only accessible on your local network. For internet access, use `--share` to create a temporary public Gradio link.

### CLI Mode

**Basic generation:**
```bash
python main.py --cli --prompt "1girl, cat ears" --steps 35
```

**With DoRA adapter:**
```bash
python main.py --cli --prompt "1girl" --enable-dora --dora-adapter 0 --steps 35
```

**Custom model path:**
```bash
python main.py --cli --model-path "/path/to/model.safetensors" --prompt "1girl"
```

**List available DoRA adapters:**
```bash
python main.py --list-dora-adapters
```

**Full CLI options:**
```bash
python main.py --cli --help
```

## Directory Structure

```
noobai-xl-modular/
├── main.py                 # Entry point (GUI/CLI)
├── cli.py                  # CLI argument parsing
├── config.py               # Configuration constants
├── state.py                # State management
├── prompt_formatter.py     # Character/artist search
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── engine/                 # Core inference engine
│   ├── __init__.py
│   ├── core.py             # Main engine class
│   ├── model_loader.py     # Model loading and precision
│   ├── dora_manager.py     # DoRA adapter management
│   ├── memory.py           # GPU memory utilities
│   ├── progress.py         # Progress tracking
│   └── prompt/             # Long prompt support
│       ├── tokenizer.py
│       └── embedding.py
├── ui/                     # Gradio interface
│   ├── __init__.py
│   ├── interface.py        # Main UI components
│   ├── engine_manager.py   # Engine lifecycle
│   ├── generation.py       # Image generation handlers
│   ├── widgets.py          # UI widget helpers
│   └── styles.py           # CSS and JavaScript
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── validation.py       # Path and parameter validation
│   ├── dora.py             # DoRA discovery
│   └── schedules.py        # DoRA schedule parsing
├── dora/                   # DoRA adapter files (*.safetensors)
│   └── .gitkeep
├── style/                  # Character/artist CSV databases
│   └── .gitkeep
└── outputs/                # Generated images
    └── .gitkeep
```

## Optimal Settings

**For highest quality:**
- Resolution: 832x1216 or 1216x832
- Steps: 35
- CFG Scale: 4.5
- Rescale CFG: 0.7
- DoRA: Enabled (v0.280a recommended)

**For faster generation (6GB VRAM GPUs):**
- Resolution: 768x1024 or 1024x768
- Steps: 25
- CFG Scale: 4.5
- CPU Offloading: Automatic

## Model Precision Guide

### Supported Model Formats

**BF16 Single File (`NoobAI-XL-Vpred-v1.0.safetensors` - 7GB):**
- ✅ Developer's canonical version
- ✅ Auto-upcast to FP32 on platforms without BF16 support
- ✅ Cross-platform parity guaranteed
- ✅ Smaller download size

**FP32 Directory (`NoobAI-XL-Vpred-v1.0-FP32/` - 13GB):**
- ✅ Pre-converted for maximum determinism
- ✅ Direct FP32 weights (no upcast needed)
- ✅ Validated precision on load
- ✅ Cross-platform parity guaranteed

**FP16 Models - NOT Supported:**
- ❌ Created via **lossy quantization** from BF16
- ❌ Model weights **already degraded** during FP16 conversion
- ❌ Cannot recover lost precision (even with FP32 upcast)
- ❌ Application will **refuse to load** FP16 models

### Platform Behavior

**Apple Silicon (M1/M2/M3/M4):**
- Native BF16 via AMX instructions
- Optimal performance + quality

**RTX 50xx (Blackwell):**
- Native BF16 via tensor cores
- Requires CUDA 12.8+ and PyTorch 2.9+
- Optimal performance + quality

**RTX 30xx/40xx (Ampere/Ada):**
- Native BF16 via tensor cores
- Optimal performance + quality

**RTX 20xx (Turing):**
- BF16 → FP32 lossless upcast
- Slower but **identical quality** to other platforms

**Older GPUs:**
- BF16 → FP32 lossless upcast
- Slower but **identical quality** to other platforms

### Precision Pipeline

**Lossless Quality Enforcement:**
1. Model loaded (BF16 single file or FP32 directory)
2. FP16 models rejected with error
3. BF16 → Auto-upcast to FP32 on platforms without BF16 support
4. FP32 directory → Precision validated on load
5. VAE always runs in FP32 (lossless decode)
6. Result: Identical quality + identical hashes across platforms

**Why FP32 (not FP16)?**
- BF16 → FP32 = **lossless** (mathematical subset)
- BF16 → FP16 = **lossy** (incompatible formats)
- FP32 guarantees numerical parity across all platforms

### Cross-Platform Reproducibility

✅ **Same seed → same image hash** (macOS/Windows/Linux)
✅ **Deterministic algorithms enforced**
✅ **CPU generator for consistent RNG**
✅ **Lossless precision handling** (BF16/FP32 only)
✅ **FP32 VAE** (no decode variance)

## Troubleshooting

### ModuleNotFoundError: No module named 'transformers.utils' (Windows)

**Symptom:** Application crashes immediately with transformers import error

**Cause:** Corrupted transformers installation due to Windows pip cache issues

**Fix:**
```bash
pip uninstall transformers -y
pip install transformers>=4.40.0
```

**Alternative fix (clears all pip cache):**
```bash
pip cache purge
pip install -r requirements.txt
```

This is a known Windows-specific issue where pip's cache can become corrupted during installation, causing the transformers package to install incompletely. The fix works without requiring Windows Long Paths to be enabled.

### "No model found" error
- Ensure model file is in the repository root or `models/` directory
- Check filename matches: `NoobAI-XL-Vpred-v1.0*.safetensors`
- Use `--model-path` flag to specify custom location

### CUDA out of memory
- Model automatically enables CPU offloading for GPUs <8GB VRAM
- Reduce resolution (try 512x768 or 768x1024)
- Reduce inference steps (try 25 instead of 35)
- Close other GPU applications
- Consider using BF16 single file (7GB) instead of FP32 directory (13GB)

### DoRA adapters not loading
- Verify `peft` is installed: `pip install peft>=0.18.0`
- Check adapter files are in `dora/` directory
- Ensure adapters are `.safetensors` format

### Character/artist search not working
- Verify CSV files are in `style/` directory
- Check `pandas` is installed: `pip install pandas>=2.0.0`
- Ensure CSV files are not corrupted

### Slow generation on RTX 20xx GPU
- BF16 model is auto-upcast to FP32 (slower but lossless)
- CPU offloading is enabled automatically for <8GB VRAM
- Reduce resolution or steps for faster generation

## Performance Benchmarks

**RTX 2060 6GB (with CPU offloading):**
- Resolution: 512x512, Steps: 5 → ~13 seconds
- Resolution: 832x1216, Steps: 35 → ~90 seconds
- VRAM usage: ~400MB (offloading enabled)

**RTX 3060 12GB (full GPU):**
- Resolution: 832x1216, Steps: 35 → ~30 seconds
- VRAM usage: ~6.7GB

## Credits

- **Model**: NoobAI XL V-Pred 1.0
- **Framework**: Diffusers (Hugging Face)
- **UI**: Gradio
- **Optimization**: Lossless BF16/FP32 precision, CPU offloading, DoRA adapters
- **Long Prompts**: sd_embed for >77 token support

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).
