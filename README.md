# NoobAI XL V-Pred 1.0 - Modular Implementation

Clean, modular implementation of NoobAI XL V-Pred 1.0 with FP16 optimization, DoRA adapter support, and interactive prompt formatting.

## Features

- **FP16 Optimization**: Automatic precision detection with FP16 support for RTX 20xx series GPUs
- **CPU Offloading**: Automatic sequential CPU offloading for GPUs with <8GB VRAM
- **DoRA Adapters**: Support for Weight-Decomposed Low-Rank Adaptation stabilizers
- **Interactive GUI**: Gradio-based web interface with character/artist search
- **CLI Mode**: Command-line interface for batch processing
- **Hash Consistency**: Deterministic image generation with seed-based hashing

## System Requirements

### Minimum
- Python 3.8+
- 8GB System RAM
- GPU: NVIDIA GTX 1060 6GB or equivalent (CUDA support)
- ~15GB free disk space (7GB model + dependencies)

### Recommended
- Python 3.10+
- 16GB System RAM
- GPU: NVIDIA RTX 2060 6GB or better
- CUDA 11.8 or newer

### GPU Compatibility
- **RTX 20xx series (Turing)**: FP16 optimized (use `-fp16-all.safetensors` model)
- **RTX 30xx/40xx series (Ampere/Ada)**: FP16 or BF16 supported
- **<8GB VRAM**: Automatic CPU offloading enabled
- **≥8GB VRAM**: Full GPU loading

## Installation

### 1. Install PyTorch

Choose the appropriate PyTorch version for your platform:

**CUDA (NVIDIA GPU - Recommended):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Apple Silicon (MPS):**
```bash
pip install torch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `diffusers` - Stable Diffusion pipeline
- `accelerate` - CPU offloading for low-VRAM GPUs
- `peft` - DoRA adapter support
- `gradio` - Web interface
- `pandas` - CSV data processing for character/artist search
- `safetensors` - Model format optimization

### 3. Download Model File

Download one of the following model files and place in the repository root directory:

**Recommended (FP16 Optimized for RTX 20xx/30xx):**
- `NoobAI-XL-Vpred-v1.0-fp16-all.safetensors` (6.5GB)

**Alternative (Original BF16 - requires RTX 30xx+):**
- `NoobAI-XL-Vpred-v1.0.safetensors` (7.0GB)

The application will automatically detect the model file in:
- Repository root directory
- `./models/` subdirectory
- `~/Downloads/`
- `~/Models/`

### 4. Download Style Data (Required for GUI)

The `style/` directory contains character and artist databases for autocomplete search:

**Download the following CSV files and place in `style/` directory:**
- `danbooru_character_webui.csv`
- `danbooru_artist_webui.csv`
- `e621_character_webui.csv`
- `e621_artist_webui.csv`

**Total size:** ~110MB

**Without style data:** Character/artist search features will be unavailable in the GUI.

### 5. Download DoRA Adapters (Optional)

DoRA (Weight-Decomposed Low-Rank Adaptation) adapters provide image stabilization:

**Download and place in `dora/` directory:**
- `noobai_vp10_stabilizer_v0.271_fp16.safetensors` (44MB)
- `noobai_vp10_stabilizer_v0.280a_fp16.safetensors` (64MB)

**Without DoRA adapters:** DoRA functionality will be unavailable but the application will run normally.

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
├── engine.py              # Core inference engine
├── ui.py                  # Gradio interface
├── cli.py                 # CLI functionality
├── config.py              # Configuration constants
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── dora/                  # DoRA adapter files (*.safetensors)
│   └── .gitkeep
├── style/                 # Character/artist CSV databases
│   └── .gitkeep
└── outputs/               # Generated images
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

## Troubleshooting

### "No model found" error
- Ensure model file is in the repository root or `models/` directory
- Check filename matches: `NoobAI-XL-Vpred-v1.0*.safetensors`
- Use `--model-path` flag to specify custom location

### CUDA out of memory
- Model automatically enables CPU offloading for GPUs <8GB VRAM
- Reduce resolution (try 512x768)
- Close other GPU applications
- Use FP16 model variant instead of BF16

### DoRA adapters not loading
- Verify `peft` is installed: `pip install peft>=0.18.0`
- Check adapter files are in `dora/` directory
- Ensure adapters are `.safetensors` format

### Character/artist search not working
- Verify CSV files are in `style/` directory
- Check `pandas` is installed: `pip install pandas>=2.0.0`
- Ensure CSV files are not corrupted

### Slow generation on RTX 20xx GPU
- Use FP16 model (`-fp16-all.safetensors`)
- Engine automatically detects Turing architecture and uses FP16
- CPU offloading is enabled automatically for optimal performance

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
- **Optimization**: FP16 precision, CPU offloading, DoRA adapters

## License

See repository for license information.
