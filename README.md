# 4CGT — 四彫画匠

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?style=for-the-badge&logo=github)](https://github.com/sponsors/teenu)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Assets-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/epigene/4cgt)

**Shichō Gashō** — the art of sculpted mastery.

Anime image generation that nails anatomy, follows your prompt, and gives you the same pixel-perfect result on any machine. Every time.

<p align="center">
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/frieren_beach.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/lucy_cyberpunk.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/miku.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/makima.png" width="24%" />
</p>
<p align="center">
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/asuka.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/2b_ruins.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/zero_two.png" width="24%" />
<img src="https://huggingface.co/epigene/4cgt/resolve/main/showcase/yor.png" width="24%" />
</p>
<p align="center"><sub>All generated with 4CGT. DoRA v0.271. 42 steps. Same seed = same image on any machine.</sub></p>

---

SDXL v-prediction frontend built around [NoobAI XL V-Pred 1.0](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0). DoRA weight-decomposed adapters for anatomical accuracy. BF16/FP32 lossless pipeline — FP16 is rejected on load. Deterministic output: same seed produces the same image hash across macOS, Windows, and Linux. Long prompt support up to ~600 tokens with A1111 weight syntax. Integrated 2D-to-3D Gaussian Splat conversion via Apple Sharp. OpenPose ControlNet tuned for v-prediction. Runs locally on GTX 1060 through RTX 5090 and Apple Silicon.

## Quick start

```bash
git clone https://github.com/teenu/4cgt.git && cd 4cgt

# Install PyTorch (pick your GPU — see full list below)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Download assets (see table below for sources)
# Then place them: model in repo root, adapters in dora/, CSVs in style/

# Launch
python main.py
```

Opens a web UI at `http://localhost:7860`. That's it.

### CLI mode

```bash
# DoRA None mode auto-applies optimal settings (42 steps, CFG 5.53, rescale 0.61)
python main.py --cli \
  --prompt "very awa, masterpiece, best quality, 1girl, silver hair, red eyes, black dress, night sky, cinematic lighting" \
  --enable-dora --dora-adapter 0 --seed 7777
```

## Assets

4CGT requires third-party model assets. Download from their original sources:

| Asset | Source | Required |
|-------|--------|----------|
| NoobAI XL V-Pred 1.0 (BF16, 6.7 GB) | [Laxhar/noobai-XL-Vpred-1.0](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0) | Yes — place in repo root |
| DoRA stabilizers (v0.271 recommended) | [Civitai: reakaakasky](https://civitai.com/models/971952) | Yes — place in `dora/` |
| Style CSVs (Danbooru + e621) | [Laxhar/noob-wiki](https://huggingface.co/datasets/Laxhar/noob-wiki) | Yes — place in `style/` |
| ControlNet OpenPose SDXL | [xinsir/controlnet-openpose-sdxl-1.0](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0) | Optional — place in `controlnet/` |

These assets carry their own licenses. See each source for terms.

## How it works

### DoRA anatomical stabilizers

Weight-Decomposed Low-Rank Adaptation adapters trained for v-prediction. Three modes:

- **None mode** — optimal defaults (42 steps, CFG 5.53, rescale 0.61). Best starting point.
- **Optimized** — preset-locked 34-step schedule. Parameters are frozen.
- **Manual** — per-step binary toggle grid. Full control over when DoRA activates.

### Lossless precision pipeline

FP16 is rejected on load. Not downcast, not warned — rejected.

The pipeline is BF16 or FP32 only. VAE always decodes in FP32. The result: no precision-dependent artifacts, no platform-dependent drift. The image you generate on your Mac is byte-identical to the one on your Linux workstation.

### Long prompts

Standard CLIP cuts off at 77 tokens. 4CGT extends this to ~600 tokens via `sd_embed`, with full A1111-compatible syntax: `(emphasis:1.2)`, `[de-emphasis]`, `((nested weights))`.

### Image to 3D

Integrated Apple Sharp pipeline. Generate an image, convert it to a 3D Gaussian Splat (.ply), optionally render a camera flythrough video — all without leaving the UI. Sharp runs in a subprocess so it doesn't evict SDXL from VRAM.

### ControlNet pose control

OpenPose skeleton conditioning optimized for v-prediction. Default conditioning scale is 2.0 (vs. 1.0 in standard pipelines) because v-pred models need stronger guidance.

## GPU compatibility

| GPU | BF16 | Notes |
|-----|------|-------|
| RTX 5090 (Blackwell) | Native | CUDA 12.8+, PyTorch 2.9+ |
| RTX 30xx/40xx (Ampere/Ada) | Native | Optimal |
| RTX 20xx (Turing) | Upcast to FP32 | Slower, identical quality |
| Apple Silicon (M1-M4) | Native (AMX) | Optimal |
| GTX 1060+ (6GB) | Upcast to FP32 | Auto CPU offloading |

### PyTorch install by GPU

```bash
# RTX 5090
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# RTX 30xx/40xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# RTX 20xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Apple Silicon
pip install torch torchvision torchaudio
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Benchmarks

| GPU | Resolution | Steps | Time |
|-----|-----------|-------|------|
| RTX 5090 32GB | 1216x832 | 42 | ~25s |
| RTX 3060 12GB | 1216x832 | 35 | ~30s |
| RTX 2060 6GB | 1216x832 | 35 | ~90s (CPU offload) |

## Troubleshooting

**Windows `transformers` import error** — corrupted pip cache. Fix: `pip uninstall transformers -y && pip install transformers>=4.40.0`

**No model found** — place `NoobAI-XL-Vpred-v1.0.safetensors` in the repo root or use `--model-path`

**CUDA out of memory** — reduce resolution to 768x1024, reduce steps to 25, or close other GPU apps. CPU offloading kicks in automatically under 8GB VRAM.

**Windows path too long** — enable long paths: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` then restart.

## License

[AGPL-3.0](LICENSE). Free to use, modify, and distribute. If you run a modified version as a network service, share your source.
