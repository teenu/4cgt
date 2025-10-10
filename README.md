# NoobAI XL V-Pred 1.0 - Modularized Edition

This repository contains both the original monolithic script and the modularized version of the NoobAI XL V-Pred 1.0 image generation tool.

## 📁 Repository Structure

### Modularized Version (9 modules)
- **`main.py`** - Entry point orchestrating all modules
- **`config.py`** - Configuration constants and exceptions
- **`state.py`** - Performance monitoring and state management
- **`utils.py`** - Utility functions (validation, file operations, DoRA discovery)
- **`engine.py`** - NoobAIEngine class (model loading, generation, DoRA management)
- **`prompt_formatter.py`** - CSV data loading and indexed search
- **`ui_helpers.py`** - UI helper functions and generation handlers
- **`ui.py`** - Gradio interface creation
- **`cli.py`** - CLI functions and argument parsing

### Original Version
- **`noobai_cli_final.py`** - Original monolithic script (2,853 lines)

## ✅ Output Parity Verification

The modularized version produces **byte-for-byte identical output** to the original:

| Metric | Modularized | Original | Match |
|--------|-------------|----------|-------|
| MD5 Hash | `a7995771724acf7ec63b7c04ae7afea3` | `a7995771724acf7ec63b7c04ae7afea3` | ✅ |
| SHA256 Hash | `f04361124b41831c566b9157fa5f45a3...` | `f04361124b41831c566b9157fa5f45a3...` | ✅ |
| File Size | 973,989 bytes | 973,989 bytes | ✅ |

## 🚀 Usage

### GUI Mode (Default)
```bash
python main.py
```

### CLI Mode
```bash
python main.py --cli --prompt "anime girl, detailed" --steps 35
```

### List DoRA Adapters
```bash
python main.py --list-dora-adapters
```

## 📝 Key Features

- **Hash Consistency**: Standardized PNG saving ensures reproducible results
- **DoRA Support**: Weight-Decomposed Low-Rank Adaptation for enhanced generation
- **Smart UI**: Auto-detection of adapters with smart defaults
- **Performance Monitoring**: Optional profiling for debugging
- **Thread-safe State**: Robust state management for generation control
- **CSV Search**: Indexed prompt formatter with Danbooru/E621 data

## 🔧 Requirements

- Python 3.8+
- PyTorch
- diffusers
- Gradio
- PIL/Pillow
- safetensors
- pandas (optional, for CSV loading)

## 📊 Module Statistics

| Module | Lines | Purpose |
|--------|-------|---------|
| config.py | ~190 | Configuration & constants |
| state.py | ~160 | State management |
| utils.py | ~310 | Utility functions |
| engine.py | ~550 | Core engine logic |
| prompt_formatter.py | ~235 | CSV data & search |
| ui_helpers.py | ~480 | UI helper functions |
| ui.py | ~570 | Gradio interface |
| cli.py | ~270 | CLI functionality |
| main.py | ~95 | Entry point |
| **Total** | **~2,860** | **9 focused modules** |

## 🎯 Design Principles

1. **Lossless Refactoring**: Zero functional changes, perfect output parity
2. **Logical Boundaries**: Each module has a clear, single responsibility
3. **Minimal Changes**: Only necessary imports added to each module
4. **Preserved State**: Global instances (engine, prompt_formatter_data) remain accessible
5. **Execution Flow**: Identical behavior in both CLI and GUI modes

## 📄 License

Same as original NoobAI XL V-Pred 1.0 license.
