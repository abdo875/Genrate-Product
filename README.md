# GEnrate Product

Tkinter desktop app that runs a local Stable Diffusion XL pipeline (via `diffusers`) to render front, side, and back views of apparel items. Each render can be zoomed, previewed, and downloaded locally.

## Features

- Prompt builder inputs (product name, design notes, extra instructions)
- Generates coordinated prompts for all three views
- Calls a Hugging Face-hosted image model (configurable via `HF_MODEL_ID`)
- Inline previews with zoom controls and per-view save buttons
- Bulk download to a chosen directory

## Quick start

```powershell
cd "Generate-Product"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
@'
HF_TOKEN=hf_xxx_your_token
HF_MODEL_ID=black-forest-labs/FLUX.1-dev
'@ | Set-Content .env
python main.py
```

The app now calls Hugging Face's router endpoint (`https://router.huggingface.co/hf-inference/models/<model>`). Set `HF_MODEL_ID` to any router-compatible model you have access to (e.g. `black-forest-labs/FLUX.1-schnell`). The previous local SDXL pipeline is preserved in comments inside `model_api_client.py` if you need to revert.

## Using the GUI

1. Launch `python main.py`.
2. Fill in *Product name* and optional *Design details* / *Extra instructions*.
3. Click **Generate Images**. Progress appears in the status bar while the three local renders (front/side/back) run sequentially using fixed settings (1024×1024, 6 steps, 0.0 guidance).
4. When images appear, either use each card’s **Save…** button, the **Zoom** viewer, or **Download All…** to export every render to a folder—the saved PNGs include embedded prompt metadata.

## Configuration

Environment variable | Description | Default
---|---|---
`HF_TOKEN` | Hugging Face personal access token with Inference API access | **required**
`HF_MODEL_ID` | Hosted Hugging Face model to call via router | `black-forest-labs/FLUX.1-dev`

> ⚠️ Hosted inference incurs rate limits. Use `HF_MODEL_ID` to point at another model if you have access (e.g. `black-forest-labs/FLUX.1-schnell`). The previous local SDXL pipeline implementation is commented inside `model_api_client.py` for reference.

Runtime controls allow adjusting output size, inference steps, and guidance scale without modifying code.

## Repository contents

File | Purpose
---|---
`main.py` | Tkinter GUI entrypoint
`model_api_client.py` | Wrapper around the local `StableDiffusionXLPipeline`
`prompt_builder.py` | Generates harmonized prompts per view
`task-upwork.ipynb` | Original experimentation notebook

