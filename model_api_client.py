'''
"""
Local Stable Diffusion pipeline wrapper used by the GUI.
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


class ModelLoadError(RuntimeError):
    """Raised when the local Stable Diffusion pipeline cannot be initialized."""


@dataclass
class ModelAPIConfig:
    model_id: str = "sdxl_turbo_model"
    inference_steps: int = 6
    guidance_scale: float = 0.0
    image_size: int = 1024
    negative_prompt: str = (
        "duplicate, two garments, low quality, blurry, distorted, watermark, text, cropped, human, person, model, body"
    )


class ModelAPIClient:
    def __init__(self, *, token: Optional[str] = None, config: Optional[ModelAPIConfig] = None):
        self._config = config or ModelAPIConfig()
        env_override = os.getenv("HF_MODEL_ID")
        self._model_source = env_override or self._config.model_id
        self._token = token or os.getenv("HF_TOKEN")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        model_kwargs = {
            "torch_dtype": self._dtype,
            "use_safetensors": True,
        }

        if self._dtype == torch.float16:
            model_kwargs["variant"] = "fp16"

        model_path = Path(self._model_source)
        is_local_folder = model_path.exists()

        if self._token and not is_local_folder:
            model_kwargs["use_auth_token"] = self._token
        if is_local_folder:
            model_kwargs["local_files_only"] = True

        try:
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self._model_source,
                **model_kwargs,
            )
            self._pipe.to(self._device)
            self._pipe.set_progress_bar_config(disable=True)
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        except Exception as exc:  # pragma: no cover - initialization issues
            hint = (
                f"Ensure the folder '{self._model_source}' exists with the SDXL files."
                if is_local_folder
                else f"Ensure '{self._model_source}' is a valid Hugging Face repo id."
            )
            raise ModelLoadError(
                f"Failed to load '{self._model_source}'. {hint} Details: {exc}"
            ) from exc

    def generate_image(
        self,
        prompt: str,
        *,
        image_size: Optional[int] = None,
        inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        size = image_size or self._config.image_size
        steps = inference_steps or self._config.inference_steps
        guidance = guidance_scale if guidance_scale is not None else self._config.guidance_scale
        neg_prompt = negative_prompt or self._config.negative_prompt

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._dtype) if self._device == "cuda" else nullcontext()
        )

        with torch.inference_mode():
            with autocast_ctx:
                result = self._pipe(
                    prompt,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    negative_prompt=neg_prompt,
                    generator=generator,
                )

        return result.images[0]
'''

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image


class MissingAPITokenError(RuntimeError):
    """Raised when the Hugging Face token is not provided in the environment."""


class ModelAPIError(RuntimeError):
    """Raised when the Inference API request fails."""


@dataclass
class ModelAPIConfig:
    model_id: str = "black-forest-labs/FLUX.1-dev"
    width: int = 1024
    height: int = 1024
    inference_steps: int = 30
    guidance_scale: float = 3.5
    timeout: int = 300
    max_history: int = 6


class ModelAPIClient:
    def __init__(self, *, token: Optional[str] = None, config: Optional[ModelAPIConfig] = None):
        self._config = config or ModelAPIConfig()
        self._token = token or os.getenv("HF_TOKEN")
        if not self._token:
            raise MissingAPITokenError(
                "HF_TOKEN is missing. Provide it via environment variable or a .env file."
            )

        env_model = os.getenv("HF_MODEL_ID")
        self._model_id = env_model or self._config.model_id
        self._api_url = f"https://router.huggingface.co/hf-inference/models/{self._model_id}"
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }
        self._history: List[str] = []

    # Legacy helpers from notebook logic ---------------------------------
    def _make_seed_from_text(self, text: str) -> int:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return int(digest[:16], 16) % (2**31 - 1)

    def _build_prompt(self, user_input: str, product_type: str, extra_notes: str) -> str:
        history_tail = self._history[-5:]
        history_block = ""
        if history_tail:
            history_block = "previous user inputs: " + " | ".join(history_tail) + ". "

        base_prompt = (
            f"{history_block}"
            f"product description: {user_input}. "
            f"product type: {product_type}. "
            "You are generating a professional PRODUCT MOCKUP for e-commerce. "
            "The system must support ANY product category: clothing (t-shirt, hoodie, jacket, pants), "
            "accessories (glasses, sunglasses, bags), objects (cup, mug, bottle, container), "
            "or any other physical item. "
            "The output must be a SINGLE clean product image with NO text and NO humans. "
            "NO models, NO body parts, NO mannequins, NO symbols, NO warning icons, NO triangles, NO signs, "
            "NO props, and NOTHING unrelated to the product. "
            "Show ONLY the product itself. "
            "Ensure the colors, materials, textures, stitching, reflections, and overall style "
            "remain IDENTICAL across all required views. "
            "Style: ultra-realistic product mockup, high-resolution detailed surfaces, accurate shadows, "
            "realistic material physics. "
            "Background: solid white or light gray studio backdrop, clean, no gradients, no shadows cutoff. "
            "Lighting: soft diffused studio lighting, no shine artifacts, no reflections from the environment. "
            "Camera: orthographic controlled product photography angle. "
            f"Required view: {extra_notes}. "
            "The product must appear centered, isolated, correctly scaled, floating or standing naturally. "
            "Maintain perfect consistency between front, back, and side views of the same exact item."
            "If the request is for a side view, render a pure 90-degree side profile with zero front-facing details."
        )
        return base_prompt

    def _call_model(self, prompt: str, seed: int) -> Image.Image:
        payload = {
            "inputs": prompt,
            "parameters": {
                "seed": seed,
                "width": self._config.width,
                "height": self._config.height,
                "num_inference_steps": self._config.inference_steps,
                "guidance_scale": self._config.guidance_scale,
            },
            "options": {"wait_for_model": True},
        }

        response = requests.post(
            self._api_url,
            headers=self._headers,
            json=payload,
            timeout=self._config.timeout,
        )
        if response.status_code != 200:
            raise ModelAPIError(
                f"Inference API failed ({response.status_code}): {response.text[:500]}"
            )

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type.lower():
            data = response.json()
            images = data.get("images")
            if not images:
                raise ModelAPIError(f"Inference API response missing 'images': {data}")
            image_base64 = images[0]
            image_bytes = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_bytes)).convert("RGB")

        return Image.open(BytesIO(response.content)).convert("RGB")

    # Public interface ---------------------------------------------------
    def start_session(self, user_input: str) -> Tuple[int, str]:
        self._history.append(user_input)
        if len(self._history) > self._config.max_history:
            self._history = self._history[-self._config.max_history :]

        product_id = f"product_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        seed_source = " ".join(self._history[-self._config.max_history :]) + product_id
        seed = self._make_seed_from_text(seed_source)
        return seed, product_id

    def generate_view(
        self,
        user_input: str,
        product_type: str,
        *,
        extra_notes: str,
        seed: int,
    ) -> Tuple[Image.Image, str]:
        prompt = self._build_prompt(user_input, product_type, extra_notes)
        image = self._call_model(prompt, seed)
        return image, prompt
