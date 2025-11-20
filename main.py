from __future__ import annotations

import threading
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

import logging

from dotenv import load_dotenv
from PIL import Image, ImageTk, PngImagePlugin

from model_api_client import (
    MissingAPITokenError,
    ModelAPIClient,
    ModelAPIConfig,
    ModelAPIError,
)


load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def save_image_with_metadata(image: Image.Image, filepath: Path | str, metadata: Optional[Dict[str, Any]] = None):
    path_str = str(filepath)
    if path_str.lower().endswith(".png") and metadata:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            if value is None:
                continue
            pnginfo.add_text(str(key), str(value))
        image.save(path_str, pnginfo=pnginfo)
    else:
        image.save(path_str)


class ImageSlot(ttk.LabelFrame):
    def __init__(self, master: tk.Widget, view_name: str):
        super().__init__(master, text=view_name.title())
        self.view_name = view_name
        self._pil_image: Optional[Image.Image] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._metadata: Dict[str, Any] = {}

        self.image_label = ttk.Label(self, text="No image yet", width=40, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=8, pady=8)

        button_row = ttk.Frame(self)
        button_row.pack(fill="x", padx=8, pady=(0, 8))

        self.save_button = ttk.Button(button_row, text="Save…", command=self._save, state="disabled")
        self.save_button.pack(side="left")

        self.zoom_button = ttk.Button(button_row, text="Zoom", command=self._open_zoom, state="disabled")
        self.zoom_button.pack(side="left", padx=6)

    def set_placeholder(self, text: str):
        self._pil_image = None
        self._photo_image = None
        self._metadata = {}
        self.image_label.configure(image="", text=text)
        self.save_button.configure(state="disabled")
        self.zoom_button.configure(state="disabled")

    def set_image(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None):
        self._pil_image = image
        self._metadata = metadata or {}
        preview = image.copy()
        preview.thumbnail((380, 380), Image.Resampling.LANCZOS)
        self._photo_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self._photo_image, text="")
        self.save_button.configure(state="normal")
        self.zoom_button.configure(state="normal")

    def _save(self):
        if not self._pil_image:
            return

        default_name = f"{self.view_name}.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg *.jpeg"), ("All files", "*.*")],
            initialfile=default_name,
        )
        if not filepath:
            return

        save_image_with_metadata(self._pil_image, filepath, self._metadata)

    def _open_zoom(self):
        if not self._pil_image:
            return
        ZoomWindow(self, self._pil_image, title=f"{self.view_name.title()} view")


class ZoomWindow(tk.Toplevel):
    def __init__(self, master: tk.Widget, image: Image.Image, *, title: str):
        super().__init__(master)
        self.title(title)
        self.image = image
        self.zoom_var = tk.DoubleVar(value=1.0)
        self._rendered: Optional[ImageTk.PhotoImage] = None

        self.canvas = tk.Canvas(self, bg="#111111")
        self.canvas.pack(fill="both", expand=True)
        controls = ttk.Frame(self)
        controls.pack(fill="x")

        ttk.Label(controls, text="Zoom").pack(side="left", padx=8)
        slider = ttk.Scale(
            controls,
            from_=0.25,
            to=3.0,
            orient="horizontal",
            variable=self.zoom_var,
            command=lambda _evt=None: self._render(),
        )
        slider.pack(fill="x", expand=True, padx=(0, 8), pady=6)

        self.canvas.bind("<Configure>", lambda _evt: self._render())

    def _render(self):
        scale = self.zoom_var.get()
        if scale <= 0 or not self.image:
            return

        width = max(1, int(self.image.width * scale))
        height = max(1, int(self.image.height * scale))
        resized = self.image.resize((width, height), Image.Resampling.LANCZOS)
        self._rendered = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self._rendered,
            anchor="center",
        )


class ProductRenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Product Renderer")
        self.geometry("1200x720")

        try:
            self.api_client = ModelAPIClient(config=ModelAPIConfig())
        except MissingAPITokenError as exc:
            messagebox.showerror("Missing Token", str(exc))
            raise SystemExit(1) from exc

        self._current_thread: Optional[threading.Thread] = None
        self.generated_images: Dict[str, Dict[str, Any]] = {}
        self.metadata_bundle: Optional[Dict[str, Any]] = None
        self.latest_metadata_file: Optional[Path] = None

        self._build_ui()

    def _build_ui(self):
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        form = ttk.LabelFrame(container, text="Prompt builder", padding=12)
        form.pack(fill="x")

        ttk.Label(form, text="Product name").grid(row=0, column=0, sticky="w")
        self.product_name = ttk.Entry(form)
        self.product_name.grid(row=0, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(form, text="Design details").grid(row=1, column=0, sticky="w")
        self.design_details = ttk.Entry(form)
        self.design_details.grid(row=1, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(form, text="Extra instructions").grid(row=2, column=0, sticky="nw")
        self.extra_prompt = tk.Text(form, height=4, wrap="word")
        self.extra_prompt.grid(row=2, column=1, sticky="ew", padx=8, pady=4)

        ttk.Label(
            form,
            text="FLUX.1-dev via Hugging Face Inference\nImage size: 1024x1024  •  Steps: 30  •  Guidance: 3.5",
            justify="left",
        ).grid(row=0, column=2, rowspan=3, padx=(16, 0), sticky="nw")

        form.columnconfigure(1, weight=1)

        button_row = ttk.Frame(container)
        button_row.pack(fill="x", pady=12)

        self.generate_button = ttk.Button(button_row, text="Generate Images", command=self._on_generate)
        self.generate_button.pack(side="left")

        self.download_button = ttk.Button(
            button_row, text="Download All…", command=self._on_download_all, state="disabled"
        )
        self.download_button.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(button_row, textvariable=self.status_var).pack(side="right")

        gallery = ttk.LabelFrame(container, text="Generated views", padding=12)
        gallery.pack(fill="both", expand=True)

        self.image_slots = {
            "front": ImageSlot(gallery, "Front"),
            "side": ImageSlot(gallery, "Side"),
            "back": ImageSlot(gallery, "Back"),
        }

        for idx, view in enumerate(["front", "side", "back"]):
            self.image_slots[view].grid(row=0, column=idx, sticky="nsew", padx=8, pady=8)
            gallery.columnconfigure(idx, weight=1)
        gallery.rowconfigure(0, weight=1)

    @staticmethod
    def _compose_user_input(product_name: str, details: str, extra: str) -> str:
        segments = []
        if product_name:
            segments.append(product_name)
        if details:
            segments.append(details)
        if extra:
            segments.append(extra)
        return ". ".join(segments).strip()

    def _on_generate(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Please wait", "Image generation is already in progress.")
            return

        product_name = self.product_name.get().strip()
        details = self.design_details.get().strip()
        extra = self.extra_prompt.get("1.0", "end").strip()
        user_input = self._compose_user_input(product_name, details, extra)

        if not user_input:
            messagebox.showwarning("Missing prompt", "Please describe the product first.")
            return

        product_type = product_name or "product"
        seed, product_id = self.api_client.start_session(user_input)

        view_requirements = {
            "front": "front view, show only the front of the product",
            "back": "back view, show only the back of the product",
            "side": "side view of the same product, 3/4 angle",
        }

        for slot in self.image_slots.values():
            slot.set_placeholder("Generating…")

        self.status_var.set("Generating images… this can take ~20-40 seconds.")
        self.generate_button.configure(state="disabled")
        self.download_button.configure(state="disabled")

        self._current_thread = threading.Thread(
            target=self._run_generation,
            args=(user_input, product_type, view_requirements, seed, product_id),
            daemon=True,
        )
        self._current_thread.start()

    def _run_generation(
        self,
        user_input: str,
        product_type: str,
        view_requirements: Dict[str, str],
        seed: int,
        product_id: str,
    ):
        new_images: Dict[str, Dict[str, Any]] = {}
        metadata_bundle = {
            "product_id": product_id,
            "seed": seed,
            "user_input": user_input,
            "product_type": product_type,
            "model_id": self.api_client._model_id,
            "created_at": datetime.utcnow().isoformat(),
            "views": {},
        }
        try:
            for view, requirement in view_requirements.items():
                self._update_status(f"Rendering {view} view…")
                image, prompt = self.api_client.generate_view(
                    user_input,
                    product_type,
                    extra_notes=requirement,
                    seed=seed,
                )
                metadata = {
                    "prompt": prompt,
                    "view": view,
                    "product_id": product_id,
                    "seed": seed,
                    "model_id": self.api_client._model_id,
                    "width": self.api_client._config.width,
                    "height": self.api_client._config.height,
                    "num_inference_steps": self.api_client._config.inference_steps,
                    "guidance_scale": self.api_client._config.guidance_scale,
                }
                new_images[view] = {"image": image, "metadata": metadata}
                metadata_bundle["views"][view] = metadata
                self._update_slot(view, image, metadata)
        except (ModelAPIError, Exception) as exc:  # broad catch to surface to the UI
            logging.exception("Generation failed")
            error_message = str(exc).strip() or exc.__class__.__name__
            self._update_status(f"Generation failed: {error_message}")
            messagebox.showerror("Generation failed", error_message)
            self._reset_after_generation(success=False)
            return

        self.generated_images = new_images
        self.metadata_bundle = metadata_bundle
        try:
            metadata_path = self._write_metadata_file(metadata_bundle)
            self.latest_metadata_file = metadata_path
            self._update_status(f"All views generated. Metadata saved to {metadata_path.name}.")
        except Exception as exc:
            logging.exception("Failed to write metadata file")
            self._update_status("All views generated (metadata save failed).")
        self._reset_after_generation(success=True)

    def _update_slot(self, view: str, image: Image.Image, metadata: Dict[str, Any]):
        self.after(0, lambda: self.image_slots[view].set_image(image, metadata))

    def _update_status(self, text: str):
        self.after(0, lambda: self.status_var.set(text))

    def _reset_after_generation(self, *, success: bool):
        def _reset():
            self.generate_button.configure(state="normal")
            self.download_button.configure(state="normal" if success else "disabled")

        self.after(0, _reset)

    def _on_download_all(self):
        if not self.generated_images:
            messagebox.showinfo("No images", "Generate images first.")
            return

        target_dir = filedialog.askdirectory(title="Select folder to save renders")
        if not target_dir:
            return

        target_path = Path(target_dir)
        for view, payload in self.generated_images.items():
            filename = f"{view}_view.png"
            save_image_with_metadata(payload["image"], target_path / filename, payload["metadata"])

        messagebox.showinfo("Saved", f"Images saved to {target_dir}")

    def _write_metadata_file(self, metadata: Dict[str, Any]) -> Path:
        output_dir = Path("generated")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{metadata['product_id']}_metadata.json"
        target = output_dir / filename
        with target.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        return target


def main():
    app = ProductRenderApp()
    app.mainloop()


if __name__ == "__main__":
    main()


