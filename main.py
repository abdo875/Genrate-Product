from __future__ import annotations

import threading
import json
from datetime import datetime
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

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


def slugify(value: str) -> str:
    """Convert arbitrary text into a filesystem-friendly slug."""
    cleaned = re.sub(r"[^\w\s-]", "", value.lower()).strip()
    cleaned = re.sub(r"[\s_-]+", "_", cleaned)
    return cleaned or "product"


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
    def __init__(self, master: tk.Widget, view_name: str, *, title: Optional[str] = None):
        super().__init__(master, text=title or view_name.title())
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


class ProductPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Widget, *, title: str, base_name: str):
        super().__init__(master, text=title, padding=8)
        self.base_name = base_name
        self.slots: Dict[str, ImageSlot] = {}

        for idx, view in enumerate(["front", "side", "back"]):
            slot_title = f"{view.title()} view"
            slot = ImageSlot(self, f"{base_name}_{view}", title=slot_title)
            slot.grid(row=0, column=idx, sticky="nsew", padx=4, pady=4)
            self.columnconfigure(idx, weight=1)
            self.slots[view] = slot

    def set_placeholder(self, text: str):
        for slot in self.slots.values():
            slot.set_placeholder(text)

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
        self.generated_images: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.metadata_bundle: Optional[Dict[str, Any]] = None
        self.latest_metadata_file: Optional[Path] = None
        self.product_panels: Dict[str, ProductPanel] = {}
        self.current_product_entries: List[Dict[str, Any]] = []

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

        self.gallery_canvas = tk.Canvas(gallery, highlightthickness=0)
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery_scrollbar = ttk.Scrollbar(gallery, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_scrollbar.pack(side="right", fill="y")
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scrollbar.set)

        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")

        def _sync_scroll_region(_evt=None):
            self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

        self.gallery_inner.bind("<Configure>", _sync_scroll_region)
        self.gallery_canvas.bind(
            "<Configure>",
            lambda event: self.gallery_canvas.itemconfigure(self.gallery_window, width=event.width),
        )

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

    @staticmethod
    def _split_list(raw_value: str) -> List[str]:
        return [segment.strip() for segment in raw_value.split(",") if segment.strip()]

    def _prepare_product_entries(self, products_raw: str, designs_raw: str, extra: str) -> List[Dict[str, Any]]:
        product_names = self._split_list(products_raw)
        design_details = self._split_list(designs_raw)
        entries: List[Dict[str, Any]] = []
        if not product_names:
            return entries

        used_slugs: Dict[str, int] = {}
        for idx, name in enumerate(product_names):
            design = ""
            if design_details:
                design = design_details[idx] if idx < len(design_details) else design_details[-1]
            user_input = self._compose_user_input(name, design, extra)
            seed, product_id = self.api_client.start_session(user_input)
            label = f"{name} ({design})" if design else name
            base_slug = slugify(label)
            slug_count = used_slugs.get(base_slug, 0)
            used_slugs[base_slug] = slug_count + 1
            slug = f"{base_slug}_{slug_count + 1}" if slug_count else base_slug
            entries.append(
                {
                    "product_name": name,
                    "design": design,
                    "label": label.strip(),
                    "user_input": user_input,
                    "seed": seed,
                    "product_id": product_id,
                    "slug": slug,
                }
            )
        return entries

    def _build_product_gallery(self, entries: List[Dict[str, Any]]):
        for child in self.gallery_inner.winfo_children():
            child.destroy()
        self.product_panels.clear()
        for col in range(4):
            self.gallery_inner.columnconfigure(col, weight=0)

        if not entries:
            ttk.Label(
                self.gallery_inner,
                text="Enter comma-separated products (and optional designs) before generating images.",
                justify="center",
                wraplength=560,
            ).pack(fill="x", padx=12, pady=12)
            return

        columns = 2 if len(entries) > 1 else 1
        for idx, entry in enumerate(entries):
            panel = ProductPanel(self.gallery_inner, title=entry["label"], base_name=entry["slug"])
            row = idx // columns
            col = idx % columns
            panel.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            self.gallery_inner.columnconfigure(col, weight=1)
            self.gallery_inner.rowconfigure(row, weight=1)
            panel.set_placeholder("Waiting to generate…")
            self.product_panels[entry["product_id"]] = panel

    def _on_generate(self):
        if self._current_thread and self._current_thread.is_alive():
            messagebox.showinfo("Please wait", "Image generation is already in progress.")
            return

        products_raw = self.product_name.get().strip()
        details_raw = self.design_details.get().strip()
        extra = self.extra_prompt.get("1.0", "end").strip()

        product_entries = self._prepare_product_entries(products_raw, details_raw, extra)

        if not product_entries:
            messagebox.showwarning("Missing prompt", "Please enter at least one product.")
            return

        self.current_product_entries = product_entries
        self.generated_images = {}
        self.metadata_bundle = None
        self._build_product_gallery(product_entries)

        view_requirements = {
            "front": "front view, show only the front of the product",
            "back": "back view, show only the back of the product",
            "side": "side view of the same product, 3/4 angle",
        }

        for panel in self.product_panels.values():
            panel.set_placeholder("Generating…")

        product_count = len(product_entries)
        self.status_var.set(f"Generating {product_count * 3} images… this can take a minute.")
        self.generate_button.configure(state="disabled")
        self.download_button.configure(state="disabled")

        self._current_thread = threading.Thread(
            target=self._run_generation,
            args=([entry.copy() for entry in product_entries], view_requirements),
            daemon=True,
        )
        self._current_thread.start()

    def _run_generation(
        self,
        product_entries: List[Dict[str, Any]],
        view_requirements: Dict[str, str],
    ):
        new_images: Dict[str, Dict[str, Dict[str, Any]]] = {}
        metadata_bundle = {
            "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "model_id": self.api_client._model_id,
            "created_at": datetime.utcnow().isoformat(),
            "products": [],
        }
        try:
            for entry in product_entries:
                product_meta = {
                    "product_id": entry["product_id"],
                    "product_name": entry["product_name"],
                    "design": entry["design"],
                    "label": entry["label"],
                    "seed": entry["seed"],
                    "user_input": entry["user_input"],
                    "views": {},
                }
                for view, requirement in view_requirements.items():
                    self._update_status(f"Rendering {entry['label']} - {view} view…")
                    image, prompt = self.api_client.generate_view(
                        entry["user_input"],
                        entry["product_name"],
                        extra_notes=requirement,
                        seed=entry["seed"],
                    )
                    metadata = {
                        "prompt": prompt,
                        "view": view,
                        "product_id": entry["product_id"],
                        "seed": entry["seed"],
                        "model_id": self.api_client._model_id,
                        "width": self.api_client._config.width,
                        "height": self.api_client._config.height,
                        "num_inference_steps": self.api_client._config.inference_steps,
                        "guidance_scale": self.api_client._config.guidance_scale,
                    }
                    product_store = new_images.setdefault(entry["product_id"], {})
                    product_store[view] = {"image": image, "metadata": metadata}
                    product_meta["views"][view] = metadata
                    self._update_slot(entry["product_id"], view, image, metadata)
                metadata_bundle["products"].append(product_meta)
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

    def _update_slot(self, product_id: str, view: str, image: Image.Image, metadata: Dict[str, Any]):
        def _apply():
            panel = self.product_panels.get(product_id)
            if not panel:
                return
            slot = panel.slots.get(view)
            if slot:
                slot.set_image(image, metadata)

        self.after(0, _apply)

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
        saved_files = 0
        for entry in self.current_product_entries:
            product_payload = self.generated_images.get(entry["product_id"])
            if not product_payload:
                continue
            base_name = entry.get("slug") or slugify(entry["label"])
            for view, payload in product_payload.items():
                filename = f"{base_name}_{view}.png"
                save_image_with_metadata(payload["image"], target_path / filename, payload["metadata"])
                saved_files += 1

        if saved_files:
            messagebox.showinfo("Saved", f"{saved_files} images saved to {target_dir}")
        else:
            messagebox.showwarning("No images", "Nothing to save yet.")

    def _write_metadata_file(self, metadata: Dict[str, Any]) -> Path:
        output_dir = Path("generated")
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_id = metadata.get("batch_id") or f"batch_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        filename = f"{batch_id}_metadata.json"
        target = output_dir / filename
        with target.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        return target


def main():
    app = ProductRenderApp()
    app.mainloop()


if __name__ == "__main__":
    main()


