#!/usr/bin/env python3
"""
Foreground separation with U²-Net (via rembg)

Outputs (per image):
  - <name>_fg.png         -> foreground with transparent background
  - <name>_mask.png       -> grayscale mask (255=FG, 0=BG)
  - <name>_preview.png    -> (optional) overlay for quick QA
  - <name>_fg_cropped.png -> (optional) tight crop around main subject

Usage:
  # Single image
  python u2net_fg_separation.py input.jpg --preview --crop

  # Folder (recursively processes supported formats)
  python u2net_fg_separation.py /path/to/folder -o out --preview --crop

Install:
  pip install rembg pillow opencv-python numpy
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict
import json

import numpy as np
from PIL import Image

# --------------------------
# Config / supported formats
# --------------------------
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -------------
# I/O utilities
# -------------
def load_image(path: Path) -> Image.Image:
    """Load as RGBA to preserve alpha if present."""
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im


def save_image(im: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def ensure_out_dir(in_path: Path, out: Optional[str]) -> Path:
    if out:
        return Path(out)
    return (in_path.parent if in_path.is_file() else in_path) / "u2net_out"


def iter_images(root: Path):
    """Yield image files from a file or directory (recursive)."""
    if root.is_file():
        if root.suffix.lower() in VALID_SUFFIXES:
            yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES:
            yield p
def draw_box_and_save(original_rgba: Image.Image, bbox, out_path: Path):
    """Draw a green rectangle on top of the original RGB and save PNG."""
    import cv2
    arr = np.array(original_rgba.convert("RGB"))  # (H,W,3)
    x0, y0, x1, y1 = map(int, bbox)
    cv2.rectangle(arr, (x0, y0), (x1, y1), (0, 255, 0), 2)
    Image.fromarray(arr).save(out_path)


# -----------------
# Mask postprocess
# -----------------
def overlay_preview(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create a visibility overlay: teal fill + green edges on top of original RGB."""
    import cv2
    color = np.zeros_like(rgb)
    color[..., 1] = 200
    color[..., 2] = 200

    mask_f = (mask.astype(np.float32) / 255.0)[..., None]
    overlay = rgb.astype(np.float32) * (1 - alpha * mask_f) + color.astype(np.float32) * (alpha * mask_f)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    edges = cv2.Canny(mask, 50, 150)
    overlay[edges > 0] = [0, 255, 0]
    return overlay


def compute_mask_bbox(mask: np.ndarray, min_area_frac: float = 0.001) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) for the largest connected component above area threshold."""
    import cv2
    h, w = mask.shape[:2]
    area = h * w

    _, binmask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area_frac * area:
        return None
    x, y, bw, bh = cv2.boundingRect(largest)
    return (x, y, x + bw, y + bh)


def composite_foreground(original_rgba: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    """Replace alpha channel of original with predicted mask."""
    arr = np.array(original_rgba, dtype=np.uint8)
    if mask_u8.shape[:2] != arr.shape[:2]:
        # Safety resize (shouldn't happen with this pipeline)
        mask_u8 = np.array(Image.fromarray(mask_u8).resize((arr.shape[1], arr.shape[0]), Image.BILINEAR))
    arr[..., 3] = mask_u8
    return Image.fromarray(arr, mode="RGBA")

from rembg import remove, new_session
from rembg import remove, new_session
import io
import numpy as np
from PIL import Image

_REMBG_SESSION = None

def _get_rembg_session(model: str = "u2net"):
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        # On Apple Silicon, if you hit provider issues, try:
        # _REMBG_SESSION = new_session(model, providers=["CPUExecutionProvider"])
        _REMBG_SESSION = new_session(model)
    return _REMBG_SESSION

def rembg_remove(im_rgba: Image.Image,
                 alpha_matting: bool = False,
                 am_fg: int = 240,
                 am_bg: int = 10,
                 am_erode: int = 10):
    """
    Returns:
      - fg_rgba: PIL RGBA image with background removed
      - mask: np.uint8 HxW mask (255=FG, 0=BG)
    Works with rembg versions that DO or DO NOT support return_mask.
    """
    # Encode the input PIL image to PNG bytes (robust for rembg)
    buf = io.BytesIO()
    im_rgba.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    session = _get_rembg_session("u2net")

    # Try modern API that returns (content, mask). Fallback to old API.
    try:
        result = remove(
            data=png_bytes,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=am_fg,
            alpha_matting_background_threshold=am_bg,
            alpha_matting_erode_size=am_erode,
            only_mask=False,
            post_process_mask=True,
            return_mask=True,   # may not exist on older rembg
            bgcolor=None,
        )
        # Newer rembg: tuple of bytes or PILs
        if isinstance(result, tuple) and len(result) == 2:
            content, mask_img = result
            # Normalize to PIL objects
            if isinstance(content, (bytes, bytearray)):
                fg = Image.open(io.BytesIO(content)).convert("RGBA")
            else:
                fg = content.convert("RGBA")

            if isinstance(mask_img, (bytes, bytearray)):
                mask_pil = Image.open(io.BytesIO(mask_img)).convert("L")
            else:
                mask_pil = mask_img.convert("L")

            mask = np.array(mask_pil, dtype=np.uint8)
            return fg, mask

        # If we didn’t get a tuple, fall through to derive mask
        if isinstance(result, (bytes, bytearray)):
            fg = Image.open(io.BytesIO(result)).convert("RGBA")
        else:
            fg = result.convert("RGBA")

    except TypeError:
        # Old rembg that doesn’t accept return_mask
        out_bytes = remove(
            data=png_bytes,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=am_fg,
            alpha_matting_background_threshold=am_bg,
            alpha_matting_erode_size=am_erode,
            only_mask=False,
            post_process_mask=True,
            bgcolor=None,
        )
        fg = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    # Derive mask from alpha if no explicit mask was returned
    alpha = np.array(fg.split()[-1], dtype=np.uint8)
    return fg, alpha

# -------------
# Core routine
# -------------
def process_one(in_path: Path,
                out_dir: Path,
                *,
                preview: bool,
                crop: bool,
                min_area_frac: float,
                alpha_matting: bool,
                am_fg: int,
                am_bg: int,
                am_erode: int) -> Dict[str, Optional[str]]:
    """Process a single image and save outputs; return dict of file paths."""
    original = load_image(in_path)                 # RGBA
    rgb = np.array(original, dtype=np.uint8)[..., :3]

    # Inference
    fg_rgba, mask = rembg_remove(
        original,
        alpha_matting=alpha_matting,
        am_fg=am_fg,
        am_bg=am_bg,
        am_erode=am_erode,
    )

    stem = in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save foreground (transparent)
    fg_path = out_dir / f"{stem}_fg.png"
    save_image(fg_rgba, fg_path)

    # Save mask
    mask_path = out_dir / f"{stem}_mask.png"
    Image.fromarray(mask, mode="L").save(mask_path)

    # Optional preview overlay
    preview_path = None
    if preview:
        prev = overlay_preview(rgb, mask, alpha=0.5)
        preview_path = out_dir / f"{stem}_preview.png"
        Image.fromarray(prev).save(preview_path)

      # Optional tight crop + box + bbox json
    crop_path = None
    boxed_path = None
    bbox_json_path = None

    bbox = compute_mask_bbox(mask, min_area_frac=min_area_frac)
    if bbox is not None:
        x0, y0, x1, y1 = map(int, bbox)

        # Boxed image over original
        boxed_path = out_dir / f"{stem}_boxed.png"
        draw_box_and_save(original, (x0, y0, x1, y1), boxed_path)

        # Save bbox JSON
        corners = [
            {"x": x0, "y": y0},  # top-left
            {"x": x1, "y": y0},  # top-right
            {"x": x1, "y": y1},  # bottom-right
            {"x": x0, "y": y1},  # bottom-left
        ]
        h, w = mask.shape[:2]
        bbox_record = {
            "image": str(in_path),
            "width": w,
            "height": h,
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "corners": corners,
        }
        bbox_json_path = out_dir / f"{stem}_bbox.json"
        with open(bbox_json_path, "w") as f:
            json.dump(bbox_record, f, indent=2)

        # Optional tight crop (if you still want it behind --crop)
        if crop:
            fg_rgba.crop((x0, y0, x1, y1)).save(out_dir / f"{stem}_fg_cropped.png")
            crop_path = str(out_dir / f"{stem}_fg_cropped.png")

        # Print coordinates to stdout
        print(f"[BBOX] {stem}: "
              f"TL=({x0},{y0}) TR=({x1},{y0}) BR=({x1},{y1}) BL=({x0},{y1})")

    return {
        "input": str(in_path),
        "foreground_png": str(fg_path),
        "mask_png": str(mask_path),
        "preview_png": str(preview_path) if preview_path else None,
        "boxed_png": str(boxed_path) if boxed_path else None,
        "bbox_json": str(bbox_json_path) if bbox_json_path else None,
        "cropped_png": str(crop_path) if crop_path else None,
    }

# ------
#  CLI
# ------
def parse_args():
    p = argparse.ArgumentParser(description="Separate foreground from background using U²-Net (rembg).")
    p.add_argument("input", type=str, help="Path to an image file or a folder of images")
    p.add_argument("-o", "--out", type=str, default="", help="Output directory (default: <input>/u2net_out)")
    p.add_argument("--preview", action="store_true", help="Save an overlay preview PNG")
    p.add_argument("--crop", action="store_true", help="Save a tight-cropped foreground PNG")
    p.add_argument("--min-area-frac", type=float, default=0.001,
                   help="Min connected area (fraction of image) for crop bbox (default: 0.001)")
    # Edge refinement (helpful on hair/fine edges or microscope fringes)
    p.add_argument("--alpha-matting", action="store_true",
                   help="Enable alpha matting for cleaner edges (slower)")
    p.add_argument("--am-fg", type=int, default=240, help="Alpha-matting foreground threshold (default: 240)")
    p.add_argument("--am-bg", type=int, default=10,  help="Alpha-matting background threshold (default: 10)")
    p.add_argument("--am-erode", type=int, default=10, help="Alpha-matting erode size (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input path does not exist: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = ensure_out_dir(in_path, args.out)

    # --- NEW: clear old results if folder exists ---
    import shutil
    if out_dir.exists():
        print(f"[INFO] Removing previous output folder: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------

    any_done = False

    for img_path in iter_images(in_path):
        any_done = True
        res = process_one(
            img_path,
            out_dir,
            preview=args.preview,
            crop=args.crop,
            min_area_frac=args.min_area_frac,
            alpha_matting=args.alpha_matting,
            am_fg=args.am_fg,
            am_bg=args.am_bg,
            am_erode=args.am_erode,
        )
        print(f"[OK] {res['input']} -> {res['foreground_png']}")

    if not any_done:
        print("No supported images found.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
