#!/usr/bin/env python3


from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPSegForImageSegmentation



IMAGE_PATH = "DocCam2.png"  
TEXT_PROMPT = "all red biological tissue sample"
OUT_DIR = "outputs_clipseg"

BASE_PROB_THRESHOLD = 0.5

MIN_POSITIVE_PIXELS = 50

SAVE_OUTPUTS = True

CLIPSEG_MODEL_NAME = "CIDAS/clipseg-rd64-refined"



def load_clipseg_model(
    device: Optional[torch.device] = None
):
    """
    Load CLIPSeg model + processor.

    Returns:
        processor, model, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(CLIPSEG_MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model, device


def run_clipseg_on_pil(
    image: Image.Image,
    text_prompt: str,
    processor,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Run CLIPSeg on a PIL image and a text prompt.

    Args:
        image: PIL.Image in RGB.
        text_prompt: description of the region of interest.
        processor, model, device: from load_clipseg_model().

    Returns:
        prob_map_resized: (H, W) float32 in [0,1]
    """
    # No 'padding' arg — avoids the warning you saw
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
       
        logits = outputs.logits 

    probs = torch.sigmoid(logits) 
    prob_map = probs[0].cpu().numpy().astype(np.float32)  

    orig_w, orig_h = image.size 
    prob_map_resized = cv2.resize(
        prob_map,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    print(
        "[ClipSeg] prob stats:",
        "min =", float(prob_map_resized.min()),
        "max =", float(prob_map_resized.max()),
        "mean =", float(prob_map_resized.mean()),
    )

    return prob_map_resized



def prob_to_mask(
    prob_map: np.ndarray,
    base_threshold: float = BASE_PROB_THRESHOLD,
    min_pixels: int = MIN_POSITIVE_PIXELS,
) -> Tuple[np.ndarray, float, str]:
    """
    Convert probability map to a binary mask with robust fallback.

    Strategy:
    1. Try fixed threshold = base_threshold.
    2. If too few pixels, try quantile thresholds: 0.95, 0.9, 0.85, 0.8.
    3. If still too few, use max-probability pixel + dilation.

    Returns:
        mask: uint8 {0,1} of shape (H, W)
        threshold_used: float
        mode: description string
    """
    H, W = prob_map.shape
    total_pixels = H * W

    
    mask = (prob_map >= base_threshold).astype(np.uint8)
    count = int(mask.sum())
    print(f"[ClipSeg] Fixed threshold {base_threshold:.3f} → {count} positive pixels")

    if count >= min_pixels:
        return mask, base_threshold, "fixed"

    
    flat = prob_map.flatten()
    for q in [0.95, 0.9, 0.85, 0.8]:
        thr = float(np.quantile(flat, q))
        mask = (prob_map >= thr).astype(np.uint8)
        count = int(mask.sum())
        print(
            f"[ClipSeg] Quantile {q:.2f} (thr={thr:.3f}) "
            f"→ {count} positive pixels"
        )
        if count >= min_pixels:
            return mask, thr, f"quantile_{q:.2f}"


    print("[ClipSeg] Falling back to max-pixel dilation strategy...")
    y_max, x_max = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    mask = np.zeros_like(prob_map, dtype=np.uint8)
    mask[y_max, x_max] = 1

    kernel_size = max(5, int(min(H, W) * 0.02))  
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    count = int(mask.sum())
    thr_used = float(prob_map[y_max, x_max])

    print(
        f"[ClipSeg] Max-pixel dilation: kernel={kernel_size} "
        f"→ {count} positive pixels"
    )

    return mask, thr_used, "maxpixel_dilate"




def largest_cc_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Find largest connected component in mask and return bounding box.

    Args:
        mask: (H, W) uint8 in {0,1}

    Returns:
        (x_min, y_min, x_max, y_max) or None if no foreground.
    """
    mask_255 = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_255, connectivity=8
    )

    if num_labels <= 1:
   
        return None


    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    x_min = x
    y_min = y
    x_max = x + w - 1
    y_max = y + h - 1

    print(
        f"[ClipSeg] Largest CC label={best_label}, area={int(areas.max())} "
        f"bbox=({x_min}, {y_min}, {x_max}, {y_max})"
    )

    return x_min, y_min, x_max, y_max



def save_visualizations(
    image: Image.Image,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    out_dir: Path,
    base_name: str,
) -> None:
    """
    Save probability map, mask, and overlay (image + mask + bbox) to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(image)  
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    prob_vis = (prob_map * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{base_name}_prob.png"), prob_vis)


    mask_vis = (binary_mask * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{base_name}_mask.png"), mask_vis)

    overlay = img_bgr.copy()
    mask_bool = binary_mask.astype(bool)
    color = np.array([0, 0, 255], dtype=np.uint8)  # red in BGR

    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * color
    ).astype(np.uint8)

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),  # green
            thickness=2,
        )

    cv2.imwrite(str(out_dir / f"{base_name}_overlay.png"), overlay)



def clipseg_predict_bbox(
    image_path: Path,
    text_prompt: str,
    save_outputs: bool = SAVE_OUTPUTS,
    out_dir: Path = Path(OUT_DIR),
):
    """
    High-level function:
    - Loads model
    - Runs CLIPSeg
    - Converts prob map to mask (with robust fallback)
    - Extracts largest CC bbox
    - Optionally saves outputs
    - Returns bbox + mask
    """
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[ClipSeg] Loading model on device...")
    processor, model, device = load_clipseg_model()

    image = Image.open(str(image_path)).convert("RGB")

    print(f"[ClipSeg] Running CLIPSeg with prompt: '{text_prompt}'")
    prob_map = run_clipseg_on_pil(
        image=image,
        text_prompt=text_prompt,
        processor=processor,
        model=model,
        device=device,
    )

    mask, thr_used, mode = prob_to_mask(prob_map)
    print(
        f"[ClipSeg] Final mask mode='{mode}', threshold_used={thr_used:.4f}, "
        f"pixels={int(mask.sum())}"
    )

    bbox = largest_cc_bbox(mask)

    if save_outputs:
        save_visualizations(
            image=image,
            prob_map=prob_map,
            binary_mask=mask,
            bbox=bbox,
            out_dir=out_dir,
            base_name=image_path.stem,
        )

    return bbox, mask


def main():
    image_path = Path(IMAGE_PATH)
    out_dir = Path(OUT_DIR)

    print(f"[ClipSeg] Using image: {image_path}")
    print(f"[ClipSeg] Text prompt: '{TEXT_PROMPT}'")
    print(f"[ClipSeg] Save outputs: {SAVE_OUTPUTS} -> {out_dir}")

    bbox, mask = clipseg_predict_bbox(
        image_path=image_path,
        text_prompt=TEXT_PROMPT,
        save_outputs=SAVE_OUTPUTS,
        out_dir=out_dir,
    )

    if bbox is None:
        print("[ClipSeg] WARNING: no connected foreground region found.")
    else:
        x_min, y_min, x_max, y_max = bbox
        h, w = mask.shape
        print("[ClipSeg] Bounding box (x_min, y_min, x_max, y_max):")
        print(f"    ({x_min}, {y_min}, {x_max}, {y_max})")
        print(
            f"[ClipSeg] Box size: {x_max - x_min} x {y_max - y_min} "
            f"within image size {w} x {h}"
        )


if __name__ == "__main__":
    main()
