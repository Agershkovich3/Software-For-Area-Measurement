# Install:
#   pip install opencv-python pillow numpy
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#   pip install transformers
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

IMAGE_PATH     = "img_4177.png"
TEXT_PROMPT    = "biological tissue sample"
OUTPUT_PATH    = "clip_sam_boxed.png"

SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_MAX_SIDE = 640

MIN_AREA_FRAC   = 0.002
MAX_AREA_FRAC   = 0.60

POINTS_PER_SIDE = 8
PRED_IOU_THRESH = 0.80
STAB_SCORE_TH   = 0.90
MIN_MASK_AREA   = 512

DRAW_OVERLAY    = True
BOX_COLOR       = (0, 255, 0)
BOX_THICKNESS   = 3

NEGATIVE_PROMPTS = [
    "background", "empty background", "blank surface", "cloth",
    "sheet", "table", "edge of frame", "shadow", "reflection", "label"
]
NEG_WEIGHT = 0.7

HUGE_BOX_FRAC   = 0.50
HUGE_BOX_PEN    = 0.35
EDGE_PENALTY    = 0.15
THINNESS_PENALTY= 0.15

ERODE_PIX = 1

def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def load_image_bgr(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        fail(f"Image not found or unreadable: {path}")
    return bgr

def maybe_downscale(img_bgr, max_side):
    if max_side <= 0:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img_bgr, 1.0
    scale = max_side / float(s)
    out = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return out, scale

@torch.no_grad()
def load_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval().to(device)
    return model, proc

@torch.no_grad()
def clip_sim_multi(model, proc, device, pil_img, pos_text, neg_texts):
    in_pos = proc(text=[pos_text], images=pil_img, return_tensors="pt", padding=True).to(device)
    out_pos = model(**in_pos)
    img = out_pos.image_embeds / out_pos.image_embeds.norm(dim=-1, keepdim=True)
    txt = out_pos.text_embeds  / out_pos.text_embeds.norm(dim=-1, keepdim=True)
    s_pos = (img @ txt.T).squeeze().item()
    if not neg_texts:
        return s_pos, 0.0, s_pos
    in_neg = proc(text=neg_texts, images=pil_img, return_tensors="pt", padding=True).to(device)
    out_neg = model(**in_neg)
    tneg = out_neg.text_embeds / out_neg.text_embeds.norm(dim=-1, keepdim=True)
    s_neg = float((img @ tneg.T).squeeze(0).max().item())
    return s_pos, s_neg, s_pos - NEG_WEIGHT * s_neg

def mask_bbox(seg):
    ys, xs = np.where(seg)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def crop_dim_bg(rgb, seg, box, pad=8):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1-pad), max(0, y1-pad)
    x2, y2 = min(w-1, x2+pad), min(h-1, y2+pad)
    crop = rgb[y1:y2+1, x1:x2+1].copy()
    m = seg[y1:y2+1, x1:x2+1].astype(bool)
    bg = (crop.astype(np.float32) * 0.25).astype(np.uint8)
    out = crop.copy()
    out[~m] = bg[~m]
    return out

def gen_masks_sam(work_rgb, model_type, ckpt, points, iou_th, stab_th, min_area):
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=DEVICE); sam.eval()
    mg = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points,
        pred_iou_thresh=iou_th,
        stability_score_thresh=stab_th,
        box_nms_thresh=0.7,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area,
    )
    masks = mg.generate(work_rgb)
    return masks

def box_area(bb):
    x1, y1, x2, y2 = bb
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

def touches_edges(bb, w, h, tol=2):
    x1, y1, x2, y2 = bb
    t = 0
    if x1 <= tol: t += 1
    if y1 <= tol: t += 1
    if x2 >= w-1-tol: t += 1
    if y2 >= h-1-tol: t += 1
    return t

def mask_compactness(seg, bb):
    x1, y1, x2, y2 = bb
    sub = seg[y1:y2+1, x1:x2+1]
    return float(sub.sum()) / float(sub.size + 1e-6)

def erode_mask_boolean(seg, k=ERODE_PIX):
    if k <= 0: return seg
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    seg_u8 = (seg.astype(np.uint8) * 255)
    er = cv2.erode(seg_u8, kernel, iterations=1)
    return er.astype(bool)

def main():
    if not os.path.exists(IMAGE_PATH): fail(f"IMAGE_PATH not found: {IMAGE_PATH}")
    if not os.path.exists(SAM_CHECKPOINT): fail(f"SAM_CHECKPOINT not found: {SAM_CHECKPOINT}")

    orig_bgr = load_image_bgr(IMAGE_PATH)
    H, W = orig_bgr.shape[:2]
    work_bgr, scale = maybe_downscale(orig_bgr, RESIZE_MAX_SIDE)
    work_rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
    h, w = work_rgb.shape[:2]
    total_area = float(h * w)

    masks = gen_masks_sam(work_rgb, SAM_MODEL_TYPE, SAM_CHECKPOINT,
                          POINTS_PER_SIDE, PRED_IOU_THRESH, STAB_SCORE_TH, MIN_MASK_AREA)
    if len(masks) == 0:
        masks = gen_masks_sam(work_rgb, SAM_MODEL_TYPE, SAM_CHECKPOINT,
                              max(6, POINTS_PER_SIDE+8), 0.70, 0.86, max(64, MIN_MASK_AREA//2))
        if len(masks) == 0:
            x1, y1 = int(0.2*w), int(0.2*h)
            x2, y2 = int(0.8*w), int(0.8*h)
            m = {"segmentation": np.zeros((h,w), dtype=bool)}
            m["segmentation"][y1:y2+1, x1:x2+1] = True
            masks = [m]

    kept = []
    for m in masks:
        seg = m["segmentation"].astype(bool)
        bb  = mask_bbox(seg)
        if bb is None: continue
        fr = box_area(bb) / (total_area + 1e-6)
        if fr < MIN_AREA_FRAC: continue
        if fr > MAX_AREA_FRAC: continue
        kept.append((seg, bb))

    if not kept:
        largest = max(masks, key=lambda m: m.get("area", 0))
        seg = largest["segmentation"].astype(bool)
        bb  = mask_bbox(seg) or (int(0.2*w), int(0.2*h), int(0.8*w), int(0.8*h))
        kept = [(seg, bb)]

    model, proc = load_clip(DEVICE)
    best = {"score": -1e9, "box": None, "seg": None}
    for seg_raw, bb in kept:
        seg = erode_mask_boolean(seg_raw, k=ERODE_PIX)
        crop = crop_dim_bg(work_rgb, seg, bb, pad=8)
        pil  = Image.fromarray(crop)

        s_pos, s_neg, s_contrast = clip_sim_multi(model, proc, DEVICE, pil, TEXT_PROMPT, NEGATIVE_PROMPTS)

        box_fr   = box_area(bb) / (total_area + 1e-6)
        edges    = touches_edges(bb, w, h)
        compact  = mask_compactness(seg, bb)

        penalty = 0.0
        if box_fr > HUGE_BOX_FRAC:
            penalty += HUGE_BOX_PEN * (box_fr - HUGE_BOX_FRAC) / max(1e-6, 1.0 - HUGE_BOX_FRAC)
        if edges > 0:
            penalty += EDGE_PENALTY * edges
        if compact < 0.30:
            penalty += THINNESS_PENALTY * (0.30 - compact) / 0.30

        final_score = s_contrast - penalty

        if final_score > best["score"]:
            best = {"score": final_score, "box": bb, "seg": seg}

    if best["box"] is None:
        fail("No candidate selected (unexpected).")

    inv = 1.0 / (scale if scale > 0 else 1.0)
    bx1, by1, bx2, by2 = best["box"]
    X1, Y1 = int(round(bx1*inv)), int(round(by1*inv))
    X2, Y2 = int(round(bx2*inv)), int(round(by2*inv))
    X1, Y1 = max(0, X1), max(0, Y1)
    X2, Y2 = min(W-1, X2), min(H-1, Y2)

    out = orig_bgr.copy()
    cv2.rectangle(out, (X1, Y1), (X2, Y2), BOX_COLOR, BOX_THICKNESS)
    if DRAW_OVERLAY and best["seg"] is not None:
        seg_small = best["seg"].astype(np.uint8)*255
        seg_orig  = cv2.resize(seg_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = out.copy()
        overlay[seg_orig] = (overlay[seg_orig]*0.5 + np.array(BOX_COLOR, np.uint8)*0.5).astype(np.uint8)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

    if not cv2.imwrite(OUTPUT_PATH, out):
        fail("cv2.imwrite failed (check permissions/path)")

    print(f"[OK] Saved: {OUTPUT_PATH}")
    print(f"[OK] Best CLIP score: {best['score']:.4f}")
    print(f"[OK] Box @ original size: {(X1, Y1, X2, Y2)}")

if __name__ == "__main__":
    main()
