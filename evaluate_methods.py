#!/usr/bin/env python3
"""
Evaluate 3 methods (Contours, CLIP+SAM, U²-Net) against a user-drawn box.
This version embeds all configuration at the top so you can just run:
    python /mnt/data/evaluate_methods.py
"""

from pathlib import Path
import os, sys, io, json, time, re, contextlib, importlib.util, shutil
import numpy as np
import cv2


SAMPLE_PATH = Path("img_4170.png")   

BACKGROUND_PATH = Path("img_4169.png")                  

OUT_DIR = Path("eval_out")


MAIN4_PATH = Path("main4.py")        
MAIN5_PATH = Path("main5.py")        


SAM_CHECKPOINT = Path("sam_vit_b_01ec64.pth")  


"""
Evaluate 3 methods (Contours, CLIP+SAM, U²-Net) against a user-drawn box.
This version embeds all configuration at the top so you can just run:
    python /mnt/data/evaluate_methods.py
"""



CONTOURS_NONWHITE_THRESH = 240  
CONTOURS_DIFF_THRESH = 10        



MAIN3_PATH = Path("main3.py")      
def save_method_overlay(img_path, gt_box, pred_box, label, color_bgr, out_path):
    """
    Save a single overlay image with GT (white) and one method box (color_bgr) plus labels.
    Writes to out_path even if pred_box is None (will show GT + 'no bbox').
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # Draw GT in white
    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255,255,255), 2)

    # Draw prediction in color (if present)
    status = "ok"
    if pred_box and all(isinstance(v, int) for v in pred_box):
        cv2.rectangle(img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), color_bgr, 2)
    else:
        status = "no bbox"

    # Legend
    x0, y0 = 12, 28
    cv2.putText(img, f"{label}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)
    y0 += 26
    cv2.putText(img, f"GT: white  |  {label}: colored  |  status: {status}",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)

def method_contours(sample_path, background_path=None,
                    thresh_nonwhite=240, diff_thresh=10):
    """
    Improved contours detector:
      - Crops a thin border to avoid edge-flood artifacts
      - Tries nonwhite threshold; if it floods, tries Otsu/adaptive
      - Rejects near full-frame boxes and tiny specks via area limits
    Returns bbox (x1,y1,x2,y2) or None
    """
    img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    H, W = img.shape[:2]
    IMG_AREA = float(H * W)

   
    BORDER = max(2, int(0.003 * min(H, W)))   
    MIN_AREA_FRAC = 0.001                      
    MAX_AREA_FRAC = 0.90                     


    work = img.copy()
    roi = work[BORDER:H-BORDER, BORDER:W-BORDER]
    if roi.size == 0:
        roi = work  # fallback if image is tiny

    def postprocess_and_bbox(binmask):
        """Find external contours in mask, return bbox in full-image coordinates."""
        if binmask is None or binmask.size == 0:
            return None
        cnts, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
    
        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

    
        x1 = int(x + BORDER)
        y1 = int(y + BORDER)
        x2 = int(x + w + BORDER)
        y2 = int(y + h + BORDER)

 
        box_area = float(w * h)
        if box_area < MIN_AREA_FRAC * IMG_AREA:
            return None  
        if box_area > MAX_AREA_FRAC * IMG_AREA:
            return None 


        if (x1 == 0 or y1 == 0) and (box_area > 0.5 * IMG_AREA):
            return None

        return (x1, y1, x2, y2)

    if background_path:
        bg = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
        if bg is not None:
            if bg.shape != img.shape:
                work = cv2.resize(work, (bg.shape[1], bg.shape[0]))
                H, W = work.shape[:2]
                IMG_AREA = float(H * W)
                roi = work[BORDER:H-BORDER, BORDER:W-BORDER]
                if roi.size == 0:
                    roi = work
                bg = cv2.resize(bg, (W, H))
            # LAB diff + threshold
            bg_blur = cv2.GaussianBlur(bg, (5,5), 0)
            im_blur = cv2.GaussianBlur(work, (5,5), 0)
            diff = cv2.absdiff(cv2.cvtColor(bg_blur, cv2.COLOR_BGR2LAB),
                               cv2.cvtColor(im_blur, cv2.COLOR_BGR2LAB))
            g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(g, diff_thresh, 255, cv2.THRESH_BINARY)
            th = th[BORDER:H-BORDER, BORDER:W-BORDER]
            # Clean up
            kernel = np.ones((3,3), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
            bb = postprocess_and_bbox(th)
            if bb is not None:
                return bb
  


    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


    _, th1 = cv2.threshold(gray, thresh_nonwhite, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=1)
    bb = postprocess_and_bbox(th1)
    if bb is not None:
        return bb

   
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=1)
    bb = postprocess_and_bbox(th2)
    if bb is not None:
        return bb


    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25 if min(H, W) > 300 else 11, C=2
    )
    th3 = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel, iterations=1)
    th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel, iterations=1)
    bb = postprocess_and_bbox(th3)
    if bb is not None:
        return bb

   
    return None



def ensure_dir(path: Path, overwrite: bool = True):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aw = max(0, ax2 - ax1); ah = max(0, ay2 - ay1)
    bw = max(0, bx2 - bx1); bh = max(0, by2 - by1)
    union = aw*ah + bw*bh - inter + 1e-9
    return inter / union

def to_xyxy(x, y, w, h):
    return (int(x), int(y), int(x+w), int(y+h))

def from_xyxy(b):
    x1,y1,x2,y2 = b
    return (int(x1), int(y1), int(x2-x1), int(y2-y1))

def draw_compare(img_path, gt_box, preds, out_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    cv2.rectangle(img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255,255,255), 2)

    colors = {"contours": (0,255,0), "clip_sam": (255,0,0), "u2net": (0,0,255)}
    for name, box in preds:
        color = colors.get(name, (0,255,255))
        if box is not None:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            ty = max(22, box[1] - 6)
            cv2.putText(img, name, (box[0], ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            # If a method has no bbox, still stamp its name at the top-left so it's clear it ran
            cv2.putText(img, f"{name}: no bbox", (10, 24 if name=='contours' else 24+24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def select_user_box(image_path):
    """
    Let the user click 4 points on the image to define a rectangle.
    Suggested usage: click near each *side* (or corner) of the box in clockwise order.
    We compute the axis-aligned bounding box from the 4 points.
    Controls:
      - Left click: add a point (max 4)
      - 'r': reset points
      - ENTER or SPACE: accept if 4 points are placed
      - 'q' or ESC: cancel
    Returns (x1,y1,x2,y2)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot load {image_path}")
    vis = img.copy()
    points = []  # list of (x,y)

    win = "Click 4 points (r=reset, Enter=done)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1200, 800)

    def draw_points(canvas):
        tmp = canvas.copy()
    
        for i,(px,py) in enumerate(points):
            cv2.circle(tmp, (px,py), 6, (0,255,255), -1)
            cv2.putText(tmp, f"{i+1}", (px+8, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
   
        if len(points) >= 2:
            xs = [p[0] for p in points]; ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cv2.rectangle(tmp, (x1,y1), (x2,y2), (255,255,255), 2)
        return tmp

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((int(x), int(y)))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = draw_points(vis)
        cv2.imshow(win, disp)
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32):  
            if len(points) == 4:
                break
        elif key in (ord('r'), ord('R')):
            points = []
        elif key in (27, ord('q'), ord('Q')):
            cv2.destroyWindow(win)
            raise RuntimeError("Selection canceled")

    cv2.destroyWindow(win)
    if len(points) != 4:
        raise RuntimeError("Need exactly 4 points")
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return (int(x1), int(y1), int(x2), int(y2))

def method_contours_via_main3(sample_path, out_dir):
    """
    Prefer main3.get_bbox_from_pair(BACKGROUND_PATH, SAMPLE_PATH).
    Fallback to inline method if the function is missing or fails.
    """
    try:
        mod = import_from(main3_path := MAIN3_PATH, "contours_mod")
        if hasattr(mod, "get_bbox_from_pair") and BACKGROUND_PATH:
            bb = mod.get_bbox_from_pair(str(BACKGROUND_PATH), str(sample_path), segment_width=100)
            if bb and len(bb) == 4:
                return tuple(map(int, bb))
    except Exception:
        pass
   
    return method_contours(sample_path,
                           str(BACKGROUND_PATH) if BACKGROUND_PATH else None,
                           thresh_nonwhite=CONTOURS_NONWHITE_THRESH,
                           diff_thresh=CONTOURS_DIFF_THRESH)

def import_from(path, module_name="temp_mod"):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def method_clip_sam(sample_path, main4_path, sam_ckpt_path, out_dir):
    mod = import_from(main4_path, "clip_sam_mod")

    if hasattr(mod, "IMAGE_PATH"):
        mod.IMAGE_PATH = str(sample_path)
    if hasattr(mod, "OUTPUT_PATH"):
        mod.OUTPUT_PATH = str(out_dir / "clip_sam_out.png")
    if hasattr(mod, "SAM_CHECKPOINT"):
        mod.SAM_CHECKPOINT = str(sam_ckpt_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t0 = time.perf_counter()
        try:
            mod.main()
        except SystemExit:
            return None, None
        t1 = time.perf_counter()
    s = buf.getvalue()
    m = re.search(r"Box @ original size:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", s)
    bbox = None
    if m:
        bbox = tuple(map(int, m.groups()))
    return bbox, (t1 - t0)

def method_u2net(sample_path, main5_path, out_dir):
    """
    Robust U2-Net wrapper that tries multiple ways to get a bbox so we don't end up with None.
    Return: (bbox_xyxy or None, elapsed_seconds)
    """
    mod = import_from(main5_path, "u2_mod")

    t0 = time.perf_counter()
    try:
        res = mod.process_one(
            Path(sample_path), Path(out_dir),
            preview=False, crop=False,
            
            min_area_frac=1e-6,
            alpha_matting=False, am_fg=0, am_bg=0, am_erode=0
        )
    except TypeError:

        res = mod.process_one(Path(sample_path), Path(out_dir))
    except Exception:
        res = None
    t1 = time.perf_counter()


    def bbox_from_mask_img(mask_path):
        if not (mask_path and os.path.exists(mask_path)):
            return None
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            return None

 
        if m.ndim == 3 and m.shape[2] == 4:
            alpha = m[:, :, 3]
            binm = (alpha > 0).astype(np.uint8)
        else:
            if m.ndim == 3:
                gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
            else:
                gray = m
            binm = (gray > 0).astype(np.uint8)

        if binm.sum() == 0:
            return None

        cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return (int(x), int(y), int(x + w), int(y + h))

   
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(k, str) and k.endswith("bbox_json_path") and v and os.path.exists(v):
                try:
                    with open(v, "r") as f:
                        jd = json.load(f)
                    x0 = int(jd["corners"][0]["x"]); y0 = int(jd["corners"][0]["y"])
                    x1 = int(jd["corners"][2]["x"]); y1 = int(jd["corners"][2]["y"])
                    return (x0, y0, x1, y1), (t1 - t0)
                except Exception:
                    pass


    if isinstance(res, dict) and "bbox" in res:
        bb = res["bbox"]
        if bb and len(bb) == 4:
            return tuple(map(int, bb)), (t1 - t0)

    
    if isinstance(res, dict):
        for k, v in res.items():
            if isinstance(k, str) and k.endswith("mask_path"):
                bb = bbox_from_mask_img(v)
                if bb:
                    return bb, (t1 - t0)


    try:
        for p in sorted(Path(out_dir).glob("**/*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                name = p.name.lower()
                if any(tag in name for tag in ("mask", "matte", "alpha", "rgba", "fg")):
                    bb = bbox_from_mask_img(p)
                    if bb:
                        return bb, (t1 - t0)
    except Exception:
        pass


    try:
        for p in sorted(Path(out_dir).glob("*.png")):
            bb = bbox_from_mask_img(p)
            if bb:
                return bb, (t1 - t0)
    except Exception:
        pass

    return None, (t1 - t0)


def main():
    sample_path = SAMPLE_PATH
    bg_path = Path(BACKGROUND_PATH) if BACKGROUND_PATH else None
    out_dir = OUT_DIR
    ensure_dir(out_dir, overwrite=True)

    print("[1/4] Please draw the ground-truth box on the sample image window, then press ENTER.")
    gt = select_user_box(sample_path)
    print(f"[GT] {gt}")

    rows = []
    preds_for_viz = []

    print("[2/4] Running contours (main3)…")
    t0 = time.perf_counter()


    bb_contours = method_contours_via_main3(sample_path, out_dir)


    if bb_contours is None:
        print("    contours fallback -> inline")
        bb_contours = method_contours(
            sample_path,
            str(BACKGROUND_PATH) if BACKGROUND_PATH else None,
            thresh_nonwhite=CONTOURS_NONWHITE_THRESH,
            diff_thresh=CONTOURS_DIFF_THRESH
        )

    t1 = time.perf_counter()
    time_contours = t1 - t0
    iou_c = iou_xyxy(gt, bb_contours) if bb_contours else 0.0
    rows.append(("contours", time_contours, iou_c, bb_contours))
    preds_for_viz.append(("contours", bb_contours))
    print(f"    time={time_contours:.3f}s  IoU={iou_c:.4f}  box={bb_contours}")

    save_method_overlay(sample_path, gt, bb_contours, "contours", (0,255,0), out_dir / "contours_overlay.png")

    print("[3/4] Running CLIP+SAM...")
    bb_cs, time_cs = method_clip_sam(sample_path, MAIN4_PATH, SAM_CHECKPOINT, out_dir)
    iou_cs = iou_xyxy(gt, bb_cs) if bb_cs else 0.0
    rows.append(("clip_sam", time_cs or 0.0, iou_cs, bb_cs))
    preds_for_viz.append(("clip_sam", bb_cs))
    print(f"    time={time_cs if time_cs is not None else float('nan'):.3f}s  IoU={iou_cs:.4f}  box={bb_cs}")

    print("[4/4] Running U²-Net (rembg)...")
    bb_u2, time_u2 = method_u2net(sample_path, MAIN5_PATH, out_dir)
    iou_u2 = iou_xyxy(gt, bb_u2) if bb_u2 else 0.0
    rows.append(("u2net", time_u2 or 0.0, iou_u2, bb_u2))
    preds_for_viz.append(("u2net", bb_u2))
    print(f"    time={time_u2 if time_u2 is not None else float('nan'):.3f}s  IoU={iou_u2:.4f}  box={bb_u2}")

    # Save CSV
    import csv
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "time_sec", "IoU_vs_GT", "box_x1", "box_y1", "box_x2", "box_y2"])
        for name, tsec, iou, box in rows:
            if box is None: box = (None, None, None, None)
            w.writerow([name, f"{tsec:.6f}", f"{iou:.6f}"] + list(box))

    # Save visualization
    comp_path = out_dir / "compare.png"
    draw_compare(sample_path, gt, preds_for_viz, comp_path)

    print("[DONE] Wrote:")
    print(f"  - {csv_path}")
    print(f"  - {comp_path}")

if __name__ == "__main__":
    main()
