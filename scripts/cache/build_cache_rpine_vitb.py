
# -*- coding: utf-8 -*-
"""
Precompute frozen SAM image encoder features for cache-only TCA training.

Output cache format:
    torch.save({
        "feat": Tensor[256, 64, 64],
        "file_name": ...,
        "image_path": ...,
        "sam_model_type": ...,
        "sam_img_size": 1024,
        "orig_size": (H, W),
        "input_size": predictor.input_size,
    }, out_path)

The training scripts can read either a raw tensor or a dict with key "feat".
"""
import os

CUDA_VISIBLE_DEVICES = r"0"  # change to "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------------
# Runtime options
# ---------------------------------------------------------
BATCH_PRINT_EVERY = 50
SKIP_EXISTING = True
SAVE_FP16 = True
ALLOW_TF32 = True
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pil_read_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def list_images(image_dir):
    image_dir = Path(image_dir)
    items = []
    if not image_dir.is_dir():
        return items
    for ext in IMAGE_EXTS:
        items.extend(image_dir.rglob(f"*{ext}"))
        items.extend(image_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(items), key=lambda p: str(p).lower())


def find_image_file(image_dir, file_or_stem):
    image_dir = Path(image_dir)
    p = image_dir / file_or_stem
    if p.is_file():
        return p
    stem = Path(file_or_stem).stem
    for ext in IMAGE_EXTS:
        for e in (ext, ext.upper()):
            p = image_dir / f"{stem}{e}"
            if p.is_file():
                return p
    return None


def build_predictor(device, sam_model_type, sam_checkpoint):
    if ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def save_one_feature(predictor, image_path, out_path, sam_model_type):
    image_path = Path(image_path)
    out_path = Path(out_path)
    if SKIP_EXISTING and out_path.is_file():
        return "skip"

    image = pil_read_rgb(image_path)
    h, w = image.shape[:2]

    with torch.no_grad():
        predictor.set_image(image)
        feat = predictor.features.detach().cpu()
        if feat.ndim == 4 and feat.shape[0] == 1:
            feat = feat[0]
        if SAVE_FP16:
            feat = feat.half()
        else:
            feat = feat.float()

    cache_obj = {
        "feat": feat.contiguous(),
        "file_name": image_path.name,
        "image_path": str(image_path),
        "sam_model_type": str(sam_model_type),
        "sam_img_size": 1024,
        "orig_size": (int(h), int(w)),
        "input_size": tuple(int(x) for x in predictor.input_size),
    }
    ensure_dir(out_path.parent)
    torch.save(cache_obj, out_path)
    try:
        predictor.reset_image()
    except Exception:
        pass
    return "save"

# =========================================================
# RPINE cache config
# =========================================================
SAM_CHECKPOINT = r"sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
CACHE_ROOT = r"cache_sam_vitb_1024/RPINE"

# RPINE is cached split-by-split because the training scripts expect:
#   CACHE_ROOT/train/*.pt
#   CACHE_ROOT/test/*.pt
SPLIT_IMAGE_DIRS = {
    "train": r"RPINE/train/images",
    "test": r"RPINE/test/images",
}


def run_rpine_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("SAM_MODEL_TYPE:", SAM_MODEL_TYPE, flush=True)
    print("SAM_CHECKPOINT:", SAM_CHECKPOINT, flush=True)
    print("CACHE_ROOT:", CACHE_ROOT, flush=True)

    predictor = build_predictor(device, SAM_MODEL_TYPE, SAM_CHECKPOINT)

    total_save = 0
    total_skip = 0
    total_fail = 0
    t0 = time.time()

    for split, image_dir in SPLIT_IMAGE_DIRS.items():
        image_paths = list_images(image_dir)
        out_dir = Path(CACHE_ROOT) / split
        ensure_dir(out_dir)
        print(f"\n[RPINE] split={split} image_dir={image_dir}", flush=True)
        print(f"  images: {len(image_paths)}", flush=True)
        print(f"  out_dir: {out_dir}", flush=True)

        for i, img_path in enumerate(image_paths, start=1):
            out_path = out_dir / f"{img_path.stem}.pt"
            try:
                status = save_one_feature(predictor, img_path, out_path, SAM_MODEL_TYPE)
                if status == "skip":
                    total_skip += 1
                else:
                    total_save += 1
            except Exception as e:
                total_fail += 1
                print(f"[FAIL] {img_path} -> {e}", flush=True)
                traceback.print_exc()

            if i == 1 or i % BATCH_PRINT_EVERY == 0 or i == len(image_paths):
                print(
                    f"[{split}] {i}/{len(image_paths)} saved={total_save} skipped={total_skip} failed={total_fail}",
                    flush=True,
                )

    print("\n========== Cache done ==========")
    print(f"saved  = {total_save}")
    print(f"skipped= {total_skip}")
    print(f"failed = {total_fail}")
    print(f"time   = {time.time() - t0:.1f} sec")


if __name__ == "__main__":
    run_rpine_cache()
