
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
# FSCD-LVIS cache config
# =========================================================
SAM_CHECKPOINT = r"sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"
FSCD_LVIS_ROOT = r"FSCD_LVIS/FSCD_LVIS"

# Save all seen/unseen train/val/test image features into one folder.
# The 64+1 FSCD-LVIS training scripts search this directory through
# extra_sam_feature_cache_roots.
CACHE_DIR = r"cache_sam_vitb_1024/FSCD_LVIS/images"

ANNOTATION_FILES = [
    "instances_train.json",
    "instances_val.json",
    "instances_test.json",
    "unseen_instances_train.json",
    "unseen_instances_val.json",
    "unseen_instances_test.json",
]


def build_image_search_dirs(root):
    root = Path(root)
    candidates = [
        root,
        root / "images",
        root / "Images",
        root / "JPEGImages",
        root / "train2017",
        root / "val2017",
        root / "test2017",
        root / "FSC147",
        root / "images_384_VarV2",
    ]
    out = []
    for d in candidates:
        if d.is_dir() and d not in out:
            out.append(d)
    return out


def find_lvis_image(file_name, search_dirs):
    # 1) file_name as relative path
    for base in search_dirs:
        p = Path(base) / file_name
        if p.is_file():
            return p

    # 2) basename in common image folders
    base_name = Path(file_name).name
    for base in search_dirs:
        p = Path(base) / base_name
        if p.is_file():
            return p

    # 3) stem + common extensions
    stem = Path(file_name).stem
    for base in search_dirs:
        p = find_image_file(base, stem)
        if p is not None:
            return p
    return None


def collect_lvis_images(root):
    root = Path(root)
    ann_dir = root / "annotations"
    search_dirs = build_image_search_dirs(root)
    print("[FSCD-LVIS] image search dirs:", flush=True)
    for d in search_dirs:
        print(f"  - {d}", flush=True)

    records = {}
    missing = 0

    for ann_name in ANNOTATION_FILES:
        ann_path = ann_dir / ann_name
        if not ann_path.is_file():
            print(f"[Skip missing annotation] {ann_path}", flush=True)
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", []) if isinstance(data, dict) else []
        print(f"[Annotation] {ann_name} images={len(images)}", flush=True)

        for img in images:
            file_name = img.get("file_name", None)
            if not file_name:
                continue
            img_path = find_lvis_image(file_name, search_dirs)
            if img_path is None:
                missing += 1
                continue
            key = Path(img_path).stem
            if key not in records:
                records[key] = {
                    "image_path": img_path,
                    "file_name": Path(img_path).name,
                    "source_file_name": file_name,
                }

    print(f"[FSCD-LVIS] unique images found={len(records)} missing={missing}", flush=True)
    return records


def run_fscdlvis_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("SAM_MODEL_TYPE:", SAM_MODEL_TYPE, flush=True)
    print("SAM_CHECKPOINT:", SAM_CHECKPOINT, flush=True)
    print("FSCD_LVIS_ROOT:", FSCD_LVIS_ROOT, flush=True)
    print("CACHE_DIR:", CACHE_DIR, flush=True)

    records = collect_lvis_images(FSCD_LVIS_ROOT)
    ensure_dir(CACHE_DIR)

    predictor = build_predictor(device, SAM_MODEL_TYPE, SAM_CHECKPOINT)

    total_save = 0
    total_skip = 0
    total_fail = 0
    t0 = time.time()
    items = sorted(records.items(), key=lambda kv: kv[0].lower())

    manifest_path = Path(CACHE_DIR) / "cache_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for i, (stem, rec) in enumerate(items, start=1):
            img_path = Path(rec["image_path"])
            out_path = Path(CACHE_DIR) / f"{stem}.pt"
            try:
                status = save_one_feature(predictor, img_path, out_path, SAM_MODEL_TYPE)
                if status == "skip":
                    total_skip += 1
                else:
                    total_save += 1
                mf.write(json.dumps({
                    "stem": stem,
                    "file_name": rec["file_name"],
                    "source_file_name": rec["source_file_name"],
                    "image_path": str(img_path),
                    "cache_path": str(out_path),
                    "status": status,
                }, ensure_ascii=False) + "\n")
            except Exception as e:
                total_fail += 1
                print(f"[FAIL] {img_path} -> {e}", flush=True)
                traceback.print_exc()

            if i == 1 or i % BATCH_PRINT_EVERY == 0 or i == len(items):
                print(
                    f"[FSCD-LVIS] {i}/{len(items)} saved={total_save} skipped={total_skip} failed={total_fail}",
                    flush=True,
                )

    print("\n========== Cache done ==========")
    print(f"saved   = {total_save}")
    print(f"skipped = {total_skip}")
    print(f"failed  = {total_fail}")
    print(f"manifest= {manifest_path}")
    print(f"time    = {time.time() - t0:.1f} sec")


if __name__ == "__main__":
    run_fscdlvis_cache()
