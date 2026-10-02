# -*- coding: utf-8 -*-
"""
Precompute frozen SAM ViT-H image encoder features for FSCD-147 cache-only TCA training.

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

Expected cache structure for the 64+1 FSCD-147 training scripts:
    cache_sam_vith_1024/FSCD_147/train/*.pt
    cache_sam_vith_1024/FSCD_147/val/*.pt
    cache_sam_vith_1024/FSCD_147/test/*.pt
"""
import os

CUDA_VISIBLE_DEVICES = r"0"  # change to "0", "1", "2", ...
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

# ---------------------------------------------------------
# FSCD-147 cache config: SAM ViT-H
# ---------------------------------------------------------
SAM_CHECKPOINT = r"sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_IMG_SIZE = 1024

FSCD_ROOT = r"FSCD_147/FSC147"
IMAGE_DIR = r"FSCD_147/FSC147/images_384_VarV2"
ANNOTATION_DIR = r"FSCD_147/FSC147/annotations"
SPLIT_FILE = r"FSCD_147/FSC147/annotations/Train_Test_Val_FSC_147.json"

CACHE_ROOT = r"cache_sam_vith_1024/FSCD_147"
SPLITS = ["train", "val", "test"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def pil_read_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def find_image_file(image_dir, file_or_stem):
    image_dir = Path(image_dir)

    # 1) Exact relative path or exact file name.
    p = image_dir / str(file_or_stem)
    if p.is_file():
        return p

    # 2) Basename under image_dir.
    p = image_dir / Path(str(file_or_stem)).name
    if p.is_file():
        return p

    # 3) Stem + common image extensions.
    stem = Path(str(file_or_stem)).stem
    for ext in IMAGE_EXTS:
        for e in (ext, ext.upper()):
            p = image_dir / f"{stem}{e}"
            if p.is_file():
                return p

    return None


def normalize_split_item(item):
    """Accept a split item as a string or a dict with file_name/name/image fields."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ["file_name", "name", "image", "img", "filename"]:
            if key in item and item[key]:
                return str(item[key])
    return None


def load_split_names_from_split_file(split):
    split_path = Path(SPLIT_FILE)
    if not split_path.is_file():
        return []

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    raw = None
    for key in [split, split.lower(), split.upper(), split.capitalize()]:
        if isinstance(split_data, dict) and key in split_data:
            raw = split_data[key]
            break

    if raw is None:
        return []

    names = []
    for item in raw:
        name = normalize_split_item(item)
        if name:
            names.append(name)
    return names


def load_split_names_from_instances(split):
    inst_path = Path(ANNOTATION_DIR) / f"instances_{split}.json"
    if not inst_path.is_file():
        return []

    with open(inst_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    names = []
    for img in data.get("images", []):
        name = img.get("file_name", None)
        if name:
            names.append(str(name))
    return names


def collect_split_images(split):
    # Prefer the official FSCD-147 split file because the training script iterates it.
    names = load_split_names_from_split_file(split)

    # Fallback to COCO instances file if the split file is unavailable or uses a different key.
    if len(names) == 0:
        names = load_split_names_from_instances(split)

    records = {}
    missing = 0
    for name in names:
        img_path = find_image_file(IMAGE_DIR, name)
        if img_path is None:
            missing += 1
            continue
        key = img_path.stem
        if key not in records:
            records[key] = {
                "image_path": img_path,
                "file_name": img_path.name,
                "source_file_name": name,
            }

    print(f"[FSCD-147] split={split} split_names={len(names)} found={len(records)} missing={missing}", flush=True)
    return records


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
        feat = feat.half() if SAVE_FP16 else feat.float()

    cache_obj = {
        "feat": feat.contiguous(),
        "file_name": image_path.name,
        "image_path": str(image_path),
        "sam_model_type": str(sam_model_type),
        "sam_img_size": int(SAM_IMG_SIZE),
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


def run_fscd147_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device, flush=True)
    print("CUDA_VISIBLE_DEVICES:", CUDA_VISIBLE_DEVICES, flush=True)
    print("SAM_MODEL_TYPE:", SAM_MODEL_TYPE, flush=True)
    print("SAM_CHECKPOINT:", SAM_CHECKPOINT, flush=True)
    print("FSCD_ROOT:", FSCD_ROOT, flush=True)
    print("IMAGE_DIR:", IMAGE_DIR, flush=True)
    print("SPLIT_FILE:", SPLIT_FILE, flush=True)
    print("CACHE_ROOT:", CACHE_ROOT, flush=True)

    predictor = build_predictor(device, SAM_MODEL_TYPE, SAM_CHECKPOINT)

    total_save = 0
    total_skip = 0
    total_fail = 0
    t0 = time.time()

    for split in SPLITS:
        records = collect_split_images(split)
        out_dir = Path(CACHE_ROOT) / split
        ensure_dir(out_dir)
        manifest_path = out_dir / "cache_manifest.jsonl"

        print(f"\n[FSCD-147] split={split}", flush=True)
        print(f"  out_dir: {out_dir}", flush=True)
        print(f"  images : {len(records)}", flush=True)

        items = sorted(records.items(), key=lambda kv: kv[0].lower())
        split_save = 0
        split_skip = 0
        split_fail = 0

        with open(manifest_path, "w", encoding="utf-8") as mf:
            for i, (stem, rec) in enumerate(items, start=1):
                img_path = Path(rec["image_path"])
                out_path = out_dir / f"{stem}.pt"

                try:
                    status = save_one_feature(predictor, img_path, out_path, SAM_MODEL_TYPE)
                    if status == "skip":
                        total_skip += 1
                        split_skip += 1
                    else:
                        total_save += 1
                        split_save += 1

                    mf.write(json.dumps({
                        "split": split,
                        "stem": stem,
                        "file_name": rec["file_name"],
                        "source_file_name": rec["source_file_name"],
                        "image_path": str(img_path),
                        "cache_path": str(out_path),
                        "status": status,
                    }, ensure_ascii=False) + "\n")

                except Exception as e:
                    total_fail += 1
                    split_fail += 1
                    print(f"[FAIL] split={split} {img_path} -> {e}", flush=True)
                    traceback.print_exc()

                if i == 1 or i % BATCH_PRINT_EVERY == 0 or i == len(items):
                    print(
                        f"[{split}] {i}/{len(items)} "
                        f"saved={split_save} skipped={split_skip} failed={split_fail} "
                        f"| total_saved={total_save} total_skipped={total_skip} total_failed={total_fail}",
                        flush=True,
                    )

        print(f"  manifest: {manifest_path}", flush=True)

    print("\n========== Cache done ==========")
    print(f"saved  = {total_save}")
    print(f"skipped= {total_skip}")
    print(f"failed = {total_fail}")
    print(f"time   = {time.time() - t0:.1f} sec")


if __name__ == "__main__":
    run_fscd147_cache()
