import os

# =========================================================
# Inference controls: edit here
# =========================================================
CUDA_VISIBLE_DEVICES = r"0"   # e.g. "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

CHECKPOINT_PATH = r"model/fscd147_vith/best_model.pth"
OUTPUT_DIR = r"infer_outputs/fscd147_vith_64plus1"

# Evaluation protocol
USE_THREE_SHOT = True          # True = 3-shot, False = 1-shot
NUM_EVAL_SHOTS = 3             # used only when USE_THREE_SHOT=True

# Post-processing
SCORE_THRESH = 0.2
NMS_IOU_THRESH = 0.4
TOPK_PER_IMAGE_INFER = 4500
MAX_DETS_PER_IMAGE_INFER = 4500
AP_MAX_DETS_PER_IMAGE_INFER = 4500

# Output controls
SAVE_METRICS_JSON = True
SAVE_PREDICTIONS_JSONL = True
SAVE_VISUALS = False
NUM_VIS_IMAGES_INFER = 20

# Checkpoint loading. Keep True for paper-aligned evaluation.
STRICT_LOAD = True

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import json
import time
import random
import contextlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import roi_align
from torchmetrics.detection import MeanAveragePrecision
from pycocotools.coco import COCO

# =========================================================
# 配置区：直接改这里
# =========================================================
# =========================================================
# FSCD-147 配置
# 目录结构要求：
#   FSCD147/
#       annotations/
#           annotation_FSC147_384.json
#           Train_Test_Val_FSC_147.json
#           instances_train.json
#           instances_val.json
#           instances_test.json
#       images_384_VarV2/
#           *.jpg
#
# instances_*.json 是 COCO 格式：
#   images: [{"id", "file_name", "width", "height"}, ...]
#   annotations: [{"image_id", "bbox": [x,y,w,h], ...}, ...]
#
# 本脚本：train/test/test 全部使用 instances_{split}.json 的 bbox 框监督。
# annotation_FSC147_384.json 只用于读取 box_examples_coordinates 作为模板框。
# =========================================================
fscd_root = r"FSCD_147/FSC147"
train_split = "train"
eval_split = "test"   # inference uses official test split

sam_checkpoint = r"sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
save_dir = r"model/fscd147_vith"

# 预提取 SAM encoder 特征缓存目录。
# 目录结构：sam_feature_cache_root/train/*.pt, test/*.pt, test/*.pt
sam_feature_cache_root = r"cache_sam_vith_1024/FSCD_147"
cached_feature_to_float32 = False

# =========================================================
# Speed options for cached-feature training
# =========================================================
# Mixed precision only works safely because every model forward below is wrapped by amp_autocast().
use_amp = True
amp_dtype = "float16"   # "float16" or "bfloat16"; RTX 4090/5090 用 float16 通常最快
allow_tf32 = True

# Cached mode: do not convert cache to float32 on CPU. Keep fp16 cache as fp16.
# If your cache is float32, autocast will still handle model forward.
skip_cached_image_resize = True
# =========================================================
tca_version = "tca_v1_no_attention_ablation_officialval"

# =========================================================
# Ablation note: no attention module
# =========================================================
# This version removes TemplateContextAttentionGate from TDCCDetector.
# The prediction head only receives [feat_hr, sim_center].
# There is no attention map, no sim_refined channel, and lambda_attn = 0.
# TCA multi-branch aggregation, template center matching, regularization,
# AMP, and cache-only data loading are kept unchanged.
# =========================================================

# 自动断点续训：如果 save_dir/last_model.pth 存在，就从下一轮 epoch 继续
auto_resume = True  # no-attention ablation: train from scratch
# 留空表示默认读取 os.path.join(save_dir, "last_model.pth")
resume_checkpoint_path = r""

sam_img_size = 1024
pred_size = 256
project_dim = 128
decoder_mid_dim = 96
decoder_out_dim = 64
template_roi_size = 5
num_context_branches = 6

epochs = 100
batch_size = 4
# 评估 batch，建议 4 或 8；避免 batch=1 连续小 kernel 调用太多
eval_batch_size = 16
num_workers = 0
lr = 2e-4
weight_decay = 3e-4
grad_clip_norm = 5.0

lambda_center = 1.0
lambda_giou = 2.0
lambda_size = 1.0
lambda_attn = 0.0  # no-attention ablation: disable attention auxiliary loss
lambda_offset_cls = 0.5

gaussian_min_radius = 1
gaussian_radius_ratio = 0.15
attention_radius_ratio = 0.30

# =========================================================
# Cache-mode regularization
# =========================================================
# cache 特征训练无法做常规图像增强，这里使用轻量正则化：
# 1) template box jitter：训练时轻微扰动模板框；
# 2) feature dropout：投影后特征做轻量 Dropout2d；
# 3) branch dropout：多分支聚合时随机屏蔽少量分支。
template_jitter_center = 0.05
template_jitter_scale = 0.10
feature_dropout_p = 0.05
branch_dropout_p = 0.10
# =========================================================

# AP 评估时为了得到完整 PR 曲线，decode 阈值应低一些
ap_score_thresh = 0.2
# AP 评估每图最多预测数，避免评估阶段候选过多
ap_max_dets_per_image = 4500

# eval 内部追踪设置
eval_trace_every = 10
eval_empty_cache_every = 20

score_thresh = 0.2
topk_per_image = 4500
max_dets_per_image = 4500
nms_iou_thresh = 0.4

eval_interval = 1
vis_interval = 100000000  # disable training-time visuals for speed
num_vis_images = 20
seed = 42
debug_anomaly = False
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def pil_read_rgb(image_path):
    """
    PIL 读取 RGB，避免 cv2 在 Windows 下偶发 native crash。
    """
    img = Image.open(image_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)



def pil_image_size(image_path):
    """
    只读取图片尺寸，不把整张图转成 numpy。
    cache 训练只需要 h/w 来做坐标映射，不需要 resize 原图。
    """
    with Image.open(image_path) as img:
        w, h = img.size
    return int(w), int(h)


def make_sam_meta_from_hw(h, w, target_size=1024):
    """
    SAM ResizeLongestSide 的坐标 meta。
    不生成 padded image，只保留坐标换算所需字段。
    """
    scale = target_size / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return {
        "orig_h": int(h),
        "orig_w": int(w),
        "scale": float(scale),
        "resized_h": int(new_h),
        "resized_w": int(new_w),
        "target_size": int(target_size),
    }


def get_amp_dtype():
    if str(amp_dtype).lower() in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float16


def amp_autocast(device):
    if not (bool(use_amp) and device.type == "cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=get_amp_dtype(), enabled=True)


def make_grad_scaler(device):
    enabled = bool(use_amp) and device.type == "cuda" and get_amp_dtype() == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def pil_resize_rgb(image_rgb, new_w, new_h):
    """
    PIL resize，输入/输出都是 numpy RGB uint8。
    """
    img = Image.fromarray(image_rgb)
    img = img.resize((int(new_w), int(new_h)), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)



def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_image_file(image_dir, file_or_stem):
    image_dir = Path(image_dir)
    p = image_dir / file_or_stem
    if p.is_file():
        return str(p), p.name
    stem = Path(file_or_stem).stem
    for ext in image_exts:
        for e in (ext, ext.upper()):
            p = image_dir / f"{stem}{e}"
            if p.is_file():
                return str(p), p.name
    return None, None


def clamp_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(w - 1)))
    y1 = max(0.0, min(float(y1), float(h - 1)))
    x2 = max(0.0, min(float(x2), float(w - 1)))
    y2 = max(0.0, min(float(y2), float(h - 1)))
    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)
    return [x1, y1, x2, y2]


def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)



def jitter_template_box(box, sam_size=1024, center_jitter=0.05, scale_jitter=0.10):
    """
    Training-only template jitter in SAM padded coordinate space.
    GT boxes remain unchanged; only the support/template ROI is slightly shifted/scaled.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(2.0, x2 - x1)
    h = max(2.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    dx = random.uniform(-center_jitter, center_jitter) * w
    dy = random.uniform(-center_jitter, center_jitter) * h
    sw = math.exp(random.uniform(-scale_jitter, scale_jitter))
    sh = math.exp(random.uniform(-scale_jitter, scale_jitter))

    nw = max(2.0, w * sw)
    nh = max(2.0, h * sh)
    ncx = cx + dx
    ncy = cy + dy

    nx1 = max(0.0, min(float(sam_size - 1), ncx - 0.5 * nw))
    ny1 = max(0.0, min(float(sam_size - 1), ncy - 0.5 * nh))
    nx2 = max(nx1 + 1.0, min(float(sam_size - 1), ncx + 0.5 * nw))
    ny2 = max(ny1 + 1.0, min(float(sam_size - 1), ncy + 0.5 * nh))

    return [nx1, ny1, nx2, ny2]

def resize_longest_side_and_pad_rgb(image_rgb, target_size=1024):
    h, w = image_rgb.shape[:2]
    scale = target_size / float(max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = pil_resize_rgb(image_rgb, new_w, new_h)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    meta = {"orig_h": h, "orig_w": w, "scale": scale, "resized_h": new_h, "resized_w": new_w, "target_size": target_size}
    return padded, meta


def boxes_original_to_sam_padded(boxes, meta, target_size=1024):
    out = []
    scale = meta["scale"]
    for box in boxes:
        x1, y1, x2, y2 = box
        nb = clamp_box_xyxy([x1 * scale, y1 * scale, x2 * scale, y2 * scale], target_size, target_size)
        if box_area_xyxy(nb) >= 4:
            out.append(nb)
    return out


def boxes_sam_padded_to_original(boxes, meta):
    out = []
    scale = meta["scale"]
    for box in boxes:
        x1, y1, x2, y2 = box
        out.append(clamp_box_xyxy([x1 / scale, y1 / scale, x2 / scale, y2 / scale], meta["orig_w"], meta["orig_h"]))
    return out


def read_txt_boxes_xyxy(txt_path, image_w, image_h):
    boxes = []
    if not os.path.isfile(txt_path):
        return boxes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.replace(",", " ").split()
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in parts]
            except Exception:
                continue
            box = clamp_box_xyxy([x1, y1, x2, y2], image_w, image_h)
            if box_area_xyxy(box) >= 4:
                boxes.append(box)
    return boxes


def load_exemplar_records(exemplar_json_path, image_dir):
    with open(exemplar_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = {}

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        anns_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)
        for img in data.get("images", []):
            file_name = img["file_name"]
            image_path, real_name = find_image_file(image_dir, file_name)
            if image_path is None:
                continue
            boxes = []
            for ann in anns_by_image.get(int(img["id"]), []):
                bbox = ann.get("bbox")
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, bw, bh = [float(v) for v in bbox]
                boxes.append([x, y, x + bw, y + bh])
            if boxes:
                records[Path(real_name).stem] = {"file_name": real_name, "image_path": image_path, "template_boxes": boxes}
        return records

    if isinstance(data, dict):
        for key, value in data.items():
            file_name = key
            boxes_raw = None
            if isinstance(value, dict):
                file_name = value.get("file_name", key)
                boxes_raw = value.get("boxes", value.get("bboxes", None))
            elif isinstance(value, list):
                boxes_raw = value
            if not boxes_raw:
                continue
            image_path, real_name = find_image_file(image_dir, file_name)
            if image_path is None:
                image_path, real_name = find_image_file(image_dir, key)
            if image_path is None:
                continue
            boxes = []
            for b in boxes_raw:
                if isinstance(b, dict):
                    b = b.get("bbox")
                if b is None or len(b) != 4:
                    continue
                boxes.append([float(v) for v in b])
            if boxes:
                records[Path(real_name).stem] = {"file_name": real_name, "image_path": image_path, "template_boxes": boxes}
    return records


class FSCD147InstancesDataset(Dataset):
    """
    FSCD-147 全框监督 Dataset。

    读取：
        annotations/instances_{split}.json
            COCO 格式，bbox = [x, y, w, h]
        annotations/annotation_FSC147_384.json
            box_examples_coordinates，作为 template/exemplar boxes
        annotations/Train_Test_Val_FSC_147.json
            train/test/test split 文件名列表

    返回字段兼容原训练脚本：
        image
        image_id
        file_name
        template_box
        template_candidates
        gt_boxes
        meta
    """
    def __init__(self, root, split="train", training=True, max_exemplars=3):
        self.root = root
        self.split = split
        self.training = training
        self.max_exemplars = max_exemplars

        self.image_dir = os.path.join(root, "images_384_VarV2")
        self.anno_file = os.path.join(root, "annotations", "annotation_FSC147_384.json")
        self.split_file = os.path.join(root, "annotations", "Train_Test_Val_FSC_147.json")
        self.instance_file = os.path.join(root, "annotations", f"instances_{split}.json")
        self.cache_dir = os.path.join(sam_feature_cache_root, split)

        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(
                f"Feature cache dir not found: {self.cache_dir}. "
                f"Expected files like {self.cache_dir}/<image_stem>.pt"
            )

        if not os.path.isfile(self.anno_file):
            raise FileNotFoundError(self.anno_file)
        if not os.path.isfile(self.split_file):
            raise FileNotFoundError(self.split_file)
        if not os.path.isfile(self.instance_file):
            raise FileNotFoundError(self.instance_file)

        with open(self.anno_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)
        with open(self.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        self.split_names = list(split_data[split])

        self.coco = COCO(self.instance_file)
        self.img_name_to_coco_id = {}
        for _, v in self.coco.imgs.items():
            self.img_name_to_coco_id[v["file_name"]] = int(v["id"])

        valid = []
        miss_img = miss_fsc_anno = miss_coco_img = miss_exemplar = miss_box = 0

        for name in self.split_names:
            if name not in self.annotations:
                miss_fsc_anno += 1
                continue

            img_path, real_name = find_image_file(self.image_dir, name)
            if img_path is None:
                miss_img += 1
                continue

            if real_name not in self.img_name_to_coco_id:
                miss_coco_img += 1
                continue

            exemplars = self.get_exemplars_from_annotation(real_name)
            if len(exemplars) == 0:
                miss_exemplar += 1
                continue

            try:
                with Image.open(img_path) as im:
                    image_w, image_h = im.size
            except Exception:
                miss_img += 1
                continue

            boxes = self.get_bboxes_from_coco(real_name, image_w, image_h)
            if len(boxes) == 0:
                miss_box += 1
                continue

            cache_path = os.path.join(self.cache_dir, f"{Path(real_name).stem}.pt")
            if not os.path.isfile(cache_path):
                miss_img += 1
                continue

            valid.append(real_name)

        self.keys = valid

        print(f"[FSCD147InstancesDataset]")
        print(f"  root: {root}")
        print(f"  split: {split}, training={training}")
        print(f"  image_dir: {self.image_dir}")
        print(f"  fsc annotation: {self.anno_file}")
        print(f"  split file: {self.split_file}")
        print(f"  instances: {self.instance_file}")
        print(f"  cache_dir: {self.cache_dir}")
        print(f"  split names: {len(self.split_names)}")
        print(f"  valid images: {len(self.keys)}")
        print(f"  missing image: {miss_img}")
        print(f"  missing annotation_FSC147 item: {miss_fsc_anno}")
        print(f"  missing COCO image entry: {miss_coco_img}")
        print(f"  missing exemplar boxes: {miss_exemplar}")
        print(f"  missing bbox: {miss_box}")

    def __len__(self):
        return len(self.keys)

    def exemplar_polygon_to_xyxy(self, poly):
        """
        FSC147 的 exemplar 一般是四个角点：
            [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
        这里统一 min/max 成 xyxy。
        """
        arr = np.asarray(poly, dtype=np.float32)
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def get_exemplars_from_annotation(self, img_name):
        item = self.annotations[img_name]
        boxes_raw = item.get("box_examples_coordinates", [])[:self.max_exemplars]
        boxes = []
        for poly in boxes_raw:
            if poly is None or len(poly) < 2:
                continue
            boxes.append(self.exemplar_polygon_to_xyxy(poly))
        return boxes

    def get_bboxes_from_coco(self, img_name, image_w, image_h):
        """
        从 COCO instances json 读取当前图的所有框。
        COCO bbox: [x, y, w, h]
        转成 xyxy: [x1, y1, x2, y2]
        """
        if img_name not in self.img_name_to_coco_id:
            return []

        img_id = self.img_name_to_coco_id[img_name]
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annos = self.coco.loadAnns(ann_ids)

        boxes = []
        for ann in annos:
            if ann.get("iscrowd", 0) == 1:
                continue
            bbox = ann.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x, y, bw, bh = [float(v) for v in bbox]
            if bw <= 0 or bh <= 0:
                continue
            box = clamp_box_xyxy([x, y, x + bw, y + bh], image_w, image_h)
            if box_area_xyxy(box) >= 4:
                boxes.append(box)
        return boxes

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        img_path, real_name = find_image_file(self.image_dir, img_name)
        if img_path is None:
            raise FileNotFoundError(img_name)

        w, h = pil_image_size(img_path)

        gt_orig = self.get_bboxes_from_coco(real_name, w, h)
        tpl_orig = [clamp_box_xyxy(b, w, h) for b in self.get_exemplars_from_annotation(real_name)]

        gt_orig = [b for b in gt_orig if box_area_xyxy(b) >= 4]
        tpl_orig = [b for b in tpl_orig if box_area_xyxy(b) >= 4]

        if len(gt_orig) == 0:
            gt_orig = [[0, 0, 10, 10]]
        if len(tpl_orig) == 0:
            tpl_orig = gt_orig

        meta = make_sam_meta_from_hw(h, w, target_size=sam_img_size)
        padded = None
        gt_sam = boxes_original_to_sam_padded(gt_orig, meta, target_size=sam_img_size)
        tpl_sam = boxes_original_to_sam_padded(tpl_orig, meta, target_size=sam_img_size)

        if len(gt_sam) == 0:
            gt_sam = [[0, 0, 10, 10]]
        if len(tpl_sam) == 0:
            tpl_sam = gt_sam

        template_box = random.choice(tpl_sam) if self.training else tpl_sam[0]
        if self.training:
            template_box = jitter_template_box(
                template_box,
                sam_size=sam_img_size,
                center_jitter=template_jitter_center,
                scale_jitter=template_jitter_scale,
            )

        cache_path = os.path.join(self.cache_dir, f"{Path(real_name).stem}.pt")
        cache_obj = torch.load(cache_path, map_location="cpu")
        sam_feat = cache_obj["feat"] if isinstance(cache_obj, dict) and "feat" in cache_obj else cache_obj

        return {
            "image": padded,
            "image_id": Path(real_name).stem,
            "file_name": real_name,
            "template_box": template_box,
            "template_candidates": tpl_sam,
            "gt_boxes": gt_sam,
            "meta": meta,
            "sam_feat": sam_feat,
        }


def collate_fn(batch):
    return batch


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        g = min(8, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def crop_feature_roi_to_fixed(feat, boxes_sam, out_size=7, sam_size=1024):
    """
    真正 RoIAlign 版模板 ROI 采样。
    模板框整体 -> 7×7 token，用于生成空间分支权重和通道门控。
    """
    B, C, H, W = feat.shape
    device = feat.device
    dtype = feat.dtype

    rois = []
    for i, box in enumerate(boxes_sam):
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(float(x1), sam_size - 1.0))
        y1 = max(0.0, min(float(y1), sam_size - 1.0))
        x2 = max(x1 + 1.0, min(float(x2), sam_size - 1.0))
        y2 = max(y1 + 1.0, min(float(y2), sam_size - 1.0))
        rois.append([i, x1, y1, x2, y2])

    rois = torch.tensor(rois, device=device, dtype=dtype)
    spatial_scale = W / float(sam_size)

    roi_feat = roi_align(
        input=feat,
        boxes=rois,
        output_size=(out_size, out_size),
        spatial_scale=spatial_scale,
        sampling_ratio=2,
        aligned=True,
    )

    return roi_feat


class TemplateDynamicContextAggregation(nn.Module):
    """
    模板条件动态上下文聚合：6 组多尺度/多形状分支 + 通道自适应门控。

    分支顺序：
        0 identity
        1 3×3
        2 5×5
        3 7×7
        4 cross5 = 0.5 × (1×5 + 5×1)
        5 cross7 = 0.5 × (1×7 + 7×1)
    """
    def __init__(self, in_ch=256, out_ch=128, roi_size=7, num_branches=6):
        super().__init__()
        self.out_ch = out_ch
        self.roi_size = roi_size
        self.num_branches = num_branches
        self.feat_drop = nn.Dropout2d(p=float(feature_dropout_p))

        self.proj = ConvGNAct(in_ch, out_ch, k=1, s=1, p=0)
        self.branch_identity = nn.Identity()

        self.branch_dw3 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.branch_dw5 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5, padding=2, groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.branch_dw7 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=3, groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

        self.branch_h1x5 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 5), padding=(0, 2), groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.branch_v5x1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(5, 1), padding=(2, 0), groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.branch_h1x7 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 7), padding=(0, 3), groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.branch_v7x1 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 1), padding=(3, 0), groups=out_ch, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

        self.visual_mlp = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, num_branches),
        )
        self.size_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, num_branches),
        )
        self.size_embed_for_channel = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )
        self.channel_gate_mlp = nn.Sequential(
            nn.Linear(out_ch * 2 + 32, out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )
        self.refine = ConvGNAct(out_ch, out_ch, k=1, s=1, p=0)

        nn.init.zeros_(self.visual_mlp[-1].weight)
        nn.init.zeros_(self.visual_mlp[-1].bias)
        nn.init.zeros_(self.size_mlp[-1].weight)
        nn.init.zeros_(self.size_mlp[-1].bias)
        nn.init.zeros_(self.channel_gate_mlp[-1].weight)
        nn.init.zeros_(self.channel_gate_mlp[-1].bias)

    def build_size_feat(self, boxes, device, dtype):
        vals = []
        for x1, y1, x2, y2 in boxes:
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            vals.append([
                math.log(w / sam_img_size + 1e-6),
                math.log(h / sam_img_size + 1e-6),
                math.log((w * h) / (sam_img_size * sam_img_size) + 1e-6),
                math.log(w / h + 1e-6),
            ])
        return torch.tensor(vals, device=device, dtype=dtype)

    def forward(self, sam_feat, template_boxes_sam):
        feat = self.proj(sam_feat)
        if self.training and feature_dropout_p > 0:
            feat = self.feat_drop(feat)
        roi = crop_feature_roi_to_fixed(feat, template_boxes_sam, out_size=self.roi_size, sam_size=sam_img_size)

        c = self.roi_size // 2
        center_feat = roi[:, :, c-1:c+2, c-1:c+2].mean(dim=(-2, -1))
        context_feat = roi.mean(dim=(-2, -1))
        size_feat = self.build_size_feat(template_boxes_sam, feat.device, feat.dtype)

        visual_logits = self.visual_mlp(torch.cat([center_feat, context_feat], dim=1))
        size_logits = self.size_mlp(size_feat)
        branch_weights = F.softmax(visual_logits + size_logits, dim=1)
        if self.training and branch_dropout_p > 0:
            keep = (torch.rand_like(branch_weights) > float(branch_dropout_p)).to(branch_weights.dtype)
            empty = keep.sum(dim=1, keepdim=True) < 1
            keep = torch.where(empty, torch.ones_like(keep), keep)
            branch_weights = branch_weights * keep
            branch_weights = branch_weights / branch_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        b0 = self.branch_identity(feat)
        b1 = self.branch_dw3(feat)
        b2 = self.branch_dw5(feat)
        b3 = self.branch_dw7(feat)
        b4 = 0.5 * (self.branch_h1x5(feat) + self.branch_v5x1(feat))
        b5 = 0.5 * (self.branch_h1x7(feat) + self.branch_v7x1(feat))

        branches = torch.stack([b0, b1, b2, b3, b4, b5], dim=1)
        out = (branches * branch_weights.view(feat.shape[0], self.num_branches, 1, 1, 1)).sum(dim=1)

        size_emb = self.size_embed_for_channel(size_feat)
        gate_input = torch.cat([center_feat, context_feat, size_emb], dim=1)
        channel_gate = torch.sigmoid(self.channel_gate_mlp(gate_input)).view(feat.shape[0], self.out_ch, 1, 1)
        out = out * (2.0 * channel_gate)
        out = self.channel_mix(out)
        out = self.refine(out)

        return out, {
            "branch_weights": branch_weights.detach(),
            "channel_gate_mean": channel_gate.detach().mean(),
            "channel_gate_std": channel_gate.detach().std(),
            "template_context_feat": context_feat,
        }


class HighResolutionDecoder(nn.Module):
    def __init__(self, in_ch=128, mid_ch=96, out_ch=64):
        super().__init__()
        self.pre = ConvGNAct(in_ch, in_ch, 3, 1, 1)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvGNAct(in_ch, mid_ch, 3, 1, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), ConvGNAct(mid_ch, out_ch, 3, 1, 1))
    def forward(self, x):
        return self.up2(self.up1(self.pre(x)))


def sample_template_center_query(feat_hr, template_boxes_sam):
    """
    RoIAlign 版中心 query。
    不聚合中心小区域，只取模板中心所在 token 对应的 1×1 RoIAlign 特征。
    """
    B, C, H, W = feat_hr.shape
    device = feat_hr.device
    dtype = feat_hr.dtype

    rois = []
    token_size = sam_img_size / float(pred_size)

    for i, box in enumerate(template_boxes_sam):
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        qx1 = cx - token_size * 0.5
        qy1 = cy - token_size * 0.5
        qx2 = cx + token_size * 0.5
        qy2 = cy + token_size * 0.5

        qx1 = max(0.0, min(qx1, sam_img_size - 1.0))
        qy1 = max(0.0, min(qy1, sam_img_size - 1.0))
        qx2 = max(qx1 + 1.0, min(qx2, sam_img_size - 1.0))
        qy2 = max(qy1 + 1.0, min(qy2, sam_img_size - 1.0))

        rois.append([i, qx1, qy1, qx2, qy2])

    rois = torch.tensor(rois, device=device, dtype=dtype)
    spatial_scale = W / float(sam_img_size)

    q_roi = roi_align(
        input=feat_hr,
        boxes=rois,
        output_size=(1, 1),
        spatial_scale=spatial_scale,
        sampling_ratio=2,
        aligned=True,
    )

    return q_roi[:, :, 0, 0]



class TemplateContextAttentionGate(nn.Module):
    """
    模板上下文注意力抑制模块。

    目标：
        - 用模板整体 context_feat 生成上下文 query；
        - 和全图 high-res local context 特征匹配，生成 attention logits/map；
        - 用更宽的 Gaussian attention target 做辅助监督；
        - 用 attention map 抑制“中心像但上下文不像”的假响应。
    """
    def __init__(self, template_dim=128, feat_dim=64):
        super().__init__()

        self.template_proj = nn.Sequential(
            nn.Linear(template_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.local_context = nn.Sequential(
            ConvGNAct(feat_dim, feat_dim, 3, 1, 1),
            ConvGNAct(feat_dim, feat_dim, 3, 1, 1),
        )

        # cosine ∈ [-1, 1]，scale 成更适合 focal 的 logit 范围
        self.logit_scale = nn.Parameter(torch.tensor(5.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, feat_hr, template_context_feat):
        local_feat = self.local_context(feat_hr)  # [B,64,H,W]
        q_context = self.template_proj(template_context_feat)  # [B,64]

        local_norm = F.normalize(local_feat, dim=1)
        q_norm = F.normalize(q_context, dim=1).view(q_context.shape[0], q_context.shape[1], 1, 1)

        context_sim = (local_norm * q_norm).sum(dim=1, keepdim=True)
        attn_logits = context_sim * self.logit_scale + self.logit_bias
        attn_map = torch.sigmoid(attn_logits)
        return attn_logits, attn_map, context_sim


class PredictionHead(nn.Module):
    def __init__(self, in_ch=67, hidden=64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(in_ch, hidden, 3, 1, 1),
            ConvGNAct(hidden, hidden, 3, 1, 1),
        )
        self.center_head = nn.Conv2d(hidden, 1, 1)
        self.offset_cls_head = nn.Conv2d(hidden, 4, 1)
        self.size_scale_head = nn.Conv2d(hidden, 2, 1)

        nn.init.constant_(self.center_head.bias, -4.0)
        nn.init.zeros_(self.size_scale_head.weight)
        nn.init.zeros_(self.size_scale_head.bias)

    def forward(self, x):
        f = self.stem(x)
        center_logits = self.center_head(f)
        offset_cls_logits = self.offset_cls_head(f)
        size_scale = torch.clamp(self.size_scale_head(f), min=-2.0, max=2.0)
        return center_logits, offset_cls_logits, size_scale


class TDCCDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Cached-feature 版本：不再加载 SAM 模型。
        # sample["sam_feat"] 是预提取的 frozen SAM encoder output: [256,64,64]
        self.dynamic_context = TemplateDynamicContextAggregation(256, project_dim, template_roi_size, num_context_branches)
        self.decoder = HighResolutionDecoder(project_dim, decoder_mid_dim, decoder_out_dim)
        # No-attention ablation: remove TemplateContextAttentionGate.
        # Head input = high-res feature + center-similarity map.
        self.head = PredictionHead(decoder_out_dim + 1, decoder_out_dim)

    def load_cached_sam_features(self, samples, device):
        feats = []
        for s in samples:
            feat = s["sam_feat"]
            if not torch.is_tensor(feat):
                feat = torch.as_tensor(feat)
            if cached_feature_to_float32:
                feat = feat.float()
            feats.append(feat.to(device, non_blocking=True).contiguous())
        return torch.stack(feats, dim=0)

    def forward(self, samples):
        device = next(self.parameters()).device
        template_boxes = [s["template_box"] for s in samples]
        sam_feat = self.load_cached_sam_features(samples, device)

        ctx, info = self.dynamic_context(sam_feat, template_boxes)
        feat_hr = self.decoder(ctx)

        q = sample_template_center_query(feat_hr, template_boxes)
        sim_center = (F.normalize(feat_hr, dim=1) * F.normalize(q, dim=1).view(q.shape[0], q.shape[1], 1, 1)).sum(dim=1, keepdim=True)

        # No-attention ablation:
        # do not compute context attention; do not multiply sim_center by attention map.
        head_in = torch.cat([feat_hr, sim_center], dim=1)
        center_logits, offset_cls_logits, size_scale = self.head(head_in)

        return {
            "center_logits": center_logits,
            "offset_cls_logits": offset_cls_logits,
            "size_scale": size_scale,
            "sim_map": sim_center,
            "branch_weights": info["branch_weights"],
            "channel_gate_mean": info.get("channel_gate_mean", None),
            "channel_gate_std": info.get("channel_gate_std", None),
        }

def numpy_nms(boxes, scores, iou_thresh=0.45, max_dets=300):
    """
    纯 numpy NMS，避免 torchvision.ops.nms 在 Windows 下 native crash。
    boxes: numpy [N,4], xyxy
    scores: numpy [N]
    """
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_dets:
            break

        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[rest] - inter + 1e-9
        iou = inter / union

        order = rest[iou <= iou_thresh]

    return np.asarray(keep, dtype=np.int64)

def get_offset_table(device=None, dtype=torch.float32):
    table = torch.tensor([
        [-0.25, -0.25],
        [ 0.25, -0.25],
        [-0.25,  0.25],
        [ 0.25,  0.25],
    ], dtype=dtype)
    if device is not None:
        table = table.to(device)
    return table


def template_wh_256_from_sample(sample):
    stride = sam_img_size / float(pred_size)
    x1, y1, x2, y2 = sample["template_box"]
    tw = max(1.0, (x2 - x1) / stride)
    th = max(1.0, (y2 - y1) / stride)
    return tw, th


def gaussian2d(radius, sigma=None):
    if sigma is None:
        sigma = radius / 3.0 if radius > 0 else 1.0
    d = 2 * radius + 1
    x = np.arange(0, d, 1, np.float32)
    y = x[:, None]
    return np.exp(-((x - radius) ** 2 + (y - radius) ** 2) / (2 * sigma ** 2 + 1e-12))


def draw_gaussian(heatmap, cx, cy, radius):
    H, W = heatmap.shape
    radius = int(radius)
    g = gaussian2d(radius)
    x, y = int(cx), int(cy)
    left, right = min(x, radius), min(W - x - 1, radius)
    top, bottom = min(y, radius), min(H - y - 1, radius)
    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return
    patch = heatmap[y-top:y+bottom+1, x-left:x+right+1]
    gp = g[radius-top:radius+bottom+1, radius-left:radius+right+1]
    if patch.shape == gp.shape:
        np.maximum(patch, gp, out=patch)


def build_targets(samples, device):
    B, H, W = len(samples), pred_size, pred_size
    center_t = np.zeros((B, 1, H, W), dtype=np.float32)
    attn_t = np.zeros((B, 1, H, W), dtype=np.float32)
    offset_cls_t = np.full((B, H, W), -1, dtype=np.int64)
    size_scale_t = np.zeros((B, 2, H, W), dtype=np.float32)
    template_wh_t = np.zeros((B, 2, H, W), dtype=np.float32)
    box_t = np.zeros((B, 4, H, W), dtype=np.float32)
    pos_mask = np.zeros((B, 1, H, W), dtype=np.float32)
    stride = sam_img_size / float(pred_size)
    all_gt_boxes_256 = []

    for b, s in enumerate(samples):
        boxes_256 = []
        tw, th = template_wh_256_from_sample(s)
        for x1, y1, x2, y2 in s["gt_boxes"]:
            x1h, y1h, x2h, y2h = x1 / stride, y1 / stride, x2 / stride, y2 / stride
            x1h, y1h = max(0, min(x1h, W - 1)), max(0, min(y1h, H - 1))
            x2h, y2h = max(0, min(x2h, W - 1)), max(0, min(y2h, H - 1))
            if x2h <= x1h or y2h <= y1h:
                continue
            cx, cy = 0.5 * (x1h + x2h), 0.5 * (y1h + y2h)
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue
            gw, gh = max(1e-3, x2h - x1h), max(1e-3, y2h - y1h)
            radius = max(gaussian_min_radius, int(round(min(gw, gh) * gaussian_radius_ratio)))
            attn_radius = max(gaussian_min_radius + 1, int(round(min(gw, gh) * attention_radius_ratio)))
            draw_gaussian(center_t[b, 0], ix, iy, radius)
            draw_gaussian(attn_t[b, 0], ix, iy, attn_radius)
            dx = cx - ix
            dy = cy - iy
            if dx < 0 and dy < 0:
                cls = 0
            elif dx >= 0 and dy < 0:
                cls = 1
            elif dx < 0 and dy >= 0:
                cls = 2
            else:
                cls = 3
            pos_mask[b, 0, iy, ix] = 1.0
            offset_cls_t[b, iy, ix] = cls
            size_scale_t[b, :, iy, ix] = [
                math.log(gw / max(tw, 1e-3)),
                math.log(gh / max(th, 1e-3)),
            ]
            template_wh_t[b, :, iy, ix] = [tw, th]
            box_t[b, :, iy, ix] = [x1h, y1h, x2h, y2h]
            boxes_256.append([x1h, y1h, x2h, y2h])
        all_gt_boxes_256.append(boxes_256)

    return {
        "center": torch.tensor(center_t, device=device),
        "attn": torch.tensor(attn_t, device=device),
        "offset_cls": torch.tensor(offset_cls_t, device=device),
        "size_scale": torch.tensor(size_scale_t, device=device),
        "template_wh": torch.tensor(template_wh_t, device=device),
        "box": torch.tensor(box_t, device=device),
        "pos_mask": torch.tensor(pos_mask, device=device),
        "gt_boxes_256": all_gt_boxes_256,
    }


def center_focal_loss(logits, target):
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1-1e-4)
    pos = target.eq(1.0).float()
    neg = target.lt(1.0).float()
    neg_w = torch.pow(1 - target, 4)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def boxes_iou_xyxy(boxes1, boxes2):
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0]); y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2]); y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-7)


def giou_loss_xyxy(boxes1, boxes2):
    iou = boxes_iou_xyxy(boxes1, boxes2)
    cx1 = torch.min(boxes1[:,0], boxes2[:,0]); cy1 = torch.min(boxes1[:,1], boxes2[:,1])
    cx2 = torch.max(boxes1[:,2], boxes2[:,2]); cy2 = torch.max(boxes1[:,3], boxes2[:,3])
    c_area = (cx2-cx1).clamp(min=0) * (cy2-cy1).clamp(min=0) + 1e-7
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    x1 = torch.max(boxes1[:,0], boxes2[:,0]); y1 = torch.max(boxes1[:,1], boxes2[:,1])
    x2 = torch.min(boxes1[:,2], boxes2[:,2]); y2 = torch.min(boxes1[:,3], boxes2[:,3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    union = area1 + area2 - inter + 1e-7
    giou = iou - (c_area - union) / c_area
    return 1.0 - giou


def gather_center_size_predictions(outputs, targets):
    pos = torch.nonzero(targets["pos_mask"][:, 0] > 0.5, as_tuple=False)
    if pos.numel() == 0:
        return None
    bs, ys, xs = pos[:, 0], pos[:, 1], pos[:, 2]
    offset_logits = outputs["offset_cls_logits"][bs, :, ys, xs]
    size_scale = outputs["size_scale"][bs, :, ys, xs]
    offset_cls_t = targets["offset_cls"][bs, ys, xs]
    size_scale_t = targets["size_scale"][bs, :, ys, xs]
    template_wh = targets["template_wh"][bs, :, ys, xs]
    true_boxes = targets["box"][bs, :, ys, xs]

    offset_table = get_offset_table(device=offset_logits.device, dtype=offset_logits.dtype)
    offset_prob = F.softmax(offset_logits, dim=1)
    pred_offset = offset_prob @ offset_table
    cx = xs.float() + pred_offset[:, 0]
    cy = ys.float() + pred_offset[:, 1]
    scale = torch.exp(torch.clamp(size_scale, min=-2.0, max=2.0))
    pred_w = (template_wh[:, 0] * scale[:, 0]).clamp(min=1e-3, max=pred_size - 1)
    pred_h = (template_wh[:, 1] * scale[:, 1]).clamp(min=1e-3, max=pred_size - 1)
    pred_boxes = torch.stack([cx - 0.5 * pred_w, cy - 0.5 * pred_h, cx + 0.5 * pred_w, cy + 0.5 * pred_h], dim=1)
    pred_boxes = torch.stack([pred_boxes[:, 0].clamp(0, pred_size - 1), pred_boxes[:, 1].clamp(0, pred_size - 1), pred_boxes[:, 2].clamp(0, pred_size - 1), pred_boxes[:, 3].clamp(0, pred_size - 1)], dim=1)
    true_boxes = torch.stack([true_boxes[:, 0].clamp(0, pred_size - 1), true_boxes[:, 1].clamp(0, pred_size - 1), true_boxes[:, 2].clamp(0, pred_size - 1), true_boxes[:, 3].clamp(0, pred_size - 1)], dim=1)
    return offset_logits, offset_cls_t, size_scale, size_scale_t, pred_boxes, true_boxes


def compute_loss(outputs, targets):
    center_loss = center_focal_loss(outputs["center_logits"], targets["center"])
    # No-attention ablation: no attention logits and no auxiliary attention loss.
    attn_loss = outputs["center_logits"].new_tensor(0.0)
    gathered = gather_center_size_predictions(outputs, targets)
    if gathered is None:
        offset_loss = outputs["center_logits"].new_tensor(0.0)
        size_loss = outputs["center_logits"].new_tensor(0.0)
        giou = outputs["center_logits"].new_tensor(0.0)
        npos = 0.0
    else:
        offset_logits, offset_cls_t, size_scale, size_scale_t, pred_boxes, true_boxes = gathered
        offset_loss = F.cross_entropy(offset_logits, offset_cls_t.long(), reduction="mean")
        size_loss = F.smooth_l1_loss(size_scale, size_scale_t, reduction="mean")
        giou = giou_loss_xyxy(pred_boxes, true_boxes).mean()
        npos = float(pred_boxes.shape[0])
    total = lambda_center * center_loss + lambda_attn * attn_loss + lambda_offset_cls * offset_loss + lambda_size * size_loss + lambda_giou * giou
    return total, {
        "total": float(total.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
        "attn": float(attn_loss.detach().cpu()),
        "offset": float(offset_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "giou": float(giou.detach().cpu()),
        "pos": npos,
    }


def decode_predictions(outputs, samples, score_thr=0.25):
    probs = torch.sigmoid(outputs["center_logits"])
    offset_logits_all = outputs["offset_cls_logits"]
    size_scale_all = outputs["size_scale"]
    B, _, H, W = probs.shape
    # No local-maximum suppression before thresholding.
    # This is equivalent to a 1x1 max-pooling peak test: every pixel is a candidate.
    peak = torch.ones_like(probs, dtype=torch.bool)
    offset_table = get_offset_table(device=probs.device, dtype=probs.dtype)
    results = []
    for b in range(B):
        score_map = probs[b, 0]
        mask = peak[b, 0] & (score_map >= score_thr)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            results.append({"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)})
            continue
        scores = score_map[ys, xs]
        if scores.numel() > topk_per_image:
            scores, idx = torch.topk(scores, topk_per_image)
            ys, xs = ys[idx], xs[idx]
        offset_logits = offset_logits_all[b, :, ys, xs].permute(1, 0)
        offset_cls = torch.argmax(offset_logits, dim=1)
        pred_offset = offset_table[offset_cls]
        cx = xs.float() + pred_offset[:, 0]
        cy = ys.float() + pred_offset[:, 1]
        size_scale = size_scale_all[b, :, ys, xs].permute(1, 0)
        size_scale = torch.clamp(size_scale, min=-2.0, max=2.0)
        tw, th = template_wh_256_from_sample(samples[b])
        template_wh = torch.tensor([tw, th], device=probs.device, dtype=probs.dtype).view(1, 2)
        pred_wh = template_wh * torch.exp(size_scale)
        pred_w = pred_wh[:, 0].clamp(min=1e-3, max=W - 1)
        pred_h = pred_wh[:, 1].clamp(min=1e-3, max=H - 1)
        boxes_256 = torch.stack([cx - 0.5 * pred_w, cy - 0.5 * pred_h, cx + 0.5 * pred_w, cy + 0.5 * pred_h], dim=1)
        boxes_256 = torch.stack([boxes_256[:, 0].clamp(0, W - 1), boxes_256[:, 1].clamp(0, H - 1), boxes_256[:, 2].clamp(0, W - 1), boxes_256[:, 3].clamp(0, H - 1)], dim=1)
        boxes_sam = boxes_256 * (sam_img_size / float(pred_size))
        boxes_np = boxes_sam.detach().cpu().numpy().astype(np.float32)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        if boxes_np.shape[0] > 0:
            keep = numpy_nms(boxes_np, scores_np, iou_thresh=nms_iou_thresh, max_dets=max_dets_per_image)
            boxes_np = boxes_np[keep]
            scores_np = scores_np[keep]
        boxes_orig = boxes_sam_padded_to_original(boxes_np.tolist(), samples[b]["meta"])
        results.append({"boxes": np.asarray(boxes_orig, np.float32), "scores": scores_np.astype(np.float32)})
    return results


def iou_np(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = max(0, ax2-ax1) * max(0, ay2-ay1)
    ab = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (aa + ab - inter + 1e-9)


def make_map_metric():
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        max_detection_thresholds=[1, 10, int(ap_max_dets_per_image)],
        class_metrics=False,
        backend="faster_coco_eval",
    )


def safe_float_metric(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def tensors_for_map_from_preds(preds, samples):
    pred_list = []
    target_list = []
    for s, p in zip(samples, preds):
        pred_boxes_np = p["boxes"].astype(np.float32)
        pred_scores_np = p["scores"].astype(np.float32)
        if pred_boxes_np.shape[0] == 0:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
            pred_labels = torch.zeros((0,), dtype=torch.long)
        else:
            pred_boxes = torch.as_tensor(pred_boxes_np, dtype=torch.float32)
            pred_scores = torch.as_tensor(pred_scores_np, dtype=torch.float32)
            pred_labels = torch.ones((pred_boxes.shape[0],), dtype=torch.long)
        gt_boxes_np = np.asarray(boxes_sam_padded_to_original(s["gt_boxes"], s["meta"]), dtype=np.float32)
        if gt_boxes_np.shape[0] == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.long)
        else:
            target_boxes = torch.as_tensor(gt_boxes_np, dtype=torch.float32)
            target_labels = torch.ones((target_boxes.shape[0],), dtype=torch.long)
        # 显式提供 iscrowd，避免 torchmetrics/faster_coco_eval 在内部构造默认 crowd
        # 时触发 Windows/PyTorch 的 SymIntArrayRef native error。
        target_list.append({
            "boxes": target_boxes.contiguous().float().cpu(),
            "labels": target_labels.contiguous().long().cpu(),
        })

        pred_list.append({
            "boxes": pred_boxes.contiguous().float().cpu(),
            "scores": pred_scores.contiguous().float().cpu(),
            "labels": pred_labels.contiguous().long().cpu(),
        })

    return pred_list, target_list


@torch.no_grad()
def evaluate(model, loader, device, score_thr=None, trace_path=None, epoch=None):
    model.eval()
    if score_thr is None:
        score_thr = ap_score_thresh
    metric = make_map_metric()
    total_gt = 0
    total_pred = 0

    def trace(msg):
        if trace_path is not None:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\\n")
        print(msg, flush=True)

    trace(f"epoch {epoch}: eval_start score_thr={score_thr}")
    for eval_i, samples in enumerate(loader, start=1):
        image_ids = [s["image_id"] for s in samples]
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} batch_size={len(samples)} image_ids={image_ids[:3]} before_forward")
        with amp_autocast(device):
            out = model(samples)
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} after_forward")
        preds = decode_predictions(out, samples, score_thr)
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} after_decode")
        pred_list, target_list = tensors_for_map_from_preds(preds, samples)
        metric.update(pred_list, target_list)
        for s, p in zip(samples, preds):
            total_gt += len(s["gt_boxes"])
            total_pred += len(p["boxes"])
        del out, preds, pred_list, target_list
        if torch.cuda.is_available() and eval_i % eval_empty_cache_every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} after_metric_update total_gt={total_gt} total_pred={total_pred}")
    trace(f"epoch {epoch}: eval_loop_done total_gt={total_gt} total_pred={total_pred}")
    trace(f"epoch {epoch}: before_metric_compute")
    result = metric.compute()
    ap = safe_float_metric(result.get("map", torch.tensor(0.0)))
    ap50 = safe_float_metric(result.get("map_50", torch.tensor(0.0)))
    ap75 = safe_float_metric(result.get("map_75", torch.tensor(0.0)))
    ap = 0.0 if ap < 0 else ap
    ap50 = 0.0 if ap50 < 0 else ap50
    ap75 = 0.0 if ap75 < 0 else ap75
    trace(f"epoch {epoch}: eval_done AP={ap:.6f} AP50={ap50:.6f} AP75={ap75:.6f}")
    return {"AP": ap, "AP50": ap50, "AP75": ap75, "GT": total_gt, "Pred": total_pred, "NumPredForAP": total_pred, "score_thr_for_ap": score_thr}


def draw_vis(samples, preds, out_dir, max_images=20):
    ensure_dir(out_dir)

    for i, (s, p) in enumerate(zip(samples, preds)):
        if i >= max_images:
            break
        img_for_vis = s.get("image", None)
        if img_for_vis is None:
            img_for_vis = np.zeros((sam_img_size, sam_img_size, 3), dtype=np.uint8)
        pil = Image.fromarray(img_for_vis.copy()).convert("RGB")
        draw = ImageDraw.Draw(pil)
        for j, gb in enumerate(s["gt_boxes"]):
            x1,y1,x2,y2 = gb; draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
            if j < 30: draw.text((x1, max(0,y1-12)), f"G{j}", fill=(0,255,0))
        x1,y1,x2,y2 = s["template_box"]; draw.rectangle([x1,y1,x2,y2], outline=(0,128,255), width=4); draw.text((x1,max(0,y1-16)), "TEMPLATE", fill=(0,128,255))
        scale = s["meta"]["scale"]
        for j, (box, sc) in enumerate(zip(p["boxes"], p["scores"])):
            px1,py1,px2,py2 = box
            sb = [px1*scale, py1*scale, px2*scale, py2*scale]
            draw.rectangle(sb, outline=(255,0,0), width=2)
            if j < 50: draw.text((sb[0], max(0,sb[1]-12)), f"{sc:.2f}", fill=(255,0,0))
        pil.save(os.path.join(out_dir, f"{i:03d}_{s['image_id']}.jpg"), quality=95)



def gpu_mem(prefix=""):
    """
    打印 CUDA 显存状态，用于定位 epoch 末尾 native crash 前的显存状态。
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_reserved() / 1024**3
    print(f"[GPU] {prefix} allocated={alloc:.2f}GB reserved={reserv:.2f}GB peak_reserved={peak:.2f}GB", flush=True)


def get_trainable_state_dict(model):
    """
    只保存可训练模块，不保存冻结 SAM。
    这样 checkpoint 小很多，也能显著降低 torch.save 在 epoch 末尾触发 native crash 的概率。
    加载时 strict=False。
    """
    state = model.state_dict()
    return {k: v.detach().cpu() for k, v in state.items() if not k.startswith("sam.")}



def append_jsonl(path, record):
    """
    训练日志 JSONL，出问题时至少能知道最后完成到哪一步。
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\\n")





def resolve_resume_path():
    """
    返回断点续训 checkpoint 路径。
    resume_checkpoint_path 留空时，默认读取 save_dir/last_model.pth。
    """
    if resume_checkpoint_path is not None and str(resume_checkpoint_path).strip():
        return resume_checkpoint_path
    return os.path.join(save_dir, "last_model.pth")


def try_resume_training(model, optimizer, device):
    """
    如果 last_model.pth 存在，则恢复 model / optimizer，并返回 start_epoch。
    返回：
        start_epoch: 下一轮要开始训练的 epoch
        resumed: 是否成功恢复
        checkpoint_path: 实际读取的 checkpoint 路径
    """
    if not auto_resume:
        print("[Resume] auto_resume=False, start from epoch 1.", flush=True)
        return 1, False, None

    checkpoint_path = resolve_resume_path()

    if not os.path.isfile(checkpoint_path):
        print(f"[Resume] checkpoint not found: {checkpoint_path}", flush=True)
        print("[Resume] start from epoch 1.", flush=True)
        return 1, False, checkpoint_path

    print(f"[Resume] loading checkpoint: {checkpoint_path}", flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[Resume] model loaded. missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if len(missing) > 0:
        print("[Resume] missing examples:", missing[:10], flush=True)
    if len(unexpected) > 0:
        print("[Resume] unexpected examples:", unexpected[:10], flush=True)

    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[Resume] optimizer loaded.", flush=True)
        except Exception as e:
            print(f"[Resume] optimizer load failed, continue with fresh optimizer. reason: {repr(e)}", flush=True)
    else:
        print("[Resume] no optimizer state in checkpoint, continue with fresh optimizer.", flush=True)

    last_epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    start_epoch = last_epoch + 1

    if start_epoch > epochs:
        print(f"[Resume] checkpoint epoch={last_epoch}, configured epochs={epochs}. Nothing to train unless you increase epochs.", flush=True)
    else:
        print(f"[Resume] last_epoch={last_epoch}, continue from epoch {start_epoch}.", flush=True)

    return start_epoch, True, checkpoint_path


def train():
    set_seed(seed)
    if debug_anomaly:
        torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        try:
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        except Exception:
            pass
    ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print("TDCC-SAM FSCD-147 Cached ViT-H Fully Box-Supervised Detector")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count visible:", torch.cuda.device_count())
    print("save_dir:", save_dir)
    print("="*80)

    train_set = FSCD147InstancesDataset(fscd_root, split=train_split, training=True, max_exemplars=3)
    val_set = FSCD147InstancesDataset(fscd_root, split=eval_split, training=False, max_exemplars=1)
    if len(train_set) == 0:
        raise RuntimeError("FSCD-147 train_set is empty. Check fscd_root, instances_train.json and SAM cache.")
    if len(val_set) == 0:
        raise RuntimeError("FSCD-147 val_set is empty. Check fscd_root, instances_val.json and SAM cache.")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), drop_last=False)

    model = TDCCDetector().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(device)

    start_epoch, resumed, resumed_from = try_resume_training(model, opt, device)

    config = {k: v for k, v in globals().items() if k in ["fscd_root","train_split","eval_split","sam_checkpoint","sam_model_type","sam_img_size","pred_size","project_dim","batch_size","eval_batch_size","lr","weight_decay","lambda_center","lambda_attn","lambda_offset_cls","lambda_size","lambda_giou","attention_radius_ratio","ap_score_thresh","ap_max_dets_per_image","max_dets_per_image","nms_iou_thresh","auto_resume","resume_checkpoint_path","sam_feature_cache_root","cached_feature_to_float32", "tca_version"]}
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log_jsonl_path = os.path.join(save_dir, "train_log.jsonl")
    epoch_csv_path = os.path.join(save_dir, "epoch_metrics.csv")
    stage_log_path = os.path.join(save_dir, "stage_log.txt")

    # 断点续训时不要覆盖旧日志；从头训练时重新写日志头。
    if resumed:
        with open(stage_log_path, "a", encoding="utf-8") as f:
            f.write(f"\nresume from {resumed_from}, start_epoch={start_epoch}\n")
        if not os.path.isfile(epoch_csv_path):
            with open(epoch_csv_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,center,attn,offset,size,giou,pos,AP,AP50,AP75,GT,Pred,NumPredForAP,score_thr_for_ap,time_sec\n")
    else:
        with open(stage_log_path, "w", encoding="utf-8") as f:
            f.write("stage log\n")
        with open(epoch_csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,center,attn,offset,size,giou,pos,AP,AP50,AP75,GT,Pred,NumPredForAP,score_thr_for_ap,time_sec\n")

    best_ap = best_ap50 = best_ap75 = -1.0

    # 断点续训时读取已有 best_model.pth 的 AP，避免后续保存逻辑从 -1 重新开始。
    best_path = os.path.join(save_dir, "best_model.pth")
    if os.path.isfile(best_path):
        try:
            best_ckpt = torch.load(best_path, map_location="cpu")
            best_metrics = best_ckpt.get("metrics", {})
            best_ap = float(best_metrics.get("AP", -1.0))
            print(f"[Resume] existing best_model.pth: AP={best_ap:.4f}", flush=True)
        except Exception as e:
            print(f"[Resume] failed to read {best_path}: {repr(e)}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[Start Epoch {epoch:03d}]", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.train()
        t0 = time.time()
        running = defaultdict(float); steps = 0
        gpu_mem(f"epoch {epoch:03d} start")
        for it, samples in enumerate(train_loader, start=1):
            opt.zero_grad(set_to_none=True)
            with amp_autocast(device):
                out = model(samples)
                targets = build_targets(samples, device)
                loss, info = compute_loss(out, targets)

            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            steps += 1
            for k, v in info.items(): running[k] += v
            if it % 100 == 0:
                bw = out["branch_weights"].detach().mean(dim=0).cpu().numpy()
                cg_mean = out.get("channel_gate_mean", None)
                cg_std = out.get("channel_gate_std", None)
                if cg_mean is not None and cg_std is not None:
                    cg_text = f" cg_mean={float(cg_mean.detach().cpu()):.3f} cg_std={float(cg_std.detach().cpu()):.3f}"
                else:
                    cg_text = ""
                print(f"[Epoch {epoch:03d} | Iter {it:04d}/{len(train_loader):04d}] loss={running['total']/steps:.5f} center={running['center']/steps:.5f} attn={running['attn']/steps:.5f} offset={running['offset']/steps:.5f} size={running['size']/steps:.5f} giou={running['giou']/steps:.5f} pos={running['pos']/steps:.1f} branch={np.round(bw,3)}{cg_text}", flush=True)

        # 清理最后一个 iteration 的输出引用，避免带着训练图进入评估/保存阶段。
        try:
            del out, targets, loss
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        with open(stage_log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch {epoch:03d}: before_last_save\n")
        print(f"[Epoch {epoch:03d}] before normal last save", flush=True)
        gpu_mem(f"epoch {epoch:03d} before last save")

        torch.save({
            "epoch": epoch,
            "model": get_trainable_state_dict(model),
            "optimizer": opt.state_dict(),
            "config": config,
        }, os.path.join(save_dir, "last_model.pth"))

        with open(stage_log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch {epoch:03d}: after_last_save\n")
        print(f"[Epoch {epoch:03d}] after normal last save", flush=True)
        gpu_mem(f"epoch {epoch:03d} after last save")
        if epoch % eval_interval == 0:
            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: before_eval\n")
            print(f"[Epoch {epoch:03d}] before eval", flush=True)
            gpu_mem(f"epoch {epoch:03d} before eval")

            metrics = evaluate(model, val_loader, device, score_thr=ap_score_thresh, trace_path=stage_log_path, epoch=epoch)

            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: after_eval\n")
            print(f"[Epoch {epoch:03d}] after eval", flush=True)
            gpu_mem(f"epoch {epoch:03d} after eval")

            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={running['total']/max(steps,1):.5f} "
                f"center={running['center']/max(steps,1):.5f} "
                f"attn={running['attn']/max(steps,1):.5f} "
                f"offset={running['offset']/max(steps,1):.5f} "
                f"size={running['size']/max(steps,1):.5f} "
                f"giou={running['giou']/max(steps,1):.5f} "
                f"pos={running['pos']/max(steps,1):.1f} | "
                f"AP={metrics['AP']:.4f} "
                f"AP50={metrics['AP50']:.4f} "
                f"AP75={metrics['AP75']:.4f} | "
                f"GT={metrics['GT']} Pred={metrics['Pred']} "
                f"score_thr_for_ap={metrics['score_thr_for_ap']} "
                f"time={time.time()-t0:.1f}s"
            )
            epoch_record = {
                "epoch": epoch,
                "train_loss": running['total']/max(steps,1),
                "center": running['center']/max(steps,1),
                "attn": running['attn']/max(steps,1),
                "giou": running['giou']/max(steps,1),
                "offset": running['offset']/max(steps,1),
                "size": running['size']/max(steps,1),
                "pos": running['pos']/max(steps,1),
                **metrics,
                "time_sec": time.time()-t0,
            }
            append_jsonl(log_jsonl_path, epoch_record)

            with open(epoch_csv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch_record['epoch']},"
                    f"{epoch_record['train_loss']:.6f},"
                    f"{epoch_record['center']:.6f},"
                    f"{epoch_record['attn']:.6f},"
                    f"{epoch_record['offset']:.6f},"
                    f"{epoch_record['size']:.6f},"
                    f"{epoch_record['giou']:.6f},"
                    f"{epoch_record['pos']:.3f},"
                    f"{epoch_record['AP']:.6f},"
                    f"{epoch_record['AP50']:.6f},"
                    f"{epoch_record['AP75']:.6f},"
                    f"{epoch_record['GT']},"
                    f"{epoch_record['Pred']},"
                    f"{epoch_record['NumPredForAP']},"
                    f"{epoch_record['score_thr_for_ap']},"
                    f"{epoch_record['time_sec']:.3f}\n"
                )

            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: before_best_saves\n")
            print(f"[Epoch {epoch:03d}] before best saves", flush=True)

            if metrics["AP"] > best_ap:
                best_ap = metrics["AP"]
                torch.save({
                    "epoch": epoch,
                    "model": get_trainable_state_dict(model),
                    "optimizer": opt.state_dict(),
                    "config": config,
                    "metrics": metrics,
                }, os.path.join(save_dir, "best_model.pth"))
                print(f"[Save Best Model by AP] AP={best_ap:.4f} AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f}", flush=True)

            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: after_best_saves\n")
            print(f"[Epoch {epoch:03d}] after best saves", flush=True)
            gpu_mem(f"epoch {epoch:03d} after best saves")
        if epoch % vis_interval == 0:
            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: before_visual\n")
            print(f"[Epoch {epoch:03d}] before visual", flush=True)
            gpu_mem(f"epoch {epoch:03d} before visual")

            model.eval()
            vis_samples = [val_set[i] for i in range(min(num_vis_images, len(val_set)))]
            preds = []
            with torch.no_grad():
                for vs in vis_samples:
                    with amp_autocast(device):
                        out_vis = model([vs])
                    pred_vis = decode_predictions(out_vis, [vs], score_thresh)
                    preds.extend(pred_vis)
                    del out_vis, pred_vis
            out_dir = os.path.join(save_dir, "visuals", f"epoch_{epoch:03d}")
            draw_vis(vis_samples, preds, out_dir, num_vis_images)
            print("[Visual saved]", out_dir, flush=True)

            with open(stage_log_path, "a", encoding="utf-8") as f:
                f.write(f"epoch {epoch:03d}: after_visual\n")
            gpu_mem(f"epoch {epoch:03d} after visual")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with open(stage_log_path, "a", encoding="utf-8") as f:
            f.write(f"epoch {epoch:03d}: end_epoch\n")
        print(f"[End Epoch {epoch:03d}] time={time.time()-t0:.1f}s", flush=True)



# =========================================================
# Standalone inference entry. No training-script import is used.
# =========================================================
import csv

# Force paper-aligned 64+1 post-processing limits.
ap_max_dets_per_image = int(AP_MAX_DETS_PER_IMAGE_INFER)
topk_per_image = int(TOPK_PER_IMAGE_INFER)
max_dets_per_image = int(MAX_DETS_PER_IMAGE_INFER)
nms_iou_thresh = float(NMS_IOU_THRESH)
score_thresh = float(SCORE_THRESH)
num_eval_shots = int(NUM_EVAL_SHOTS)
eval_trace_every = 20


def _safe_load_state_dict(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Edit CHECKPOINT_PATH at the top of this script."
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=bool(STRICT_LOAD))
    if not bool(STRICT_LOAD):
        print(f"[Load] missing keys: {len(missing)} unexpected keys: {len(unexpected)}", flush=True)
    return ckpt


def _predict_1shot_batch(model, samples, device, score_thr):
    with torch.no_grad():
        with amp_autocast(device):
            outputs = model(samples)
        preds = decode_predictions(outputs, samples, score_thr=score_thr)
    return preds


def _predict_multishot_batch(model, samples, device, score_thr, n_shots):
    # Generic standalone 1/3-shot inference. It does not depend on training helper functions.
    if n_shots <= 1:
        return _predict_1shot_batch(model, samples, device, score_thr), [1 for _ in samples]

    merged_preds = []
    used_shots_all = []
    for sample in samples:
        candidates = sample.get("template_candidates", None)
        if candidates is None or len(candidates) == 0:
            candidates = [sample["template_box"]]
        candidates = list(candidates)[:int(n_shots)]

        all_boxes = []
        all_scores = []
        for tpl_box in candidates:
            s_i = dict(sample)
            s_i["template_box"] = tpl_box
            with torch.no_grad():
                with amp_autocast(device):
                    outputs_i = model([s_i])
                pred_i = decode_predictions(outputs_i, [s_i], score_thr=score_thr)[0]
            if len(pred_i["boxes"]) > 0:
                all_boxes.append(np.asarray(pred_i["boxes"], dtype=np.float32))
                all_scores.append(np.asarray(pred_i["scores"], dtype=np.float32))

        if len(all_boxes) == 0:
            merged_preds.append({"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32)})
            used_shots_all.append(len(candidates))
            continue

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep = numpy_nms(boxes, scores, iou_thresh=float(nms_iou_thresh), max_dets=int(max_dets_per_image))
        boxes = boxes[keep]
        scores = scores[keep]
        if len(scores) > int(max_dets_per_image):
            order = np.argsort(-scores)[:int(max_dets_per_image)]
            boxes = boxes[order]
            scores = scores[order]
        merged_preds.append({"boxes": boxes.astype(np.float32), "scores": scores.astype(np.float32)})
        used_shots_all.append(len(candidates))

    return merged_preds, used_shots_all


def build_infer_dataset():
    # Keep up to 3 exemplar boxes so USE_THREE_SHOT can switch between 1-shot and 3-shot.
    return FSCD147InstancesDataset(
        fscd_root,
        split=eval_split,
        training=False,
        max_exemplars=max(3, int(NUM_EVAL_SHOTS)),
    )


def evaluate_for_paper(model, loader, device):
    model.eval()
    metric = make_map_metric()

    total_gt = 0
    total_pred = 0
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    n_images = 0
    pred_rows = []
    used_shots_sum = 0

    protocol = "3-shot" if bool(USE_THREE_SHOT) else "1-shot"
    n_shots = int(NUM_EVAL_SHOTS) if bool(USE_THREE_SHOT) else 1
    print(f"[Infer] protocol={protocol} score_thr={SCORE_THRESH} nms={NMS_IOU_THRESH}", flush=True)
    print(f"[Infer] topk={topk_per_image} max_dets={max_dets_per_image} ap_max_dets={ap_max_dets_per_image}", flush=True)

    for batch_idx, samples in enumerate(loader, start=1):
        preds, used_shots = _predict_multishot_batch(model, samples, device, float(SCORE_THRESH), n_shots)
        pred_list, target_list = tensors_for_map_from_preds(preds, samples)
        metric.update(pred_list, target_list)

        for s, p, us in zip(samples, preds, used_shots):
            gt_count = int(len(s["gt_boxes"]))
            pred_count = int(len(p["boxes"]))
            err = float(pred_count - gt_count)
            total_gt += gt_count
            total_pred += pred_count
            abs_err_sum += abs(err)
            sq_err_sum += err * err
            n_images += 1
            used_shots_sum += int(us)
            pred_rows.append({
                "image_id": str(s.get("image_id", "")),
                "file_name": str(s.get("file_name", "")),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "error": err,
                "used_shots": int(us),
            })

        if batch_idx == 1 or batch_idx % int(eval_trace_every) == 0:
            print(
                f"[Infer] batch {batch_idx}/{len(loader)} images={n_images} "
                f"GT={total_gt} Pred={total_pred}",
                flush=True,
            )
        if torch.cuda.is_available() and batch_idx % int(eval_empty_cache_every) == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    result = metric.compute()
    ap = max(0.0, safe_float_metric(result.get("map", torch.tensor(0.0))))
    ap50 = max(0.0, safe_float_metric(result.get("map_50", torch.tensor(0.0))))
    ap75 = max(0.0, safe_float_metric(result.get("map_75", torch.tensor(0.0))))
    mae = abs_err_sum / max(n_images, 1)
    rmse = math.sqrt(sq_err_sum / max(n_images, 1))
    bias = (total_pred - total_gt) / max(n_images, 1)
    pred_gt_ratio = total_pred / max(total_gt, 1)
    avg_used_shots = used_shots_sum / max(n_images, 1)

    metrics = {
        "protocol": protocol,
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "GT": int(total_gt),
        "Pred": int(total_pred),
        "NumImages": int(n_images),
        "score_thr": float(SCORE_THRESH),
        "nms_iou_thresh": float(NMS_IOU_THRESH),
        "topk_per_image": int(topk_per_image),
        "max_dets_per_image": int(max_dets_per_image),
        "ap_max_dets_per_image": int(ap_max_dets_per_image),
        "avg_used_shots": float(avg_used_shots),
    }
    if not False:
        metrics.update({
            "MAE": float(mae),
            "RMSE": float(rmse),
            "Bias": float(bias),
            "PredGT": float(pred_gt_ratio),
        })
    return metrics, pred_rows


def save_outputs(metrics, pred_rows):
    ensure_dir(OUTPUT_DIR)
    if SAVE_METRICS_JSON:
        with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        with open(os.path.join(OUTPUT_DIR, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
    if SAVE_PREDICTIONS_JSONL:
        with open(os.path.join(OUTPUT_DIR, "counts_per_image.jsonl"), "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if bool(allow_tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device} CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}", flush=True)

    dataset = build_infer_dataset()
    loader = DataLoader(
        dataset,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
    )

    model = TDCCDetector().to(device)
    ckpt = _safe_load_state_dict(model, CHECKPOINT_PATH, device)
    if isinstance(ckpt, dict) and "metrics" in ckpt:
        print(f"[Checkpoint metrics] {ckpt.get('metrics')}", flush=True)

    metrics, pred_rows = evaluate_for_paper(model, loader, device)
    save_outputs(metrics, pred_rows)

    print("\n[Final Metrics]", flush=True)
    print(f"AP={metrics['AP']:.6f} AP50={metrics['AP50']:.6f} AP75={metrics['AP75']:.6f}", flush=True)
    if "MAE" in metrics:
        print(
            f"MAE={metrics['MAE']:.6f} RMSE={metrics['RMSE']:.6f} "
            f"Bias={metrics['Bias']:.6f} PredGT={metrics['PredGT']:.6f}",
            flush=True,
        )
    print(f"GT={metrics['GT']} Pred={metrics['Pred']} Images={metrics['NumImages']}", flush=True)
    print(f"[Saved] {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
