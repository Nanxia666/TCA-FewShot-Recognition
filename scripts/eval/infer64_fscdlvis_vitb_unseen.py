import os

# =========================================================
# Inference controls: edit here
# =========================================================
CUDA_VISIBLE_DEVICES = r"0"   # e.g. "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

CHECKPOINT_PATH = r"model/fscdlvis_unseen_vitb/best_model_by_AP_1shot.pth"
OUTPUT_DIR = r"infer_outputs/fscdlvis_vitb_unseen_64plus1"

# Evaluation protocol
USE_THREE_SHOT = True          # True = 3-shot, False = 1-shot
NUM_EVAL_SHOTS = 3             # used only when USE_THREE_SHOT=True

# Post-processing
SCORE_THRESH = 0.03
NMS_IOU_THRESH = 0.5
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
# FSCD-LVIS 配置
# 目录结构大致要求：
#   FSCD_LVIS/
#       annotations/
#           instances_train.json
#           instances_val.json
#           instances_test.json
#       images/ 或者图片在 json 的 file_name 相对路径中
#
# 这个版本不依赖 Train_Test_Val_FSC_147.json，也不依赖 annotation_FSC147_384.json。
# 直接以 instances_train.json 的 images 列表作为 train，
# 以 instances_test.json 的 images 列表作为 test。
#
# 由于当前你说只有 instances_*.json，没有额外 exemplar json，
# 本脚本默认从同一张图的 GT boxes 里选 template：
#   train: 随机选一个 GT box 当 template
#   test : 默认选第一个 GT box 当 template
# 如果后面你找到官方 support/exemplar 文件，再替换 Dataset 里的 template 读取逻辑即可。
# =========================================================
fscd_lvis_root = r"FSCD_LVIS/FSCD_LVIS"
train_split = "train"
eval_split = "test"
# FSCD-LVIS 子集：可选 "seen" 或 "unseen"
# seen 使用 count_train.json / instances_train.json
# unseen 使用 unseen_count_train.json / unseen_instances_train.json
fscd_lvis_subset = "unseen"
# FSCD-LVIS counting 标注：里面的 boxes 作为官方 support/template boxes
count_annotation_files = {
    "seen": {
        "train": "count_train.json",
        "val": "count_val.json",
        "test": "count_test.json",
    },
    "unseen": {
        "train": "unseen_count_train.json",
        "val": "unseen_count_val.json",
        "test": "unseen_count_test.json",
    },
}
# 每张图最多使用多少个 count boxes 作为候选模板；训练随机选 1 个，测试默认用第 1 个
max_count_templates = 5
# 训练和逐 epoch checkpoint 选择均使用 1-shot；仅训练结束后对候选最佳模型做 3-shot。
num_eval_shots = 3  # only used for final evaluation after training

sam_checkpoint = r"sam_vit_b_01ec64.pth"
sam_model_type = "vit_b"

# =========================================================
# 预提取 SAM feature cache 配置
# =========================================================
# 默认兼容以下结构：
#   cache_sam_vitb_1024/train/*.pt
#   cache_sam_vitb_1024/test/*.pt
#   cache_sam_vitb_1024/FSCD_LVIS/train/*.pt
#   cache_sam_vitb_1024/FSCD_LVIS/test/*.pt
#   cache_sam_vitb_1024/seen/train/*.pt 或 cache_sam_vitb_1024/unseen/train/*.pt
# 若你的实际缓存目录不同，直接修改 sam_feature_cache_root 或追加 extra_sam_feature_cache_roots。
sam_feature_cache_root = r"cache_sam_vitb_1024"
extra_sam_feature_cache_roots = [
    r"cache_sam_vitb_1024/FSCD_LVIS",
    r"cache_sam_vitb_1024/FSCD-LVIS",
]
strict_feature_cache = True
cached_feature_to_float32 = True

# =========================================================
# Speed / AMP options for inference
# =========================================================
use_amp = True
amp_dtype = "float16"   # "float16" or "bfloat16"
allow_tf32 = True
save_dir = r"outputs_tdcc_sam_fscdlvis_unseen_no_attention_train1shot_epoch1shot_final3shot_cached"

# 自动断点续训：如果 save_dir/last_model.pth 存在，就从下一轮 epoch 继续
auto_resume = True
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
eval_batch_size = 4
num_workers = 0
lr = 2e-4
weight_decay = 1e-4
grad_clip_norm = 5.0

lambda_center = 1.0
lambda_giou = 2.0
lambda_size = 1.0
lambda_attn = 0.0  # attention suppression removed
lambda_offset_cls = 0.5

gaussian_min_radius = 1
gaussian_radius_ratio = 0.15
attention_radius_ratio = 0.30

# AP 评估时为了得到完整 PR 曲线，decode 阈值应低一些
ap_score_thresh = 0.25
# AP 评估每图最多预测数，避免评估阶段候选过多
ap_max_dets_per_image = 4500

# eval 内部追踪设置
eval_trace_every = 10
eval_empty_cache_every = 20

score_thresh = 0.25
topk_per_image = 4500
max_dets_per_image = 4500
nms_iou_thresh = 0.4

eval_interval = 1
vis_interval = 5
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


def get_amp_dtype():
    if str(amp_dtype).lower() in ["bf16", "bfloat16"]:
        return torch.bfloat16
    return torch.float16


def amp_autocast(device):
    if not (bool(use_amp) and device.type == "cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=get_amp_dtype(), enabled=True)


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


class FSCDLVISInstancesDataset(Dataset):
    """
    FSCD-LVIS 全框监督 Dataset。

    GT 检测框：
        annotations/instances_{split}.json
        COCO 格式，bbox = [x, y, w, h]

    Template/support 框：
        annotations/count_train.json / count_val.json / count_test.json 或 unseen_count_train.json / unseen_count_val.json / unseen_count_test.json
        里面的 boxes 字段，格式通常是 [x, y, w, h]
        points 字段只用于 counting，不参与本检测训练。

    返回字段兼容原训练脚本：
        image
        image_id
        file_name
        template_box
        template_candidates
        gt_boxes
        meta
    """
    def __init__(self, root, split="train", training=True):
        self.root = root
        self.split = split
        self.training = training

        self.ann_dir = os.path.join(root, "annotations")

        subset = str(fscd_lvis_subset).lower().strip()
        if subset not in ["seen", "unseen"]:
            raise ValueError(f"fscd_lvis_subset must be 'seen' or 'unseen', got: {fscd_lvis_subset}")

        if subset == "seen":
            self.instance_file = os.path.join(self.ann_dir, f"instances_{split}.json")
        else:
            self.instance_file = os.path.join(self.ann_dir, f"unseen_instances_{split}.json")

        count_name = count_annotation_files.get(subset, {}).get(split, f"{'unseen_' if subset == 'unseen' else ''}count_{split}.json")
        self.count_file = os.path.join(self.ann_dir, count_name)

        if not os.path.isfile(self.instance_file):
            raise FileNotFoundError(self.instance_file)
        if not os.path.isfile(self.count_file):
            raise FileNotFoundError(
                f"Cannot find count annotation: {self.count_file}. "
                f"Please check count_annotation_files config."
            )

        self.coco = COCO(self.instance_file)
        self.count_records_by_file = self.load_count_records(self.count_file)

        self.image_search_dirs = self.build_image_search_dirs(root)

        # ---------------------------------------------------------
        # SAM feature cache：训练、1-shot epoch eval、最终 3-shot eval 均共用。
        # 支持 root/split、root/subset/split、root/FSCD_LVIS/split 等目录。
        # ---------------------------------------------------------
        current_subset = str(globals().get("fscd_lvis_subset", "seen")).lower().strip()
        self.cache_roots = list(dict.fromkeys(
            [sam_feature_cache_root] + list(extra_sam_feature_cache_roots)
        ))
        self.cache_dirs = []
        for cache_root_i in self.cache_roots:
            candidate_dirs = [
                os.path.join(cache_root_i, current_subset, split),
                os.path.join(cache_root_i, split),
                os.path.join(cache_root_i, "FSCD_LVIS", current_subset, split),
                os.path.join(cache_root_i, "FSCD_LVIS", split),
                os.path.join(cache_root_i, "FSCD-LVIS", current_subset, split),
                os.path.join(cache_root_i, "FSCD-LVIS", split),
                os.path.join(cache_root_i, "images"),
            ]
            for d in candidate_dirs:
                if d not in self.cache_dirs:
                    self.cache_dirs.append(d)


        self.keys = []
        miss_img = 0
        miss_box = 0
        miss_count = 0
        miss_count_box = 0
        miss_feature_cache = 0

        for img_id, img_info in self.coco.imgs.items():
            file_name = img_info["file_name"]
            img_path, real_name = self.find_image_file_lvis(file_name)
            if img_path is None:
                miss_img += 1
                continue

            count_rec = self.get_count_record(file_name, real_name)
            if count_rec is None:
                miss_count += 1
                continue

            count_boxes = self.get_template_boxes_from_count_record(count_rec)
            if len(count_boxes) == 0:
                miss_count_box += 1
                continue

            image_w = int(img_info.get("width", 0)) if img_info.get("width", 0) else None
            image_h = int(img_info.get("height", 0)) if img_info.get("height", 0) else None
            if image_w is None or image_h is None:
                try:
                    with Image.open(img_path) as im:
                        image_w, image_h = im.size
                except Exception:
                    miss_img += 1
                    continue

            boxes = self.get_bboxes_from_coco_id(int(img_id), image_w, image_h)
            if len(boxes) == 0:
                miss_box += 1
                continue

            cache_path = self.find_cache_file(real_name, file_name)
            if cache_path is None and strict_feature_cache:
                miss_feature_cache += 1
                continue

            self.keys.append(int(img_id))

        print(f"[FSCDLVISInstancesDataset]")
        print(f"  root: {root}")
        print(f"  subset: {fscd_lvis_subset}")
        print(f"  split: {split}, training={training}")
        print(f"  instances: {self.instance_file}")
        print(f"  count file: {self.count_file}")
        print(f"  image search dirs:")
        for d in self.image_search_dirs:
            print(f"    - {d}")
        print(f"  coco images: {len(self.coco.imgs)}")
        print(f"  count records: {len(self.count_records_by_file)}")
        print(f"  valid images: {len(self.keys)}")
        print(f"  missing image: {miss_img}")
        print(f"  missing detection bbox: {miss_box}")
        print(f"  missing count record: {miss_count}")
        print(f"  cache roots:")
        for cache_root_i in self.cache_roots:
            print(f"    - {cache_root_i}")
        print(f"  searched cache dirs:")
        for cache_dir_i in self.cache_dirs:
            print(f"    - {cache_dir_i}")
        print(f"  missing count template boxes: {miss_count_box}")
        print(f"  missing feature cache: {miss_feature_cache}")

    def load_count_records(self, count_file):
        """
        兼容两类结构：
            1) list[dict]，每个 dict 有 file_name/image_id/boxes/points
            2) dict，key 为 file_name 或 image_id，value 为 record
        """
        with open(count_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = {}

        if isinstance(data, list):
            iterator = data
        elif isinstance(data, dict):
            # 有些 json 可能是 {"annotations":[...]} 或 {"images":[...]}
            if "annotations" in data and isinstance(data["annotations"], list):
                iterator = data["annotations"]
            elif "data" in data and isinstance(data["data"], list):
                iterator = data["data"]
            else:
                iterator = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        rec = dict(v)
                        rec.setdefault("file_name", k)
                        iterator.append(rec)
        else:
            iterator = []

        for rec in iterator:
            if not isinstance(rec, dict):
                continue
            file_name = rec.get("file_name", None)
            if file_name is None:
                continue
            records[Path(file_name).name] = rec
            records[Path(file_name).stem] = rec

            # 顺手按 image_id/id 也存一份，兜底
            if "image_id" in rec:
                records[str(rec["image_id"])] = rec
            if "id" in rec:
                records[str(rec["id"])] = rec

        return records

    def get_count_record(self, file_name, real_name):
        """
        优先按 basename/stem 匹配 count json。
        """
        candidates = [
            Path(file_name).name,
            Path(file_name).stem,
            Path(real_name).name,
            Path(real_name).stem,
        ]
        for k in candidates:
            if k in self.count_records_by_file:
                return self.count_records_by_file[k]

        # 兜底：用 COCO image id 去匹配
        if file_name in self.coco.imgs:
            img_id = self.coco.imgs[file_name]["id"]
            if str(img_id) in self.count_records_by_file:
                return self.count_records_by_file[str(img_id)]

        return None

    def count_box_to_xyxy(self, b):
        """
        count json 里的 boxes 从你给的样例看是 [x, y, w, h]。
        这里做一个宽松判断：
            - 默认按 xywh 转 xyxy
            - 如果看起来像 xyxy，也能手动在这里切换。
        """
        if b is None or len(b) != 4:
            return None

        x, y, a, c = [float(v) for v in b]

        # 默认：xywh
        w, h = a, c
        return [x, y, x + w, y + h]

    def get_template_boxes_from_count_record(self, rec):
        boxes_raw = rec.get("boxes", [])
        boxes = []
        for b in boxes_raw[:int(max_count_templates)]:
            box = self.count_box_to_xyxy(b)
            if box is not None:
                boxes.append(box)
        return boxes

    def build_image_search_dirs(self, root):
        candidates = [
            root,
            os.path.join(root, "images"),
            os.path.join(root, "Images"),
            os.path.join(root, "JPEGImages"),
            os.path.join(root, "train2017"),
            os.path.join(root, "val2017"),
            os.path.join(root, "test2017"),
            os.path.join(root, "FSC147"),
            os.path.join(root, "images_384_VarV2"),
        ]
        out = []
        for d in candidates:
            if os.path.isdir(d) and d not in out:
                out.append(d)
        return out

    def find_image_file_lvis(self, file_name):
        # 1) file_name 作为相对路径
        for base in self.image_search_dirs:
            p = Path(base) / file_name
            if p.is_file():
                return str(p), p.name

        # 2) 只用 basename 搜索常见目录
        base_name = Path(file_name).name
        for base in self.image_search_dirs:
            p = Path(base) / base_name
            if p.is_file():
                return str(p), p.name

        # 3) stem + 常见扩展
        stem = Path(file_name).stem
        for base in self.image_search_dirs:
            found, real_name = find_image_file(base, stem)
            if found is not None:
                return found, real_name

        return None, None

    def get_bboxes_from_coco_id(self, img_id, image_w, image_h):
        ann_ids = self.coco.getAnnIds(imgIds=[int(img_id)])
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


    def find_cache_file(self, real_name, file_name=None):
        """
        按图像 stem 搜索预提取的 SAM embedding .pt。
        支持 stem.pt 和 stem_*.pt 两种命名。
        """
        stems = []
        for name_i in [real_name, file_name]:
            if name_i:
                stems.append(Path(name_i).stem)
        stems = list(dict.fromkeys(stems))

        for cache_dir in self.cache_dirs:
            if not os.path.isdir(cache_dir):
                continue
            for stem in stems:
                direct_path = os.path.join(cache_dir, f"{stem}.pt")
                if os.path.isfile(direct_path):
                    return direct_path
            for stem in stems:
                hits = sorted(Path(cache_dir).glob(f"{stem}_*.pt"))
                if len(hits) > 0:
                    return str(hits[0])
        return None

    def load_cache_tensor(self, cache_path):
        cache_obj = torch.load(cache_path, map_location="cpu")

        if isinstance(cache_obj, dict):
            cached_type = cache_obj.get("sam_model_type", None)
            if cached_type is not None and str(cached_type) != str(sam_model_type):
                raise ValueError(
                    f"Feature cache backbone mismatch: cache={cached_type}, "
                    f"script={sam_model_type}, path={cache_path}"
                )
            sam_feat = None
            for cache_key in ["feat", "sam_feat", "features", "image_embedding", "image_embeddings"]:
                if cache_key in cache_obj:
                    sam_feat = cache_obj[cache_key]
                    break
            if sam_feat is None:
                raise KeyError(
                    f"No feature tensor key found in cache: {cache_path}. "
                    f"Available keys: {list(cache_obj.keys())}"
                )
        else:
            sam_feat = cache_obj

        if not torch.is_tensor(sam_feat):
            sam_feat = torch.as_tensor(sam_feat)

        # 单张图缓存可能保存为 [1, C, H, W] 或 [C, H, W]。
        if sam_feat.ndim == 4 and sam_feat.shape[0] == 1:
            sam_feat = sam_feat.squeeze(0)
        if sam_feat.ndim != 3:
            raise ValueError(
                f"Cached feature must be [C,H,W] or [1,C,H,W], got "
                f"{tuple(sam_feat.shape)} from {cache_path}"
            )
        if sam_feat.shape[0] != 256:
            raise ValueError(
                f"SAM image embedding should have 256 channels, got "
                f"{tuple(sam_feat.shape)} from {cache_path}"
            )
        return sam_feat.contiguous()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_id = self.keys[idx]
        img_info = self.coco.imgs[int(img_id)]
        file_name = img_info["file_name"]

        img_path, real_name = self.find_image_file_lvis(file_name)
        if img_path is None:
            raise FileNotFoundError(file_name)

        rgb = pil_read_rgb(img_path)
        h, w = rgb.shape[:2]

        gt_orig = self.get_bboxes_from_coco_id(int(img_id), w, h)
        gt_orig = [b for b in gt_orig if box_area_xyxy(b) >= 4]
        if len(gt_orig) == 0:
            gt_orig = [[0, 0, 10, 10]]

        count_rec = self.get_count_record(file_name, real_name)
        tpl_orig = self.get_template_boxes_from_count_record(count_rec) if count_rec is not None else []
        tpl_orig = [clamp_box_xyxy(b, w, h) for b in tpl_orig]
        tpl_orig = [b for b in tpl_orig if box_area_xyxy(b) >= 4]

        if len(tpl_orig) == 0:
            # 只作为兜底，正常不应该走到这里
            tpl_orig = [gt_orig[0]]

        if self.training:
            template_orig = random.choice(tpl_orig)
        else:
            template_orig = tpl_orig[0]

        padded, meta = resize_longest_side_and_pad_rgb(rgb, target_size=sam_img_size)
        gt_sam = boxes_original_to_sam_padded(gt_orig, meta, target_size=sam_img_size)
        tpl_sam = boxes_original_to_sam_padded(tpl_orig, meta, target_size=sam_img_size)
        template_sam = boxes_original_to_sam_padded([template_orig], meta, target_size=sam_img_size)

        if len(gt_sam) == 0:
            gt_sam = [[0, 0, 10, 10]]
        if len(tpl_sam) == 0:
            tpl_sam = [gt_sam[0]]
        if len(template_sam) == 0:
            template_sam = [tpl_sam[0]]

        cache_path = self.find_cache_file(real_name, file_name)
        if cache_path is None:
            raise FileNotFoundError(
                f"SAM feature cache was not found for image: {real_name}. "
                f"Searched directories: {self.cache_dirs}"
            )
        sam_feat = self.load_cache_tensor(cache_path)

        return {
            "image": padded,
            "image_id": Path(real_name).stem,
            "file_name": real_name,
            "template_box": template_sam[0],
            "template_candidates": tpl_sam,
            "gt_boxes": gt_sam,
            "meta": meta,
            "sam_feat": sam_feat,
            "feature_cache_path": cache_path,
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
        roi = crop_feature_roi_to_fixed(feat, template_boxes_sam, out_size=self.roi_size, sam_size=sam_img_size)

        c = self.roi_size // 2
        center_feat = roi[:, :, c-1:c+2, c-1:c+2].mean(dim=(-2, -1))
        context_feat = roi.mean(dim=(-2, -1))
        size_feat = self.build_size_feat(template_boxes_sam, feat.device, feat.dtype)

        visual_logits = self.visual_mlp(torch.cat([center_feat, context_feat], dim=1))
        size_logits = self.size_mlp(size_feat)
        branch_weights = F.softmax(visual_logits + size_logits, dim=1)

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



# TemplateContextAttentionGate removed in the final no-attention configuration.

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
    """
    Final no-attention cached-feature version:
        - 保留 template-conditioned dynamic context aggregation；
        - 保留 template center cosine similarity；
        - 完全移除 TemplateContextAttentionGate 与 attention suppression；
        - 训练直接读取提前生成的 SAM embedding cache；
        - 3-shot 评估时仅将 cached SAM feature 载入 GPU 一次并复用于三个 template。
    """
    def __init__(self):
        super().__init__()
        # SAM encoder is not instantiated here: embeddings are loaded from .pt cache.
        # This removes repeated encoder computation and its GPU memory allocation.

        self.dynamic_context = TemplateDynamicContextAggregation(
            256, project_dim, template_roi_size, num_context_branches
        )
        self.decoder = HighResolutionDecoder(project_dim, decoder_mid_dim, decoder_out_dim)

        # 输入仅为：F_hr(64) + sim_center(1)
        self.head = PredictionHead(decoder_out_dim + 1, decoder_out_dim)

    def load_cached_sam_features(self, samples, device):
        """
        从 Dataset 返回的 CPU cached embeddings 组成 batch 并送入 GPU。
        不实例化、不运行 SAM image encoder。
        """
        features = []
        for sample in samples:
            sam_feat = sample["sam_feat"]
            if not torch.is_tensor(sam_feat):
                sam_feat = torch.as_tensor(sam_feat)
            if cached_feature_to_float32:
                sam_feat = sam_feat.float()
            features.append(sam_feat.to(device, non_blocking=False).contiguous())
        return torch.stack(features, dim=0)

    def forward(self, samples, sam_feat=None):
        template_boxes = [s["template_box"] for s in samples]

        if sam_feat is None:
            device = next(self.parameters()).device
            sam_feat = self.load_cached_sam_features(samples, device)

        ctx, info = self.dynamic_context(sam_feat, template_boxes)
        feat_hr = self.decoder(ctx)

        # 中心相似：在经过 template-conditioned aggregation 的特征上寻找模板中心。
        q = sample_template_center_query(feat_hr, template_boxes)
        sim_center = (
            F.normalize(feat_hr, dim=1)
            * F.normalize(q, dim=1).view(q.shape[0], q.shape[1], 1, 1)
        ).sum(dim=1, keepdim=True)

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
    """
    No-attention objective:
        center focal loss + offset classification + size regression + GIoU.
    attention suppression 与 attention auxiliary loss 均不参与训练。
    """
    center_loss = center_focal_loss(outputs["center_logits"], targets["center"])
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

    total = (
        lambda_center * center_loss
        + lambda_offset_cls * offset_loss
        + lambda_size * size_loss
        + lambda_giou * giou
    )
    return total, {
        "total": float(total.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
        "attn": float(attn_loss.detach().cpu()),
        "offset": float(offset_loss.detach().cpu()),
        "size": float(size_loss.detach().cpu()),
        "giou": float(giou.detach().cpu()),
        "pos": npos,
    }


def decode_predictions(outputs, samples, score_thr=0.25, apply_nms=True):
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
        if apply_nms and boxes_np.shape[0] > 0:
            keep = numpy_nms(boxes_np, scores_np, iou_thresh=nms_iou_thresh, max_dets=max_dets_per_image)
            boxes_np = boxes_np[keep]
            scores_np = scores_np[keep]
        boxes_orig = boxes_sam_padded_to_original(boxes_np.tolist(), samples[b]["meta"])
        results.append({"boxes": np.asarray(boxes_orig, np.float32), "scores": scores_np.astype(np.float32)})
    return results


@torch.no_grad()
def decode_predictions_three_shot(model, samples, score_thr=0.25):
    """
    Three-shot inference protocol:
        1) 对每张图最多使用前三个 official count template；
        2) 对每个 template 分别执行 detector forward；
        3) 每个 shot 的预测不提前做 NMS；
        4) 将所有 shot 的候选合并后，统一执行一次 NMS。

    图像的缓存 SAM feature 在一个 batch 内只加载到 GPU 一次，并在不同 shot 之间复用。
    """
    batch_n = len(samples)
    merged_boxes = [[] for _ in range(batch_n)]
    merged_scores = [[] for _ in range(batch_n)]
    used_shots = [0 for _ in range(batch_n)]

    device = next(model.parameters()).device
    shared_sam_feat = model.load_cached_sam_features(samples, device)

    for shot_idx in range(int(num_eval_shots)):
        shot_samples = []
        original_indices = []

        for sample_idx, sample in enumerate(samples):
            candidates = sample.get("template_candidates", [])
            if shot_idx >= len(candidates):
                continue

            shot_sample = dict(sample)
            shot_sample["template_box"] = candidates[shot_idx]
            shot_samples.append(shot_sample)
            original_indices.append(sample_idx)
            used_shots[sample_idx] += 1

        if len(shot_samples) == 0:
            continue

        feat_indices = torch.tensor(
            original_indices, device=shared_sam_feat.device, dtype=torch.long
        )
        shot_sam_feat = shared_sam_feat.index_select(0, feat_indices)

        outputs = model(shot_samples, sam_feat=shot_sam_feat)
        shot_preds = decode_predictions(
            outputs, shot_samples, score_thr=score_thr, apply_nms=False
        )

        for local_i, pred in enumerate(shot_preds):
            source_i = original_indices[local_i]
            if pred["boxes"].shape[0] > 0:
                merged_boxes[source_i].append(pred["boxes"])
                merged_scores[source_i].append(pred["scores"])

        del outputs, shot_preds, shot_sam_feat

    results = []
    for sample_idx in range(batch_n):
        if len(merged_boxes[sample_idx]) == 0:
            results.append({
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
            })
            continue

        boxes_np = np.concatenate(merged_boxes[sample_idx], axis=0).astype(np.float32)
        scores_np = np.concatenate(merged_scores[sample_idx], axis=0).astype(np.float32)

        keep = numpy_nms(
            boxes_np,
            scores_np,
            iou_thresh=nms_iou_thresh,
            max_dets=max_dets_per_image,
        )
        results.append({
            "boxes": boxes_np[keep].astype(np.float32),
            "scores": scores_np[keep].astype(np.float32),
        })

    return results, used_shots



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
        target_iscrowd = torch.zeros((target_boxes.shape[0],), dtype=torch.int64)

        pred_list.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
        target_list.append({
            "boxes": target_boxes,
            "labels": target_labels,
            "iscrowd": target_iscrowd,
        })
    return pred_list, target_list


@torch.no_grad()
def evaluate(model, loader, device, score_thr=None, trace_path=None, epoch=None):
    """
    Fast epoch-level evaluation for checkpoint selection.
    Each validation/test image uses only its first official template box (1-shot).
    """
    model.eval()
    if score_thr is None:
        score_thr = ap_score_thresh

    metric = make_map_metric()
    total_gt = 0
    total_pred = 0

    def trace(msg):
        if trace_path is not None:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        print(msg, flush=True)

    trace(
        f"epoch {epoch}: eval_start protocol=1-shot_checkpoint_selection "
        f"score_thr={score_thr} nms_iou={nms_iou_thresh}"
    )

    for eval_i, samples in enumerate(loader, start=1):
        image_ids = [s["image_id"] for s in samples]
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(
                f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} "
                f"batch_size={len(samples)} image_ids={image_ids[:3]} before_1shot_forward"
            )

        outputs = model(samples)
        preds = decode_predictions(outputs, samples, score_thr=score_thr)

        pred_list, target_list = tensors_for_map_from_preds(preds, samples)
        metric.update(pred_list, target_list)

        for sample, pred in zip(samples, preds):
            total_gt += len(sample["gt_boxes"])
            total_pred += len(pred["boxes"])

        del outputs, preds, pred_list, target_list

        if torch.cuda.is_available() and eval_i % eval_empty_cache_every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(
                f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} "
                f"after_metric_update total_gt={total_gt} total_pred={total_pred}"
            )

    result = metric.compute()
    ap = max(0.0, safe_float_metric(result.get("map", torch.tensor(0.0))))
    ap50 = max(0.0, safe_float_metric(result.get("map_50", torch.tensor(0.0))))
    ap75 = max(0.0, safe_float_metric(result.get("map_75", torch.tensor(0.0))))

    trace(
        f"epoch {epoch}: eval_done protocol=1-shot AP={ap:.6f} "
        f"AP50={ap50:.6f} AP75={ap75:.6f}"
    )
    return {
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "GT": total_gt,
        "Pred": total_pred,
        "NumPredForAP": total_pred,
        "score_thr_for_ap": float(score_thr),
        "evaluation_protocol": "1-shot_checkpoint_selection",
        "nms_iou_thresh": float(nms_iou_thresh),
    }


@torch.no_grad()
def evaluate_three_shot(model, loader, device, score_thr=None, trace_path=None, epoch=None):
    model.eval()
    if score_thr is None:
        score_thr = ap_score_thresh

    metric = make_map_metric()
    total_gt = 0
    total_pred = 0
    total_used_shots = 0
    total_eval_images = 0

    def trace(msg):
        if trace_path is not None:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        print(msg, flush=True)

    trace(
        f"epoch {epoch}: eval_start protocol=final_{num_eval_shots}-shot "
        f"score_thr={score_thr} nms_iou={nms_iou_thresh}"
    )

    for eval_i, samples in enumerate(loader, start=1):
        image_ids = [s["image_id"] for s in samples]
        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(
                f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} "
                f"batch_size={len(samples)} image_ids={image_ids[:3]} before_3shot_forward"
            )

        preds, used_shots = decode_predictions_three_shot(model, samples, score_thr)

        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(
                f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} "
                f"after_3shot_decode used_shots={used_shots[:3]}"
            )

        pred_list, target_list = tensors_for_map_from_preds(preds, samples)
        metric.update(pred_list, target_list)

        for s, p, n_shots in zip(samples, preds, used_shots):
            total_gt += len(s["gt_boxes"])
            total_pred += len(p["boxes"])
            total_used_shots += int(n_shots)
            total_eval_images += 1

        del preds, pred_list, target_list

        if torch.cuda.is_available() and eval_i % eval_empty_cache_every == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if eval_i == 1 or eval_i % eval_trace_every == 0:
            trace(
                f"epoch {epoch}: eval_iter {eval_i}/{len(loader)} "
                f"after_metric_update total_gt={total_gt} total_pred={total_pred}"
            )

    avg_used_shots = total_used_shots / max(total_eval_images, 1)
    trace(
        f"epoch {epoch}: eval_loop_done total_gt={total_gt} total_pred={total_pred} "
        f"avg_used_shots={avg_used_shots:.4f}"
    )
    trace(f"epoch {epoch}: before_metric_compute")

    result = metric.compute()
    ap = safe_float_metric(result.get("map", torch.tensor(0.0)))
    ap50 = safe_float_metric(result.get("map_50", torch.tensor(0.0)))
    ap75 = safe_float_metric(result.get("map_75", torch.tensor(0.0)))
    ap = 0.0 if ap < 0 else ap
    ap50 = 0.0 if ap50 < 0 else ap50
    ap75 = 0.0 if ap75 < 0 else ap75

    trace(
        f"epoch {epoch}: eval_done AP={ap:.6f} AP50={ap50:.6f} AP75={ap75:.6f} "
        f"avg_used_shots={avg_used_shots:.4f}"
    )
    return {
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "GT": total_gt,
        "Pred": total_pred,
        "NumPredForAP": total_pred,
        "score_thr_for_ap": score_thr,
        "inference_shots": int(num_eval_shots),
        "avg_used_shots": float(avg_used_shots),
    }


def draw_vis(samples, preds, out_dir, max_images=20):
    ensure_dir(out_dir)

    for i, (s, p) in enumerate(zip(samples, preds)):
        if i >= max_images:
            break
        pil = Image.fromarray(s["image"].copy()).convert("RGB")
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



def run_final_three_shot_on_selected_checkpoints(
    model,
    loader,
    device,
    best_ap_path,
    best_ap50_path,
    stage_log_path,
):
    """
    Training ends with only two final 3-shot evaluations:
        - the checkpoint selected by 1-shot AP;
        - the checkpoint selected by 1-shot AP50.

    If both selections come from the same epoch, 3-shot evaluation is performed once
    and the same metric result is recorded for both selection criteria.
    """
    final_dir = os.path.join(save_dir, "final_3shot_selected_checkpoints")
    ensure_dir(final_dir)

    selected = [
        ("best_AP_1shot", best_ap_path),
        ("best_AP50_1shot", best_ap50_path),
    ]
    summary = {}
    evaluated_by_epoch = {}

    for selected_by, checkpoint_path in selected:
        if not os.path.isfile(checkpoint_path):
            print(f"[Final 3-shot] checkpoint not found, skip: {checkpoint_path}", flush=True)
            summary[selected_by] = {
                "error": "checkpoint_not_found",
                "checkpoint": checkpoint_path,
            }
            continue

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        selected_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
        deduplicate_key = selected_epoch if selected_epoch >= 0 else checkpoint_path

        if deduplicate_key in evaluated_by_epoch:
            cached_result = dict(evaluated_by_epoch[deduplicate_key])
            cached_result["selected_by"] = selected_by
            cached_result["checkpoint"] = checkpoint_path
            cached_result["reused_from_same_epoch"] = True
            summary[selected_by] = cached_result
            print(
                f"[Final 3-shot] {selected_by}: same epoch={selected_epoch} as another "
                f"selected checkpoint, reuse existing 3-shot result.",
                flush=True,
            )
            continue

        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[Final 3-shot] evaluating {selected_by}: epoch={selected_epoch}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}",
            flush=True,
        )

        metrics = evaluate_three_shot(
            model,
            loader,
            device,
            score_thr=ap_score_thresh,
            trace_path=stage_log_path,
            epoch=f"final_{selected_by}",
        )
        selection_metrics = checkpoint.get("metrics", {}) if isinstance(checkpoint, dict) else {}
        result_record = {
            **metrics,
            "selected_by": selected_by,
            "checkpoint": checkpoint_path,
            "selected_epoch": selected_epoch,
            "selection_protocol": "1-shot",
            "selection_AP": float(selection_metrics.get("AP", -1.0)),
            "selection_AP50": float(selection_metrics.get("AP50", -1.0)),
            "selection_AP75": float(selection_metrics.get("AP75", -1.0)),
            "final_evaluation_protocol": f"{num_eval_shots}-shot",
            "reused_from_same_epoch": False,
        }
        summary[selected_by] = result_record
        evaluated_by_epoch[deduplicate_key] = dict(result_record)

        result_path = os.path.join(final_dir, f"metrics_3shot_from_{selected_by}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_record, f, ensure_ascii=False, indent=2)
        print(
            f"[Final 3-shot] {selected_by}: AP={metrics['AP']:.4f} "
            f"AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f}",
            flush=True,
        )

    summary_path = os.path.join(final_dir, "final_3shot_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Final 3-shot] summary saved: {summary_path}", flush=True)
    return summary


def train():
    set_seed(seed)
    if debug_anomaly:
        torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = False
    ensure_dir(save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*80)
    print(f"TDCC-SAM FSCD-LVIS {fscd_lvis_subset} Count-Template Fully Box-Supervised Detector")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count visible:", torch.cuda.device_count())
    print("save_dir:", save_dir)
    print("SAM feature cache root:", sam_feature_cache_root)
    print("Cached encoder mode: True (SAM image_encoder will NOT run)")
    print("Training protocol: random 1-shot")
    print("Epoch evaluation protocol: 1-shot, for checkpoint selection only")
    print(f"Final evaluation protocol: {num_eval_shots}-shot, only for best AP / best AP50 checkpoints")
    print("="*80)

    train_set = FSCDLVISInstancesDataset(fscd_lvis_root, split=train_split, training=True)
    val_set = FSCDLVISInstancesDataset(fscd_lvis_root, split=eval_split, training=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=eval_batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False, drop_last=False)

    model = TDCCDetector().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    start_epoch, resumed, resumed_from = try_resume_training(model, opt, device)

    config = {k: v for k, v in globals().items() if k in ["fscd_lvis_root","train_split","eval_split","sam_checkpoint","sam_model_type","sam_img_size","pred_size","project_dim","batch_size","eval_batch_size","lr","weight_decay","lambda_center","lambda_attn","lambda_offset_cls","lambda_size","lambda_giou","attention_radius_ratio","ap_score_thresh","ap_max_dets_per_image","max_dets_per_image","nms_iou_thresh","auto_resume","resume_checkpoint_path","count_annotation_files","max_count_templates","sam_feature_cache_root","extra_sam_feature_cache_roots","strict_feature_cache","cached_feature_to_float32","num_eval_shots","fscd_lvis_subset"]}
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log_jsonl_path = os.path.join(save_dir, "train_log_epoch_eval_1shot.jsonl")
    epoch_csv_path = os.path.join(save_dir, "epoch_metrics_selection_1shot.csv")
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

    best_ap = -1.0
    best_ap50 = -1.0
    best_ap_path = os.path.join(save_dir, "best_model_by_AP_1shot.pth")
    best_ap50_path = os.path.join(save_dir, "best_model_by_AP50_1shot.pth")

    # 断点续训时，分别恢复基于 1-shot AP 与 AP50 保存的最佳值。
    for selection_name, checkpoint_path, metric_key in [
        ("AP", best_ap_path, "AP"),
        ("AP50", best_ap50_path, "AP50"),
    ]:
        if os.path.isfile(checkpoint_path):
            try:
                selection_ckpt = torch.load(checkpoint_path, map_location="cpu")
                selection_metrics = selection_ckpt.get("metrics", {})
                metric_value = float(selection_metrics.get(metric_key, -1.0))
                if metric_key == "AP":
                    best_ap = metric_value
                else:
                    best_ap50 = metric_value
                print(
                    f"[Resume] existing best_model_by_{selection_name}_1shot.pth: "
                    f"{metric_key}={metric_value:.4f}",
                    flush=True,
                )
            except Exception as e:
                print(f"[Resume] failed to read {checkpoint_path}: {repr(e)}", flush=True)

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
            out = model(samples)
            targets = build_targets(samples, device)
            loss, info = compute_loss(out, targets)
            loss.backward()
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            opt.step()
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
                    "selected_by": "AP_1shot",
                }, best_ap_path)
                print(
                    f"[Save Best Model by 1-shot AP] AP={best_ap:.4f} "
                    f"AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f}",
                    flush=True
                )

            if metrics["AP50"] > best_ap50:
                best_ap50 = metrics["AP50"]
                torch.save({
                    "epoch": epoch,
                    "model": get_trainable_state_dict(model),
                    "optimizer": opt.state_dict(),
                    "config": config,
                    "metrics": metrics,
                    "selected_by": "AP50_1shot",
                }, best_ap50_path)
                print(
                    f"[Save Best Model by 1-shot AP50] AP={metrics['AP']:.4f} "
                    f"AP50={best_ap50:.4f} AP75={metrics['AP75']:.4f}",
                    flush=True
                )

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
                    out_vis = model([vs])
                    pred_vis = decode_predictions(out_vis, [vs], score_thresh)
                    preds.extend(pred_vis)
                    del out_vis, pred_vis
            out_dir = os.path.join(save_dir, "visuals_1shot_during_training", f"epoch_{epoch:03d}")
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


    print("\n" + "=" * 80, flush=True)
    print("[Training complete] Run final 3-shot evaluation only on selected checkpoints.", flush=True)
    print("=" * 80, flush=True)
    run_final_three_shot_on_selected_checkpoints(
        model=model,
        loader=val_loader,
        device=device,
        best_ap_path=best_ap_path,
        best_ap50_path=best_ap50_path,
        stage_log_path=stage_log_path,
    )



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
    # FSCD-LVIS uses count boxes as template candidates.
    globals()["max_count_templates"] = max(int(globals().get("max_count_templates", 5)), int(NUM_EVAL_SHOTS))
    return FSCDLVISInstancesDataset(
        fscd_lvis_root,
        split=eval_split,
        training=False,
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
    if not True:
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
