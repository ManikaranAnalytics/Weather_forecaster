"""
EfficientNet-B0/B3 cloud patch scorer — validates detections and finds horizon clouds.

Uses ImageNet-pretrained EfficientNet embeddings with per-frame sky/cloud prototypes
(no extra training weights required).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

_MODEL_CACHE = {}
_PROTO_CACHE = {}


def _get_variant_name(variant: str = "B0") -> str:
    v = (variant or "B0").upper().replace("EFFICIENTNET", "").strip()
    return v if v in ("B0", "B3") else "B0"


def load_efficientnet(variant: str = "B0"):
    """Load EfficientNet feature extractor (cached)."""
    v = _get_variant_name(variant)
    if v in _MODEL_CACHE:
        return _MODEL_CACHE[v]

    from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.models import Model

    if v == "B3":
        base = EfficientNetB3(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    else:
        base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

    _MODEL_CACHE[v] = (base, preprocess_input)
    return base, preprocess_input


def _embed_patches_batch(patches_bgr: List[np.ndarray], model, preprocess_input) -> List[Optional[np.ndarray]]:
    """Batch embed patches — much faster than one-by-one predict."""
    if not patches_bgr:
        return []
    batch = []
    valid_idx = []
    for i, patch in enumerate(patches_bgr):
        if patch is None or patch.size == 0:
            continue
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        batch.append(preprocess_input(rgb.astype(np.float32)))
        valid_idx.append(i)
    if not batch:
        return [None] * len(patches_bgr)
    x = np.stack(batch, axis=0)
    embs = model.predict(x, verbose=0)
    out = [None] * len(patches_bgr)
    for j, i in enumerate(valid_idx):
        e = embs[j]
        norm = np.linalg.norm(e)
        out[i] = e / norm if norm > 1e-8 else e
    return out


def _embed_patch(patch_bgr: np.ndarray, model, preprocess_input) -> Optional[np.ndarray]:
    res = _embed_patches_batch([patch_bgr], model, preprocess_input)
    return res[0] if res else None


def _build_prototypes(sky_bgr: np.ndarray, model, preprocess_input) -> Tuple[np.ndarray, np.ndarray]:
    """Build cloud vs sky embedding prototypes (few samples, batched)."""
    H, W = sky_bgr.shape[:2]
    gray = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)

    sky_patches, cloud_patches = [], []

    for x in range(0, W - 64, max(W // 3, 1)):
        p = sky_bgr[0:min(80, H), x : x + 64]
        if p.shape[0] >= 32 and p.shape[1] >= 32:
            hue = int(np.median(hsv[0 : p.shape[0], x : x + 64, 0]))
            if 95 <= hue <= 140:
                sky_patches.append(p)

    step = max(64, min(H, W) // 5)
    for y in range(0, H - step, step):
        for x in range(0, W - step, step):
            patch = sky_bgr[y : y + step, x : x + step]
            g = gray[y : y + step, x : x + step]
            s = hsv[y : y + step, x : x + step, 1]
            if float(np.mean(g)) > 140 and float(np.mean(s)) < 80:
                cloud_patches.append(patch)
                if len(cloud_patches) >= 8:
                    break
        if len(cloud_patches) >= 8:
            break

    sky_embs = [e for e in _embed_patches_batch(sky_patches[:6], model, preprocess_input) if e is not None]
    cloud_embs = [e for e in _embed_patches_batch(cloud_patches[:8], model, preprocess_input) if e is not None]

    if not cloud_embs:
        dim = model.output_shape[-1]
        cloud_proto = np.zeros(int(dim), dtype=np.float32)
    else:
        cloud_proto = np.mean(np.stack(cloud_embs, axis=0), axis=0)
        cloud_proto /= max(np.linalg.norm(cloud_proto), 1e-8)

    if not sky_embs:
        sky_proto = cloud_proto.copy()
    else:
        sky_proto = np.mean(np.stack(sky_embs, axis=0), axis=0)
        sky_proto /= max(np.linalg.norm(sky_proto), 1e-8)

    return cloud_proto, sky_proto


def score_cloud_patch(
    patch_bgr: np.ndarray,
    model,
    preprocess_input,
    cloud_proto: np.ndarray,
    sky_proto: np.ndarray,
) -> float:
    """Return cloud confidence 0–1 (higher = more likely real cloud)."""
    emb = _embed_patch(patch_bgr, model, preprocess_input)
    if emb is None:
        return 0.0
    cloud_sim = float(np.dot(emb, cloud_proto))
    sky_sim = float(np.dot(emb, sky_proto))
    # Also use classical cues inside patch
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    bright = float(np.mean(gray)) / 255.0
    low_sat = 1.0 - float(np.mean(hsv[:, :, 1])) / 255.0
    cv_score = min(1.0, bright * 0.55 + low_sat * 0.45)

    raw = 0.45 * (cloud_sim - sky_sim + 1.0) / 2.0 + 0.55 * cv_score
    return float(np.clip(raw, 0.0, 1.0))


def is_edge_false_positive(x: int, y: int, w: int, h: int, frame_w: int, sky_h: int) -> bool:
    """Reject thin vertical edge strips (lens vignette / border artifacts)."""
    margin = max(14, int(frame_w * 0.025))
    aspect = h / max(w, 1)
    if x <= margin and aspect > 1.8:
        return True
    if x + w >= frame_w - margin and aspect > 1.8:
        return True
    if w < frame_w * 0.08 and h > sky_h * 0.55:
        return True
    if w * h > (sky_h * frame_w) * 0.32:
        return True
    if w > frame_w * 0.62:
        return True
    return False


def detect_horizon_clouds(frame_bgr: np.ndarray, sky_h: int, min_area: int = 500) -> List[Tuple]:
    """
    Extra pass for lower horizon clouds (CLAHE + adaptive threshold).
    Targets clouds circled near bottom-left / bottom-right.
    """
    sky = frame_bgr[:sky_h, :]
    H, W = sky.shape[:2]
    y0 = int(H * 0.45)
    band = sky[y0:, :].copy()
    if band.size == 0:
        return []

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    adapt = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -7
    )

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv[:, :, 0], 90, 140)
    clear_blue = cv2.bitwise_and(blue, cv2.threshold(hsv[:, :, 1], 50, 255, cv2.THRESH_BINARY)[1])
    mask = cv2.bitwise_and(adapt, cv2.bitwise_not(clear_blue))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((14, 14), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    boxes = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = (H * W) * 0.35
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y + y0, w, h, None))
    return boxes


def _nms_boxes(boxes: List[Tuple], iou_thresh: float = 0.40) -> List[Tuple]:
    if len(boxes) <= 1:
        return boxes
    rects = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
    scores = [float(b[2] * b[3]) for b in boxes]
    idxs = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.1, nms_threshold=iou_thresh)
    if len(idxs) == 0:
        return boxes
    flat = idxs.flatten() if hasattr(idxs, "flatten") else idxs
    return [boxes[int(i)] for i in flat]


def _cv_cloud_score(patch_bgr: np.ndarray) -> float:
    """Fast classical cloud score — no neural net."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    bright = float(np.mean(gray)) / 255.0
    low_sat = 1.0 - float(np.mean(hsv[:, :, 1])) / 255.0
    return float(np.clip(bright * 0.55 + low_sat * 0.45, 0.0, 1.0))


def refine_boxes_fast(frame_bgr: np.ndarray, sky_h: int, boxes: List[Tuple]) -> List[Tuple]:
    """Rule-only refine — no EfficientNet (for video frames)."""
    H, W = frame_bgr.shape[:2]
    sky = frame_bgr[:sky_h, :]
    kept = []
    horizon_y = int(sky_h * 0.55)
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        if is_edge_false_positive(x, y, w, h, W, sky_h):
            continue
        patch = sky[y : min(y + h, sky_h), x : min(x + w, W)]
        if patch.size == 0:
            continue
        thresh = 0.20 if y >= horizon_y else 0.28
        if _cv_cloud_score(patch) < thresh:
            continue
        mask = box[4] if len(box) > 4 else None
        kept.append((x, y, w, h, mask))
    return kept


def refine_boxes_efficientnet(
    frame_bgr: np.ndarray,
    sky_h: int,
    boxes: List[Tuple],
    variant: str = "B0",
    accept_thresh: float = 0.32,
    horizon_accept_thresh: float = 0.24,
) -> List[Tuple]:
    """Filter false positives and keep real clouds using EfficientNet + rules."""
    if not boxes:
        boxes = []

    H, W = frame_bgr.shape[:2]
    sky = frame_bgr[:sky_h, :]

    try:
        model, preprocess_input = load_efficientnet(variant)
        cloud_proto, sky_proto = _build_prototypes(sky, model, preprocess_input)
    except Exception:
        # Fallback: rule-based only
        cloud_proto = sky_proto = None
        model = preprocess_input = None

    kept = []
    horizon_y = int(sky_h * 0.55)

    # Batch-score all candidate patches in one predict call
    patches, meta = [], []
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        if is_edge_false_positive(x, y, w, h, W, sky_h):
            continue
        patch = sky[y : min(y + h, sky_h), x : min(x + w, W)]
        if patch.size == 0:
            continue
        patches.append(patch)
        meta.append(box)

    if model is not None and patches:
        embs = _embed_patches_batch(patches, model, preprocess_input)
        for box, patch, emb in zip(meta, patches, embs):
            x, y, w, h = box[0], box[1], box[2], box[3]
            thresh = horizon_accept_thresh if y >= horizon_y else accept_thresh
            if emb is not None:
                cloud_sim = float(np.dot(emb, cloud_proto))
                sky_sim = float(np.dot(emb, sky_proto))
                cv_score = _cv_cloud_score(patch)
                raw = 0.45 * (cloud_sim - sky_sim + 1.0) / 2.0 + 0.55 * cv_score
                score = float(np.clip(raw, 0.0, 1.0))
                if score < thresh:
                    continue
            mask = box[4] if len(box) > 4 else None
            kept.append((x, y, w, h, mask))
        return kept

    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        mask = box[4] if len(box) > 4 else None

        if is_edge_false_positive(x, y, w, h, W, sky_h):
            continue

        y2 = min(y + h, sky_h)
        x2 = min(x + w, W)
        patch = sky[y:y2, x:x2]
        if patch.size == 0:
            continue

        thresh = horizon_accept_thresh if y >= horizon_y else accept_thresh

        if model is not None:
            score = score_cloud_patch(patch, model, preprocess_input, cloud_proto, sky_proto)
            if score < thresh:
                continue
        else:
            g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if float(np.mean(g)) < 120:
                continue

        kept.append((x, y, w, h, mask))

    return kept


def merge_and_refine_detections(
    frame_bgr: np.ndarray,
    sky_h: int,
    classical_boxes: List[Tuple],
    variant: str = "B0",
    use_efficientnet: bool = True,
) -> List[Tuple]:
    """Merge classical + horizon detections, NMS, refine (fast or EfficientNet)."""
    H, W = frame_bgr.shape[:2]
    horizon_boxes = detect_horizon_clouds(frame_bgr, sky_h)
    all_boxes = list(classical_boxes) + horizon_boxes
    all_boxes = _nms_boxes(all_boxes)

    if use_efficientnet:
        refined = refine_boxes_efficientnet(frame_bgr, sky_h, all_boxes, variant=variant)
    else:
        refined = refine_boxes_fast(frame_bgr, sky_h, all_boxes)

    if refined:
        return refined

    fallback = []
    for box in classical_boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        if not is_edge_false_positive(x, y, w, h, W, sky_h):
            fallback.append(box)
    return fallback or horizon_boxes
