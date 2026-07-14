"""
Realistic cloud bounding boxes — tight contour fit, mask fill ratio, optical-flow tracking.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .efficientnet_cloud_detector import (
    detect_horizon_clouds,
    is_edge_false_positive,
    merge_and_refine_detections,
    _nms_boxes,
)


def build_cloud_mask(sky_bgr: np.ndarray) -> np.ndarray:
    """Cloud mask: bright + low-saturation regions only (not whole sky)."""
    H, W = sky_bgr.shape[:2]
    gray = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(sky_bgr, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Cloud = bright AND desaturated (intersection — avoids painting full sky)
    bright = cv2.threshold(enhanced, 142, 255, cv2.THRESH_BINARY)[1]
    low_sat = cv2.threshold(hsv[:, :, 1], 78, 255, cv2.THRESH_BINARY_INV)[1]
    high_val = cv2.threshold(hsv[:, :, 2], 112, 255, cv2.THRESH_BINARY)[1]
    cloud = cv2.bitwise_and(bright, low_sat)
    cloud = cv2.bitwise_and(cloud, high_val)

    # Remove clear blue sky
    blue_sky = cv2.inRange(hsv, (92, 48, 70), (128, 255, 210))
    cloud = cv2.bitwise_and(cloud, cv2.bitwise_not(blue_sky))

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, k_close, iterations=2)
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, k_open, iterations=1)

    # Drop tiny specks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cloud, connectivity=8)
    cleaned = np.zeros_like(cloud)
    min_px = max(350, int(H * W * 0.0008))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            cleaned[labels == i] = 255

    cleaned[int(H * 0.92) :, :] = 0
    return cleaned


def split_cloud_mask(mask: np.ndarray, min_split_area: int = 6000) -> np.ndarray:
    """Split wide merged cloud blobs using distance-transform peaks."""
    H, W = mask.shape[:2]
    out = np.zeros_like(mask)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_split_area:
            x, y, w, h = cv2.boundingRect(cnt)
            out[y : y + h, x : x + w] = cv2.bitwise_or(
                out[y : y + h, x : x + w], mask[y : y + h, x : x + w]
            )
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = mask[y : y + h, x : x + w].copy()
        # Erode slightly so separate cloud tops become distinct peaks
        eroded = cv2.erode(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
        if dist.max() < 4:
            out[y : y + h, x : x + w] = cv2.bitwise_or(out[y : y + h, x : x + w], roi)
            continue

        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        peak_thresh = 0.32 if w > W * 0.45 else 0.40
        _, sure_fg = cv2.threshold(dist, peak_thresh * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        if cv2.countNonZero(sure_fg) < 2:
            out[y : y + h, x : x + w] = cv2.bitwise_or(out[y : y + h, x : x + w], roi)
            continue

        sure_bg = cv2.dilate(roi, np.ones((9, 9), np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(roi_bgr, markers)

        wrote = False
        for lbl in np.unique(markers):
            if lbl <= 1:
                continue
            blob = np.zeros_like(roi)
            blob[markers == lbl] = 255
            blob = cv2.bitwise_and(blob, roi)
            if cv2.countNonZero(blob) < 600:
                continue
            out[y : y + h, x : x + w] = cv2.bitwise_or(out[y : y + h, x : x + w], blob)
            wrote = True
        if not wrote:
            out[y : y + h, x : x + w] = cv2.bitwise_or(out[y : y + h, x : x + w], roi)

    return out if cv2.countNonZero(out) > 0 else mask


def mask_to_tight_boxes(
    mask: np.ndarray,
    sky_h: int,
    frame_w: int,
    min_area: int = 800,
    max_area_ratio: float = 0.22,
    min_fill: float = 0.35,
) -> List[Tuple]:
    """Contour-based tight boxes with cloud-fill validation."""
    max_area = sky_h * frame_w * max_area_ratio
    boxes = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if is_edge_false_positive(x, y, w, h, frame_w, sky_h):
            continue

        box_area = max(w * h, 1)
        fill = area / box_area
        if fill < min_fill:
            continue

        # Per-blob mask for accurate overlay
        blob_mask = np.zeros((sky_h, frame_w), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

        # Shrink box to actual mask extent (tighter)
        ys, xs = np.where(blob_mask[y : y + h, x : x + w] > 0)
        if len(xs) == 0:
            continue
        tx1, ty1 = x + int(xs.min()), y + int(ys.min())
        tx2, ty2 = x + int(xs.max()), y + int(ys.max())
        tw, th = tx2 - tx1 + 1, ty2 - ty1 + 1
        if tw * th < min_area * 0.8:
            continue
        # Reject thin sliver boxes (not realistic clouds)
        if min(tw, th) / max(tw, th, 1) < 0.22:
            continue

        boxes.append((tx1, ty1, tw, th, blob_mask))

    return boxes


def filter_nested_boxes(boxes: List[Tuple], contain_thresh: float = 0.82) -> List[Tuple]:
    """Drop small boxes almost fully inside a larger box."""
    if len(boxes) <= 1:
        return boxes
    areas = [b[2] * b[3] for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep = []
    for i in order:
        x, y, w, h = boxes[i][:4]
        cx, cy = x + w / 2, y + h / 2
        nested = False
        for k in keep:
            kx, ky, kw, kh = boxes[k][:4]
            if kx <= cx <= kx + kw and ky <= cy <= ky + kh:
                inter_w = max(0, min(x + w, kx + kw) - max(x, kx))
                inter_h = max(0, min(y + h, ky + kh) - max(y, ky))
                if (inter_w * inter_h) / max(w * h, 1) >= contain_thresh:
                    nested = True
                    break
        if not nested:
            keep.append(i)
    return [boxes[i] for i in keep]


def detect_clouds_realistic(
    frame_bgr: np.ndarray,
    sky_h: int,
    use_efficientnet: bool = True,
    variant: str = "B0",
) -> List[Tuple]:
    """Main realistic detection entry."""
    sky = frame_bgr[:sky_h, :]
    W = frame_bgr.shape[1]
    mask = build_cloud_mask(sky)
    mask = split_cloud_mask(mask)
    boxes = mask_to_tight_boxes(mask, sky_h, W)

    if use_efficientnet:
        boxes = merge_and_refine_detections(
            frame_bgr, sky_h, boxes, variant=variant, use_efficientnet=True
        )
    else:
        boxes = merge_and_refine_detections(
            frame_bgr, sky_h, boxes, variant=variant, use_efficientnet=False
        )

    if len(boxes) < 2:
        horizon = detect_horizon_clouds(frame_bgr, sky_h, min_area=600)
        boxes = _nms_boxes(boxes + horizon)

    boxes = filter_nested_boxes(boxes)
    return boxes


def track_boxes_optical_flow(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    boxes: List[Tuple],
    sky_h: int,
) -> List[Tuple]:
    """Shift boxes using optical flow so they follow clouds between re-detections."""
    if prev_gray is None or not boxes:
        return boxes

    prev_sky = prev_gray[:sky_h, :]
    curr_sky = gray[:sky_h, :]
    if prev_sky.shape != curr_sky.shape:
        return boxes

    flow = cv2.calcOpticalFlowFarneback(
        prev_sky, curr_sky, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    H, W = curr_sky.shape[:2]
    tracked = []

    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        mask = box[4] if len(box) > 4 else None
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        roi = flow[y1:y2, x1:x2]
        if roi.size == 0:
            tracked.append(box)
            continue
        dx = float(np.median(roi[..., 0]))
        dy = float(np.median(roi[..., 1]))
        nx = int(round(x + dx))
        ny = int(round(y + dy))
        nx = max(0, min(nx, W - w))
        ny = max(0, min(ny, H - h))
        tracked.append((nx, ny, w, h, mask))

    return tracked


def smooth_box_speed(roi_speed_kmh: float, global_speed_kmh: float) -> float:
    """Keep per-box speed realistic — blend with global when ROI is noisy."""
    if global_speed_kmh <= 0:
        return max(0.0, roi_speed_kmh)
    if roi_speed_kmh <= 0.1:
        return global_speed_kmh
    ratio = roi_speed_kmh / global_speed_kmh
    if ratio > 1.45 or ratio < 0.45:
        return global_speed_kmh
    return 0.72 * global_speed_kmh + 0.28 * roi_speed_kmh
