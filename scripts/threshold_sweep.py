#!/usr/bin/env python3
"""
Sweep per-class confidence thresholds against the ACTUAL aBAJA scoring
rubric (not mAP), and write out the threshold set that maximizes expected
points per class:

    correct detect + correct class : +5
    detect, wrong/missing class     : +2
    false positive                  : -2
    false negative (ped/cow/cyclist): -3
    false negative (other classes)  : -2

Expects a YOLO-format validation set:
    val_images/*.jpg
    val_labels/*.txt   (class_id cx cy w h, normalized, one box per line)

Usage:
    python3 threshold_sweep.py --weights best.pt \
        --val-images /path/val/images --val-labels /path/val/labels \
        --config ../config/classes.yaml --out tuned_classes.yaml

This is a validation-set proxy for the live judged run — it won't be
perfectly identical to on-the-day scoring (judge discretion on flicker
etc. isn't modeled here), but it gets your starting thresholds aligned to
the actual cost function instead of generic precision/recall.
"""

import argparse
import glob
import os
import numpy as np
import yaml
from ultralytics import YOLO


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def load_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, w, h = map(float, parts)
            cls_id = int(cls_id)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def score_for_threshold(gt_by_class, preds_by_class, cls_id, threshold,
                         fn_penalty, iou_thresh=0.5):
    score = 0
    matched_gt = set()
    for img_id, conf, bbox, pred_cls in preds_by_class:
        if conf < threshold:
            continue
        gts = gt_by_class.get((img_id, cls_id), [])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gts):
            if (img_id, cls_id, j) in matched_gt:
                continue
            i = iou(bbox, gt_box)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thresh:
            score += 5  # correct detect + correct class
            matched_gt.add((img_id, cls_id, best_j))
        else:
            score -= 2  # false positive (no matching gt box for this class)

    total_gt = sum(len(v) for k, v in gt_by_class.items() if k[1] == cls_id)
    matched_count = len({m for m in matched_gt if m[1] == cls_id})
    misses = total_gt - matched_count
    score -= misses * fn_penalty
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--val-images', required=True)
    parser.add_argument('--val-labels', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='tuned_classes.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    id_to_name = {v['id']: k for k, v in cfg['classes'].items()}
    fn_penalties = {v['id']: v['fn_penalty'] for v in cfg['classes'].values()}

    model = YOLO(args.weights)
    image_paths = sorted(glob.glob(os.path.join(args.val_images, '*')))

    gt_by_class = {}
    all_preds = []  # (img_id, conf, bbox, cls_id)

    for img_id, img_path in enumerate(image_paths):
        result = model.predict(img_path, verbose=False, conf=0.01)[0]
        h, w = result.orig_shape
        label_path = os.path.join(
            args.val_labels,
            os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        for cls_id, box in load_yolo_labels(label_path, w, h):
            gt_by_class.setdefault((img_id, cls_id), []).append(box)

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = tuple(box.xyxy[0].tolist())
            all_preds.append((img_id, conf, xyxy, cls_id))

    tuned = {}
    thresholds_to_try = np.arange(0.10, 0.85, 0.05)

    for cls_id, name in id_to_name.items():
        preds_this_class = [p for p in all_preds if p[3] == cls_id]
        best_score, best_thresh = -1e9, 0.35
        for thresh in thresholds_to_try:
            s = score_for_threshold(
                gt_by_class, preds_this_class, cls_id, thresh,
                fn_penalties[cls_id])
            if s > best_score:
                best_score, best_thresh = s, thresh
        tuned[name] = {'conf_threshold': round(float(best_thresh), 2),
                        'expected_score': best_score}
        print(f'{name:20s} best_threshold={best_thresh:.2f}  '
              f'expected_score={best_score}')

    with open(args.out, 'w') as f:
        yaml.dump(tuned, f)
    print(f'\nWrote tuned thresholds to {args.out} — '
          f'merge conf_threshold values into config/classes.yaml')


if __name__ == '__main__':
    main()
