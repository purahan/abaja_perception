#!/usr/bin/env python3
"""
Fixes the coco_subset export into a proper Ultralytics-ready dataset:
  1. Consolidates the (buggy) images/val + labels/val subfolders from both
     train/ and validation/ into one dataset root with images/train,
     images/val, labels/train, labels/val.
  2. Remaps class ids from FiftyOne's local export order (alphabetical:
     bicyclist=0, car=1, pedestrian=2, two_wheeler=3) to the master ids
     in config/classes.yaml (bicyclist=11, car=10, pedestrian=9,
     two_wheeler=12), so label files agree with the rest of the pipeline.

Usage:
    python3 fix_coco_subset.py --src ./coco_subset --dst ./dataset_merged
"""

import argparse
import shutil
from pathlib import Path

# FiftyOne's local id (alphabetical order of the 4 classes we exported)
# -> the master id used everywhere else (config/classes.yaml)
LOCAL_TO_MASTER_ID = {
    0: 11,  # bicyclist
    1: 10,  # car
    2: 9,   # pedestrian
    3: 12,  # two_wheeler
}


def remap_label_file(src_path: Path, dst_path: Path):
    lines_out = []
    with open(src_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            local_id = int(parts[0])
            master_id = LOCAL_TO_MASTER_ID.get(local_id)
            if master_id is None:
                print(f'WARNING: unknown local class id {local_id} in {src_path}, skipping line')
                continue
            parts[0] = str(master_id)
            lines_out.append(' '.join(parts))
    dst_path.write_text('\n'.join(lines_out) + ('\n' if lines_out else ''))


def process_split(src_root: Path, split_name_in: str, split_name_out: str, dst_root: Path):
    images_src = src_root / split_name_in / 'images' / 'val'
    labels_src = src_root / split_name_in / 'labels' / 'val'

    images_dst = dst_root / 'images' / split_name_out
    labels_dst = dst_root / 'labels' / split_name_out
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    if not images_src.exists():
        print(f'WARNING: expected {images_src} not found, skipping this split')
        return 0

    count = 0
    for img_path in images_src.glob('*'):
        shutil.copy2(img_path, images_dst / img_path.name)
        label_src = labels_src / (img_path.stem + '.txt')
        label_dst = labels_dst / (img_path.stem + '.txt')
        if label_src.exists():
            remap_label_file(label_src, label_dst)
        else:
            # image with no annotated instances of our classes - still
            # needs an empty label file so Ultralytics treats it as
            # background rather than skipping/misaligning it
            label_dst.write_text('')
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='./coco_subset')
    parser.add_argument('--dst', default='./dataset_merged')
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    n_train = process_split(src_root, 'train', 'train', dst_root)
    n_val = process_split(src_root, 'validation', 'val', dst_root)

    print(f'\nWrote {n_train} train images/labels and {n_val} val images/labels to {dst_root}')
    print(f'Class ids remapped: {LOCAL_TO_MASTER_ID}')
    print('\nNext: as you add synthetic sign/light/barricade data and IDD '
          'cow data, copy them into this same dst images/train, images/val, '
          'labels/train, labels/val structure using the SAME master ids '
          'from config/classes.yaml, then write one data.yaml covering all '
          '15 classes pointing at this merged folder.')


if __name__ == '__main__':
    main()
