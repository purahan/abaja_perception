#!/usr/bin/env python3
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

COW_CLASS_ID = 13  # must match config/classes.yaml


def get_is_visible(elem) -> bool:
    for attr in elem.findall('attribute'):
        if attr.get('name') == 'is_visible':
            return attr.text.strip().lower() != 'false'
    return True


def polygon_to_bbox(points_str):
    points = []
    for pair in points_str.split(';'):
        x, y = pair.split(',')
        points.append((float(x), float(y)))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def parse_annotations(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = {}
    for image_elem in root.findall('image'):
        name = image_elem.get('name')
        img_w = float(image_elem.get('width'))
        img_h = float(image_elem.get('height'))
        boxes = []
        for box_elem in image_elem.findall('box'):
            if not get_is_visible(box_elem):
                continue
            x1 = float(box_elem.get('xtl'))
            y1 = float(box_elem.get('ytl'))
            x2 = float(box_elem.get('xbr'))
            y2 = float(box_elem.get('ybr'))
            boxes.append((x1, y1, x2, y2, img_w, img_h))
        for poly_elem in image_elem.findall('polygon'):
            if not get_is_visible(poly_elem):
                continue
            x1, y1, x2, y2 = polygon_to_bbox(poly_elem.get('points'))
            boxes.append((x1, y1, x2, y2, img_w, img_h))
        if boxes:
            results[name] = boxes
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--dst', default='./dataset_merged')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    xml_path = Path(args.xml)
    images_dir = Path(args.images_dir)
    dst = Path(args.dst)

    print(f'Parsing {xml_path}...')
    annotations = parse_annotations(xml_path)
    print(f'Found annotations for {len(annotations)} images')

    if not annotations:
        print('WARNING: no <box> or <polygon> elements found - paste me the XML structure.')
        return

    filenames = list(annotations.keys())
    random.Random(args.seed).shuffle(filenames)
    n_val = int(len(filenames) * args.val_fraction)
    val_set = set(filenames[:n_val])

    for split in ['train', 'val']:
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)

    written = 0
    missing_images = 0
    for filename, boxes in annotations.items():
        split = 'val' if filename in val_set else 'train'
        src_img = images_dir / Path(filename).name
        if not src_img.exists():
            missing_images += 1
            continue
        dst_img = dst / 'images' / split / Path(filename).name
        shutil.copy2(src_img, dst_img)
        lines = []
        for (x1, y1, x2, y2, img_w, img_h) in boxes:
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f'{COW_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        label_path = dst / 'labels' / split / (Path(Path(filename).name).stem + '.txt')
        label_path.write_text('\n'.join(lines) + '\n')
        written += 1

    print(f'\nWrote {written} cow images/labels into {dst}')
    if missing_images:
        print(f'WARNING: {missing_images} annotated filenames had no matching image file')


if __name__ == '__main__':
    main()
