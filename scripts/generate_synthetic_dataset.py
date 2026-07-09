#!/usr/bin/env python3
"""
Synthetic data generator for the 9 aBAJA classes with no usable public
dataset: 5 speed-limit sign values, 3 traffic-light states, steel
barricade.

Renders each object procedurally (PIL shapes/text), then augments with
random scale (simulating 5-25m distance per the event's designated area),
rotation, brightness/contrast jitter, blur, and composites onto a
background - either your own real photos (recommended once you have venue
access) or simple procedural backgrounds as a fallback.

Outputs directly into an existing Ultralytics-style dataset folder
(images/train, images/val, labels/train, labels/val), using the master
class ids from config/classes.yaml so it merges cleanly with your COCO
subset and any IDD cow data.

Usage:
    python3 generate_synthetic_dataset.py --dst ./dataset_merged --per-class 300
    python3 generate_synthetic_dataset.py --dst ./dataset_merged --per-class 300 \
        --backgrounds-dir ./venue_photos   # once you have real background photos
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Master class ids - MUST match config/classes.yaml exactly
CLASS_IDS = {
    'speed_limit_10': 0,
    'speed_limit_15': 1,
    'speed_limit_30': 2,
    'speed_limit_60': 3,
    'speed_limit_80': 4,
    'traffic_light_red': 5,
    'traffic_light_green': 6,
    'traffic_light_amber': 7,
    'steel_barricade': 14,
}

SIGN_VALUES = ['10', '15', '30', '60', '80']
LIGHT_COLORS = {'red': (220, 30, 30), 'green': (30, 200, 60), 'amber': (240, 170, 20)}

CANVAS_SIZE = (960, 540)  # background canvas size
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def render_speed_sign(value: str, size: int) -> Image.Image:
    """White circle, red ring, black digits - standard speed limit sign."""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = int(size * 0.04)
    draw.ellipse([margin, margin, size - margin, size - margin],
                 fill=(255, 255, 255, 255))
    ring_width = int(size * 0.11)
    draw.ellipse([margin, margin, size - margin, size - margin],
                 outline=(200, 20, 20, 255), width=ring_width)
    font = load_font(int(size * 0.42))
    bbox = draw.textbbox((0, 0), value, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
               value, font=font, fill=(20, 20, 20, 255))
    return img


def render_traffic_light(active_color: str, size: int) -> Image.Image:
    """Black housing with 3 lens circles, one lit brightly, others dim."""
    w, h = int(size * 0.55), size
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=int(w * 0.15),
                            fill=(25, 25, 25, 255))
    lens_r = int(w * 0.32)
    cx = w // 2
    centers_y = [int(h * 0.20), int(h * 0.5), int(h * 0.80)]
    order = ['red', 'amber', 'green']
    for color_name, cy in zip(order, centers_y):
        base = LIGHT_COLORS[color_name]
        if color_name == active_color:
            fill = base + (255,)
            # glow halo
            draw.ellipse([cx - lens_r * 1.4, cy - lens_r * 1.4,
                          cx + lens_r * 1.4, cy + lens_r * 1.4],
                         fill=base + (90,))
        else:
            dim = tuple(int(c * 0.25) for c in base)
            fill = dim + (255,)
        draw.ellipse([cx - lens_r, cy - lens_r, cx + lens_r, cy + lens_r], fill=fill)
    return img


def render_steel_barricade(size: int) -> Image.Image:
    """Yellow/black striped steel barrier, simplified frame + diagonal stripes."""
    w, h = size, int(size * 0.45)
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    frame_c = (60, 60, 60, 255)
    draw.rectangle([0, 0, w - 1, h - 1], outline=frame_c, width=int(h * 0.08))
    stripe_w = int(w * 0.09)
    x = -h
    toggle = False
    while x < w + h:
        color = (235, 190, 20, 255) if toggle else (30, 30, 30, 255)
        draw.polygon([(x, h), (x + h, 0), (x + h + stripe_w, 0), (x + stripe_w, h)],
                     fill=color)
        x += stripe_w
        toggle = not toggle
    # crop back to a clean rectangle mask
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).rectangle([0, 0, w - 1, h - 1], fill=255)
    out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    draw2 = ImageDraw.Draw(out)
    draw2.rectangle([0, 0, w - 1, h - 1], outline=frame_c, width=int(h * 0.08))
    return out


def make_background(bg_paths):
    if bg_paths:
        bg = Image.open(random.choice(bg_paths)).convert('RGB')
        bg = bg.resize(CANVAS_SIZE)
        return bg
    # Simple procedural fallback: flat sky-over-road gradient
    img = Image.new('RGB', CANVAS_SIZE)
    draw = ImageDraw.Draw(img)
    horizon = int(CANVAS_SIZE[1] * random.uniform(0.35, 0.55))
    sky = tuple(random.randint(120, 200) for _ in range(3))
    road = tuple(random.randint(40, 90) for _ in range(3))
    draw.rectangle([0, 0, CANVAS_SIZE[0], horizon], fill=sky)
    draw.rectangle([0, horizon, CANVAS_SIZE[0], CANVAS_SIZE[1]], fill=road)
    return img


def augment_and_place(obj_img: Image.Image, bg: Image.Image):
    """Random scale/rotation/brightness/blur, composite onto bg, return
    (composited_image, bbox_xyxy_in_bg_pixels)."""
    scale = random.uniform(0.08, 0.35)  # object size relative to canvas height,
                                        # simulating near (large) to far (small)
    target_h = int(CANVAS_SIZE[1] * scale)
    aspect = obj_img.width / obj_img.height
    target_w = int(target_h * aspect)
    obj = obj_img.resize((max(target_w, 1), max(target_h, 1)))

    angle = random.uniform(-8, 8)
    obj = obj.rotate(angle, expand=True, resample=Image.BICUBIC)

    if random.random() < 0.5:
        obj = obj.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

    enhancer = ImageEnhance.Brightness(obj)
    obj = enhancer.enhance(random.uniform(0.7, 1.3))
    enhancer = ImageEnhance.Contrast(obj)
    obj = enhancer.enhance(random.uniform(0.8, 1.2))

    max_x = CANVAS_SIZE[0] - obj.width
    max_y = CANVAS_SIZE[1] - obj.height
    if max_x < 0 or max_y < 0:
        return None, None
    px = random.randint(0, max_x)
    py = random.randint(0, max_y)

    composited = bg.copy()
    composited.paste(obj, (px, py), obj)
    bbox = (px, py, px + obj.width, py + obj.height)
    return composited, bbox


def bbox_to_yolo(bbox, canvas_size):
    x1, y1, x2, y2 = bbox
    w, h = canvas_size
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', default='./dataset_merged')
    parser.add_argument('--per-class', type=int, default=300,
                         help='Number of synthetic images per class')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--backgrounds-dir', default=None,
                         help='Optional folder of real background photos '
                              '(e.g. venue/track photos) to composite onto. '
                              'Falls back to simple procedural backgrounds if omitted.')
    args = parser.parse_args()

    dst = Path(args.dst)
    for split in ['train', 'val']:
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)

    bg_paths = []
    if args.backgrounds_dir:
        bg_dir = Path(args.backgrounds_dir)
        bg_paths = [str(p) for p in bg_dir.glob('*') if p.suffix.lower() in
                    ('.jpg', '.jpeg', '.png')]
        print(f'Using {len(bg_paths)} real background photos from {bg_dir}')
    if not bg_paths:
        print('No backgrounds provided - using procedural sky/road backgrounds. '
              'Real venue photos as backgrounds (once available) will improve '
              'domain realism significantly - rerun with --backgrounds-dir then.')

    jobs = []  # (class_name, render_fn)
    for val in SIGN_VALUES:
        jobs.append((f'speed_limit_{val}', lambda v=val, s=180: render_speed_sign(v, s)))
    for color in LIGHT_COLORS:
        jobs.append((f'traffic_light_{color}', lambda c=color, s=220: render_traffic_light(c, s)))
    jobs.append(('steel_barricade', lambda s=300: render_steel_barricade(s)))

    total_written = 0
    for class_name, render_fn in jobs:
        class_id = CLASS_IDS[class_name]
        n_val = int(args.per_class * args.val_fraction)
        n_train = args.per_class - n_val

        for split, n in [('train', n_train), ('val', n_val)]:
            written = 0
            attempts = 0
            while written < n and attempts < n * 3:
                attempts += 1
                obj_img = render_fn()
                bg = make_background(bg_paths)
                composited, bbox = augment_and_place(obj_img, bg)
                if composited is None:
                    continue
                idx = total_written
                img_name = f'{class_name}_{split}_{idx:05d}.jpg'
                label_name = f'{class_name}_{split}_{idx:05d}.txt'
                composited.convert('RGB').save(dst / 'images' / split / img_name, quality=90)
                cx, cy, bw, bh = bbox_to_yolo(bbox, CANVAS_SIZE)
                (dst / 'labels' / split / label_name).write_text(
                    f'{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                written += 1
                total_written += 1
        print(f'{class_name}: wrote {n_train} train + {n_val} val images')

    print(f'\nTotal synthetic images written: {total_written}')
    print('These are now merged into the same images/{train,val} and '
          'labels/{train,val} folders as your COCO subset - ready for one '
          'combined data.yaml covering all classes present so far.')


if __name__ == '__main__':
    main()
