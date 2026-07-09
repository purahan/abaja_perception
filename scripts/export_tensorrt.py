#!/usr/bin/env python3
"""
Export a trained Ultralytics YOLO model to a TensorRT engine on the Jetson
Orin, and print a quick FPS benchmark for FP16 vs INT8 so you have real
numbers for the technical presentation's "benchmarking results, hardware
compatibility analysis" section.

Usage:
    python3 export_tensorrt.py --weights runs/detect/train/weights/best.pt \
        --precision fp16 --imgsz 640
    python3 export_tensorrt.py --weights ... --precision int8 --imgsz 640 \
        --calib-data /path/to/calibration_images/

Requires: ultralytics (`pip install ultralytics`), which wraps the
ONNX->TensorRT export and runs on-device using the JetPack-provided
TensorRT install.
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to trained .pt weights')
    parser.add_argument('--precision', choices=['fp16', 'int8'], default='fp16')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--calib-data', default=None,
                         help='Directory of representative images, required for int8')
    parser.add_argument('--benchmark-frames', type=int, default=200)
    args = parser.parse_args()

    if args.precision == 'int8' and not args.calib_data:
        raise ValueError('--calib-data is required for int8 export '
                          '(needs representative images for calibration)')

    model = YOLO(args.weights)

    export_kwargs = dict(format='engine', imgsz=args.imgsz, device=0)
    if args.precision == 'fp16':
        export_kwargs['half'] = True
    else:
        export_kwargs['int8'] = True
        export_kwargs['data'] = args.calib_data

    print(f'Exporting {args.weights} -> TensorRT engine ({args.precision})...')
    engine_path = model.export(**export_kwargs)
    print(f'Engine written to: {engine_path}')

    # Quick benchmark on a dummy tensor so you have an FPS number to quote
    engine_model = YOLO(engine_path)
    dummy = 'https://ultralytics.com/images/bus.jpg'  # any warm-up image works
    for _ in range(10):
        engine_model.predict(dummy, verbose=False)  # warm-up

    start = time.time()
    for _ in range(args.benchmark_frames):
        engine_model.predict(dummy, verbose=False)
    elapsed = time.time() - start
    fps = args.benchmark_frames / elapsed
    print(f'\nBenchmark ({args.precision}, imgsz={args.imgsz}): '
          f'{fps:.1f} FPS over {args.benchmark_frames} frames '
          f'({elapsed / args.benchmark_frames * 1000:.1f} ms/frame)')
    print('Run this once for fp16 and once for int8 and put both numbers '
          'in your presentation\'s performance-evaluation slide.')


if __name__ == '__main__':
    main()
