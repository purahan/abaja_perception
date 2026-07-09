#!/usr/bin/env python3
"""
Pull the real-world classes (car, pedestrian, bicyclist, two-wheeler) from
COCO via FiftyOne, and export directly to YOLO format for Ultralytics
training. This sidesteps the flaky raw cocodataset.org downloads by using
FiftyOne's Dataset Zoo, which handles the download/mirroring for you.

Note: COCO doesn't have "cow in Indian road context" well-represented —
for that class specifically, still pull from IDD (India Driving Dataset,
idd.insaan.iiit.ac.in) separately, or supplement with your own photos.
COCO also doesn't reliably have "steel barricade" or the event-specific
sign/light classes — those still need the synthetic generator.

Usage:
    pip install fiftyone
    python3 fetch_coco_subset.py --out ./coco_subset --max-samples 2000
"""

import argparse
import time
import fiftyone as fo
import fiftyone.zoo as foz

# Map COCO's class names to your event's class names/ids (must match
# config/classes.yaml exactly)
COCO_TO_EVENT_CLASS = {
    'car': 'car',
    'person': 'pedestrian',
    'bicycle': 'bicyclist',   # note: COCO's "bicycle" is the object, not
                              # necessarily a rider — review samples before
                              # trusting this 1:1, may need to filter to
                              # images where a person is also on/near it
    'motorcycle': 'two_wheeler',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./coco_subset',
                         help='Output directory for YOLO-format export')
    parser.add_argument('--max-samples', type=int, default=2000,
                         help='Max samples per split (keeps download size reasonable)')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation'])
    parser.add_argument('--num-workers', type=int, default=4,
                         help='Parallel download workers - lower this if you '
                              'hit connection resets from the COCO servers')
    parser.add_argument('--retries', type=int, default=5,
                         help='Retry attempts per split on connection errors')
    args = parser.parse_args()

    coco_classes = list(COCO_TO_EVENT_CLASS.keys())

    for split in args.splits:
        print(f'\nDownloading COCO-2017 [{split}] for classes: {coco_classes}')

        dataset = None
        for attempt in range(1, args.retries + 1):
            try:
                dataset = foz.load_zoo_dataset(
                    'coco-2017',
                    split=split,
                    classes=coco_classes,
                    max_samples=args.max_samples,
                    label_types=['detections'],
                    dataset_name=f'coco_subset_{split}',
                    num_workers=args.num_workers,
                )
                break
            except Exception as e:
                print(f'Attempt {attempt}/{args.retries} failed: {e}')
                if attempt == args.retries:
                    raise
                wait = min(30, 2 ** attempt)
                print(f'Retrying in {wait}s (already-downloaded annotations '
                      f'and images are cached, so this resumes rather than '
                      f'restarting from scratch)...')
                time.sleep(wait)

        # Remap COCO label names -> your event's class names before export
        for sample in dataset:
            if sample.ground_truth is None:
                continue
            for det in sample.ground_truth.detections:
                det.label = COCO_TO_EVENT_CLASS.get(det.label, det.label)
            sample.save()

        export_dir = f'{args.out}/{split}'
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field='ground_truth',
            classes=sorted(set(COCO_TO_EVENT_CLASS.values())),
        )
        print(f'Exported {len(dataset)} samples to {export_dir}')

    print('\nDone. Merge this with your synthetic sign/light/barricade data '
          'and any IDD cow samples into one data.yaml before training.')


if __name__ == '__main__':
    main()
