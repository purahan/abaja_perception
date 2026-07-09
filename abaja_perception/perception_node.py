"""
abaja_perception perception_node

Pipeline: camera frame -> TensorRT inference -> per-class threshold filter
-> NMS -> radar-gate (optional false-positive suppression) -> IoU tracker
w/ class smoothing -> overlay (bbox + label + confidence) -> publish +
record + display.

TODOs marked below are the pieces that depend on your actual hardware
(camera driver topic/msg type, RADAR message format) — everything else
runs as-is once you set config/classes.yaml's engine_path.
"""

import time
import yaml
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from .tracker import SimpleTracker, Detection
from .image_utils import imgmsg_to_bgr8


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('abaja_perception_node')

        self.declare_parameter('config_path', '')
        config_path = self.get_parameter('config_path').get_parameter_value().string_value
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        self.class_names = list(self.cfg['classes'].keys())
        self.id_to_name = {v['id']: k for k, v in self.cfg['classes'].items()}
        self.thresholds = {v['id']: v['conf_threshold'] for v in self.cfg['classes'].values()}

        # backend: "torch" for dev-machine testing (no TensorRT needed),
        # "trt" for the actual Jetson deployment. Defaults to "torch" if
        # not set, since that's the safe choice on a regular laptop.
        self.backend = self.cfg.get('backend', 'torch')
        self.use_custom_classes = self.cfg.get('use_custom_classes', False)

        if self.backend == 'trt':
            from .trt_infer import TRTDetector
            self.detector = TRTDetector(
                engine_path=self.cfg['engine_path'],
                input_size=tuple(self.cfg['input_size']))
        elif self.backend == 'torch':
            from .torch_infer import TorchDetector
            weights = self.cfg.get('torch_weights', 'yolov8n.pt')
            self.detector = TorchDetector(weights_path=weights, device='cpu')
            if not self.use_custom_classes:
                # Pretrained COCO weights: overwrite id_to_name with the
                # model's own class names so labels display correctly
                # (person/car/bicycle/etc.) instead of your 15 custom
                # names, which won't match COCO's class ids.
                self.id_to_name = self.detector.names
                self.get_logger().warn(
                    'Using pretrained COCO weights with use_custom_classes: '
                    'false — this is a pipeline sanity check only. Labels '
                    'will show COCO classes (person, car, ...), not your '
                    '15 event classes, until you fine-tune and switch '
                    'use_custom_classes to true.')
        else:
            raise ValueError(f"Unknown backend '{self.backend}', expected 'torch' or 'trt'")

        self.tracker = SimpleTracker(
            iou_threshold=self.cfg['track_iou_threshold'],
            smoothing_window=self.cfg['smoothing_window'],
            min_hits=self.cfg['min_track_hits'])

        # TODO: confirm actual camera topic name/type from your driver.
        self.sub_image = self.create_subscription(
            Image, '/camera/image_raw', self.on_image, 10)

        # TODO: RADAR gating is optional-but-recommended for false-positive
        # suppression (e.g. shadows detected as objects). Wire this up to
        # whatever msg type your RADAR driver publishes — this stub just
        # keeps the last range reading and does a coarse presence check.
        # self.sub_radar = self.create_subscription(
        #     YourRadarMsgType, '/radar/detections', self.on_radar, 10)
        self.radar_ranges = []  # populate via on_radar callback if enabled

        self.pub_detections = self.create_publisher(
            Detection2DArray, '/abaja_perception/detections', 10)

        self.video_writer = None  # lazily initialized on first frame
        self._display_available = True  # flips to False if imshow ever fails
        self.record_path = '/mnt/user-data/outputs/operational_run.mp4'

        self.get_logger().info('abaja_perception_node initialized')

    def on_radar(self, msg):
        # TODO: extract range/bearing per your RADAR driver's message and
        # store recent readings in self.radar_ranges for gating below.
        pass

    def radar_confirms(self, bbox, frame_shape) -> bool:
        """Very coarse camera-radar gate: returns True if we have no radar
        data (fail-open, don't gate) or if a radar return roughly lines up
        with the bbox's horizontal position. Tune/replace once your radar
        driver + calibration (camera-radar extrinsics) is in place."""
        if not self.radar_ranges:
            return True
        # Placeholder logic — replace with real angle/range correlation
        # once camera-radar calibration is done.
        return True

    def decode_and_nms(self, raw_outputs: dict, frame_shape):
        """Decode YOLO-style output and apply per-class conf threshold +
        NMS. Adjust the key name / output layout to match your exported
        model — this assumes a single output tensor shaped
        (1, num_classes+4, num_boxes), the standard YOLOv8/v11 ONNX export
        layout, transposed to (num_boxes, num_classes+4)."""
        output = list(raw_outputs.values())[0]
        num_classes = len(self.class_names)
        preds = output.reshape(num_classes + 4, -1).T  # (num_boxes, 4+num_classes)

        boxes, scores, class_ids = [], [], []
        h, w = frame_shape[:2]
        sx, sy = w / self.cfg['input_size'][0], h / self.cfg['input_size'][1]

        for row in preds:
            cx, cy, bw, bh = row[:4]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])
            if conf < self.thresholds.get(cls_id, 0.4):
                continue
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=0.0,
            nms_threshold=self.cfg['nms_iou_threshold'])
        indices = np.array(indices).flatten() if len(indices) else []

        detections = []
        for i in indices:
            x, y, bw, bh = boxes[i]
            detections.append(Detection(
                class_id=class_ids[i], conf=scores[i],
                bbox=(x, y, x + bw, y + bh)))
        return detections

    def on_image(self, msg: Image):
        t0 = time.time()
        frame = imgmsg_to_bgr8(msg)

        if self.backend == 'trt':
            raw_outputs = self.detector.infer(frame)
            detections = self.decode_and_nms(raw_outputs, frame.shape)
        else:  # torch — already does its own NMS + thresholding internally
            detections = self.detector.detect(
                frame, thresholds=self.thresholds,
                use_custom_classes=self.use_custom_classes)

        detections = [d for d in detections
                      if self.radar_confirms(d.bbox, frame.shape)]

        tracks = self.tracker.update(detections)

        self.publish_detections(tracks, msg.header)
        self.draw_and_record(frame, tracks)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        self.get_logger().debug(f'inference+track fps: {fps:.1f}')

    def publish_detections(self, tracks, header):
        out_msg = Detection2DArray()
        out_msg.header = header
        for t in tracks:
            det = Detection2D()
            x1, y1, x2, y2 = t.bbox
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = self.id_to_name.get(t.smoothed_class(), 'unknown')
            hyp.hypothesis.score = t.smoothed_conf()
            det.results.append(hyp)
            out_msg.detections.append(det)
        self.pub_detections.publish(out_msg)

    def draw_and_record(self, frame, tracks):
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            label = self.id_to_name.get(t.smoothed_class(), 'unknown')
            conf = t.smoothed_conf()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} {conf:.2f}'
            cv2.putText(frame, text, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.record_path, fourcc, 30.0, (w, h))
            self.get_logger().info(f'Recording live output to {self.record_path}')

        self.video_writer.write(frame)

        if self._display_available:
            try:
                cv2.imshow('abaja_perception - live output', frame)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(
                    f'Live preview window unavailable ({e}). Recording '
                    f'continues normally - this only affects the on-screen '
                    f'display, not the saved .mp4. Fix your OpenCV install '
                    f'(needs a GUI-capable, non-headless build) if you need '
                    f'the live preview.')
                self._display_available = False

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
