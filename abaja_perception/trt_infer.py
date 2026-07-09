"""
TensorRT engine wrapper for YOLO-style detectors on Jetson Orin.

Assumes you've already exported: PyTorch (.pt) -> ONNX -> TensorRT engine
via scripts/export_tensorrt.py. This class only handles loading the engine
and running inference — it does NOT do NMS or class filtering, that's
handled in perception_node.py so it can apply the per-class thresholds
from config/classes.yaml.

Requires `tensorrt` and `pycuda` (both ship with JetPack on Orin —
no separate pip install needed if you're on the standard JetPack image).
"""

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False


class TRTDetector:
    def __init__(self, engine_path: str, input_size=(640, 640)):
        if not _TRT_AVAILABLE:
            raise RuntimeError(
                "tensorrt/pycuda not importable. Run this on-device (Jetson "
                "with JetPack), not in a generic dev container."
            )
        self.input_size = input_size
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_binding = None
        self.output_bindings = []
        self.host_buffers = {}
        self.device_buffers = {}
        self.stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.host_buffers[name] = host_mem
            self.device_buffers[name] = device_mem
            if self.engine.binding_is_input(i):
                self.input_binding = name
            else:
                self.output_bindings.append(name)

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(frame_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def infer(self, frame_bgr: np.ndarray):
        """Returns raw model output array(s) — caller does decode + NMS."""
        input_data = self.preprocess(frame_bgr)
        np.copyto(self.host_buffers[self.input_binding], input_data.ravel())
        cuda.memcpy_htod_async(
            self.device_buffers[self.input_binding],
            self.host_buffers[self.input_binding], self.stream)

        bindings = [int(self.device_buffers[name])
                    for name in self.host_buffers]
        self.context.execute_async_v2(
            bindings=bindings, stream_handle=self.stream.handle)

        outputs = {}
        for name in self.output_bindings:
            cuda.memcpy_dtoh_async(
                self.host_buffers[name], self.device_buffers[name],
                self.stream)
        self.stream.synchronize()
        for name in self.output_bindings:
            outputs[name] = self.host_buffers[name].copy()
        return outputs
