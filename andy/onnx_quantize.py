#!/usr/bin/env python3
"""
Quantize DPVO ONNX encoders (fnet and inet) from andy/onnx/ and save to a new
folder with a quantization suffix (e.g. andy/onnx_dynamic_int8/ or andy/onnx_static_int8/).

Supports:
  - int8/uint8 dynamic: ONNX Runtime dynamic quantization (no calibration).
    Uses DynamicQuantizeLinear; not supported by TensorRT.
  - int8 static: Static INT8 quantization with calibration. Uses QuantizeLinear
    with constant scale/zero_point (no DynamicQuantizeLinear), suitable for
    TensorRT execution provider.
  - fp16: Float16 conversion via onnxconverter-common (pip install onnxconverter-common).

Use the output directory as --onnx_dir when running with --backend onnx.
"""

import argparse
import os
import tempfile

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class DPVOCalibrationDataReader(CalibrationDataReader):
    """Yields calibration batches for fnet/inet: input 'images' with shape (1, 1, 3, H, W), float32 in [-0.5, 0.5]."""

    def __init__(self, model_path, shape=(1, 1, 3, 480, 640), num_batches=20, rng=None):
        self.model_path = model_path
        self.shape = shape
        self.num_batches = num_batches
        self.rng = np.random.default_rng(rng)
        self._input_name = None
        self._batch_index = 0
        # Infer input name from model
        model = onnx.load(model_path)
        if model.graph.input:
            self._input_name = model.graph.input[0].name
        else:
            self._input_name = "images"

    def get_next(self):
        if self._batch_index >= self.num_batches:
            return None
        # DPVO normalizes images as 2*(x/255)-0.5 -> range [-0.5, 0.5]
        data = self.rng.uniform(low=-0.5, high=0.5, size=self.shape).astype(np.float32)
        self._batch_index += 1
        return {self._input_name: data}

    def rewind(self):
        self._batch_index = 0


def main():
    parser = argparse.ArgumentParser(
        description="Quantize fnet.onnx and inet.onnx from an ONNX directory."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing fnet.onnx and inet.onnx (default: andy/onnx)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="dynamic_int8",
        help="Suffix for output directory: input dir parent + onnx_<suffix> (default: dynamic_int8)",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["uint8", "int8", "fp16"],
        default="int8",
        help="Weight type: int8/uint8 = dynamic quantization; fp16 = float16 conversion (default: int8)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static INT8 quantization (requires calibration). Produces QuantizeLinear with constant "
        "scale/zero_point, suitable for TensorRT. Ignored if --weight-type is fp16.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=20,
        help="Number of calibration batches for static quantization (default: 20)",
    )
    parser.add_argument(
        "--calibration-shape",
        type=str,
        default="1,1,3,480,640",
        help="Comma-separated shape for calibration input [batch,frames,channels,height,width] (default: 1,1,3,480,640)",
    )
    args = parser.parse_args()
    if args.static and args.weight_type in ("int8", "uint8") and args.suffix == "dynamic_int8":
        args.suffix = "static_int8"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = os.path.join(script_dir, "onnx")
    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    fnet_path = os.path.join(input_dir, "fnet.onnx")
    inet_path = os.path.join(input_dir, "inet.onnx")
    for name, p in [("fnet", fnet_path), ("inet", inet_path)]:
        if not os.path.isfile(p):
            raise SystemExit(f"Missing {name} model: {p}")

    # Output dir: same parent as input_dir, name = basename_input + "_" + suffix
    # e.g. andy/onnx -> andy/onnx_dynamic_int8
    parent = os.path.dirname(input_dir)
    base = os.path.basename(input_dir.rstrip(os.sep))
    output_dir = os.path.join(parent, f"{base}_{args.suffix}")
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ("fnet", fnet_path, os.path.join(output_dir, "fnet.onnx")),
        ("inet", inet_path, os.path.join(output_dir, "inet.onnx")),
    ]

    if args.weight_type == "fp16":
        try:
            from onnxconverter_common import float16 as float16_conv  # type: ignore[import-untyped]
        except ImportError:
            raise SystemExit(
                "FP16 conversion requires onnxconverter-common. Install with: pip install onnxconverter-common"
            )
        for name, src, dst in models:
            print(f"Converting {name} to FP16: {src} -> {dst}")
            model = onnx.load(src)
            model_fp16 = float16_conv.convert_float_to_float16(model, keep_io_types=True)
            onnx.save(model_fp16, dst)
            print(f"  Saved: {dst}")
    else:
        weight_type = QuantType.QInt8 if args.weight_type == "int8" else QuantType.QUInt8
        if args.static:
            # Static INT8: calibration + QuantizeLinear with constant scale/zero_point (no DynamicQuantizeLinear).
            # TensorRT-friendly. Use QDQ format and symmetric QInt8 for GPU/TRT.
            calib_shape = tuple(int(x) for x in args.calibration_shape.split(","))
            if len(calib_shape) != 5:
                raise SystemExit(
                    f"--calibration-shape must be 5 values (batch,frames,channels,height,width), got {args.calibration_shape}"
                )
            for name, src, dst in models:
                print(f"Static quantizing {name}: {src} -> {dst}")
                with tempfile.TemporaryDirectory(prefix="dpvo_quant_") as tmpdir:
                    shape_inferred = os.path.join(tmpdir, "shape_inferred.onnx")
                    quant_pre_process(
                        src,
                        shape_inferred,
                        skip_optimization=False,
                        skip_onnx_shape=False,
                    )
                    calib_reader = DPVOCalibrationDataReader(
                        shape_inferred,
                        shape=calib_shape,
                        num_batches=args.calibration_batches,
                    )
                    quantize_static(
                        shape_inferred,
                        dst,
                        calib_reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                        reduce_range=False,
                        calibrate_method=CalibrationMethod.MinMax,
                        extra_options={
                            "ActivationSymmetric": True,
                            "WeightSymmetric": True,
                        },
                    )
                print(f"  Saved: {dst}")
        else:
            for name, src, dst in models:
                print(f"Dynamic quantizing {name}: {src} -> {dst}")
                quantize_dynamic(
                    src,
                    dst,
                    weight_type=weight_type,
                    per_channel=True,
                    reduce_range=False,
                )
                print(f"  Saved: {dst}")

    print(f"Done. Output models in: {output_dir}")
    print(f"Run with: --backend onnx --onnx_dir {output_dir}")


if __name__ == "__main__":
    main()
