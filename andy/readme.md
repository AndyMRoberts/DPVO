# Modifications in order to run heterogeneously

To record baseline power readings 1 minute
```
python andy/manual_test.py
```

baseline onnx/pytorch features only models running
```bash
python launch_evaluation.py \
    --test_run_name tartan_features_only_baselining_onnx \
    --weights dpvo.pth \
    --split test \
    --power_log \
    --file_to_run features_only \
    --backend onnx \
    --onnx_dir andy/onnx
```

```bash
python launch_evaluation.py \
    --test_run_name tartan_features_only_baselining_pytorch \
    --weights dpvo.pth \
    --split test \
    --power_log \
    --file_to_run features_only
```


raw file
```bash
python evaluate_tartan_andy.py \
    --trials=1 \
    --split=test \
    --plot \
    --save_trajectory \
    --show_img \
    --viz
```

with launch file (power logging, backend options)

```bash
python launch_evaluation.py \
    --test_run_name tartan_mono_offline_onnx_static_int8 \
    --weights dpvo.pth \
    --split test \
    --power_log \
    --viz \
    --show_img \
    --save_trajectory \
    --plot \
    --backend onnx \
    --onnx_dir andy/onnx_static_int8
```

```bash
python launch_evaluation.py \
    --test_run_name tartan_mono_offline_pytorch \
    --weights dpvo.pth \
    --split test \
    --power_log \
    --viz \
    --show_img \
    --save_trajectory \
    --plot
```

### Backend: PyTorch vs PyTorch+ONNX

- **Pure PyTorch** (default): `--backend pytorch`
- **Hybrid (encoders via ONNX)**: Export encoders first with `andy/onnx_conversion.ipynb`, then run with `--backend onnx` and optionally `--onnx_dir andy/onnx`. Requires `onnxruntime` or `onnxruntime-gpu`.

Example with ONNX encoders:
```bash
python launch_evaluation.py --test_run_name tartan_onnx --backend onnx --onnx_dir andy/onnx
python evaluate_tartan_andy.py --backend onnx --onnx_dir andy/onnx
python demo.py --backend onnx --onnx_dir andy/onnx --imagedir ... --calib ...
```

# Quantisation

Can use the onnx_quantize.py file
```
python andy/onnx_quantize.py --input-dir andy/onnx --weight-type int8 --suffix dynamic_int8
python andy/onnx_quantize.py --input-dir andy/onnx --weight-type int8 --suffix static_int8 --static
python andy/onnx_quantize.py --input-dir andy/onnx --weight-type fp16 --suffix fp16
```

**Int8 (dynamic quantisation)** produces models that use `DynamicQuantizeLinear` and `ConvInteger`. TensorRT does **not** support `DynamicQuantizeLinear`, so dynamic int8 models fail on the TensorRT execution provider.

- **For TensorRT / GPU int8:** Use **static** INT8 quantization (no `DynamicQuantizeLinear`; uses `QuantizeLinear` with constant scale/zero_point):
  ```bash
  python andy/onnx_quantize.py --input-dir andy/onnx --static
  ```
  Output: `andy/onnx_static_int8/`. Run with `--onnx_dir andy/onnx_static_int8`. Optional: `--calibration-batches 20`, `--calibration-shape 1,1,3,480,640`.

- **Int8 dynamic** (`andy/onnx_dynamic_int8`): The code tries TensorRT first for quantized models; TensorRT will reject these due to `DynamicQuantizeLinear`. Fallback is CPU (if ConvInteger is implemented).
- If you need int8 on GPU and TensorRT is not available or you hit ConvInteger issues: use **FP16** (`--weight-type fp16`, `--onnx_dir andy/onnx_fp16`), which runs on CUDA.