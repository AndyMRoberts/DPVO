# Modifications in order to run heterogeneously

To record baseline power readings 1 minute
```
python andy/manual_test.py
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

with launch file (power logging, onnx switching options)

```bash
python launch_evaluation.py \
    --test_run_name tartan_mono_offline \
    --weights dpvo.pth \
    --split test \
    --power_log \
    --viz \
    --show_img \
    --save_trajectory \
    --plot
```