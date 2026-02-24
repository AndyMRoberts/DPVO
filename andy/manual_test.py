from profiler import Profiler, ProfilerSetupError
import time
import os

Profiler.verify_setup()

# Create profiler: output base name, sampling frequency (Hz), title
output_dir = f'{os.path.dirname(os.path.abspath(__file__))}/runs'

# cpu_power_max_w and gpu_power_max_w set fixed y-axis ceilings on the power subplots.
# GPU memory upper limit is read automatically from the system.
# Percentage metrics (CPU %, GPU %, RAM %) are always capped at 100.
p = Profiler(output_dir, frequency_hz=2.0, title="Profiler Reference and Initial Test",
             cpu_power_max_w=200.0, 
             gpu_power_max_w=200.0,
             gpu_memory_total_gb= 16.376)
print(f'Data will be saved to output directory: {output_dir}')

#first record a 10s reference dataset
print(f'Recording reference to {output_dir}/reference')
ref_dir = p.record_reference(duration_seconds=5)

# option to store own data in reference directory
with open(f'{ref_dir}/extra_data.txt', 'w') as f:
    f.write('Reference Data recorded in manual test. ')

# Start recording (runs in background thread)
print('Recording target data using reference data to get adjusted actual data.')
run_dir = p.start(use_reference=True)

# ... run your ML model, SLAM pipeline, etc. ...
run_time = 5
for i in range(run_time):
    print(f'Recording ready in {run_time - i}')
    time.sleep(1.0)


# Stop and write outputs; optionally pass num_frames for energy-per-frame stats
p.stop(num_frames=100)
print(f'Run Completed')

# option to store own data in reference directory
with open(f'{run_dir}/extra_data.txt', 'w') as f:
    f.write('Manual Test Complete')