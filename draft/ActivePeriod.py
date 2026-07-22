import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.signal import butter, filtfilt
from skimage.filters import threshold_otsu
import argparse
import numpy as np
from pathlib import Path

import tpu_linear_data_processing_old as tpu

csv_path = Path(__file__).resolve().parents[1] / 'measurements' / 'Data' / 'Linear' / 'Linear_64_64.csv'
df = pd.read_csv(csv_path)

power = df['EVM1 POWER Results (W)'].values
sampling_interval = 0.1

kernel_size = 11
fs = 5
cutoff = 0.1
window_size = 30

# Apply median filtering
df["median_filtered"]=tpu.median_filter_data(df['EVM1 POWER Results (W)'],kernel_size)

# Apply low-pass filtering
df["lowpass_filtered"] = tpu.lowpass_filter(df['median_filtered'], cutoff, fs)

# Smooth the data using a rolling average
df["smoothed"] = tpu.average_data(df['lowpass_filtered'], window_size)

# Compute threshold
threshold = tpu.get_threshold(df['smoothed'].dropna().values)

is_active = df["smoothed"] > threshold
change_points = np.diff(is_active.astype(int))

# Identify start and end indices of active periods (transitions from inactive to active and vice versa)
start_indices = np.where(change_points == 1)[0] + 1
end_indices = np.where(change_points == -1)[0] + 1

# If the first sample is active
if is_active.iloc[0]:
    start_indices = np.insert(start_indices, 0, 0)

# If the last sample is active
if is_active.iloc[-1]:
    end_indices = np.append(end_indices, len(power))


# print("Start Indices:", start_indices)
# print("End Indices:", end_indices)
# power_offset = []
# energy_offset = []

# filter out fan active regions
durations = [(end - start) * sampling_interval for start, end in zip(start_indices, end_indices)]
# Set a max expected duration threshold (e.g., 2× typical inference duration)
expected_duration = np.median(durations)
max_duration = expected_duration * 1.5
print(max_duration)
#filtered_regions = [(s, e) for (s, e), d in zip(zip(start_indices, end_indices), durations) if d <= max_duration]

regions =[]
for i,(start, end) in enumerate(zip(start_indices, end_indices)):
    
    duration = (end - start) * sampling_interval
    print(duration)
    if duration > max_duration:
        print(f"Skipping region {i} from {start} to {end} due to excessive duration: {duration:.2f}s")
        continue  # Skip this region if it exceeds the max duration
    average_power_active = df['smoothed'][start:end].mean()
    variance_power_active = df['smoothed'][start:end].var()
    energy_consumption_active = average_power_active * duration

    # Get idle region before
    if i == 0:
        idle_before = None
    else:
        prev_end = end_indices[i-1]
        idle_before = df['smoothed'][prev_end:start]

    # Get idle region after
    if i == len(start_indices) - 1:
        idle_after = None  # no next region
    else:
        next_start = start_indices[i + 1]
        idle_after = df['smoothed'][end:next_start]

    # Combine available idle segments
    idle_segments = []
    if idle_before is not None and not idle_before.empty:
        idle_segments.append(idle_before)
    if idle_after is not None and not idle_after.empty:
        idle_segments.append(idle_after)

    # Calculate idle power and energy if there are idle segmentss
    if idle_segments:
        idle_combined = pd.concat(idle_segments)
        #idle_duration = len(idle_combined) * sampling_interval
        
        idle_power = idle_combined.mean()
        idle_variance = idle_combined.var()

        #idle_energy = idle_power * idle_duration

        power_offset = average_power_active - idle_power
        #energy_offset.append(power_offset*duration)
    else:
        idle_power = np.nan
        power_offset = np.nan
        # energy_offset = np.nan
    
    regions.append({
        'sampling_rate': sampling_interval,

        'start_sample': start,
        'end_sample': end,
        'active_duration_s': duration,

        #'idle_before_start': end_indices[i-1] if i > 0 else 0,
        #'idle_after_end': start_indices[i+1] if i < len(start_indices) - 1 else len(df),
        #'idle_duration_s': idle_duration,

        'power_offset_W': power_offset,
        'avg_power_active_W': average_power_active,
        'variance_power_active_W2': variance_power_active,
        'avg_power_idle_W': idle_power,
        'avg_power_idle_variance_W2': idle_variance ,

        'energy_J': energy_consumption_active
        #'energy_active_J': energy_consumption_active,
        #'energy_idle_J': idle_energy,
     
    })

dd = pd.DataFrame(regions)
print("final:",dd["power_offset_W"].mean())

tpu.show_data(df)