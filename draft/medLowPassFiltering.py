# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from scipy.signal import butter, filtfilt
from skimage.filters import threshold_otsu
from pathlib import Path


# Enable inline plotting for Jupyter
#%matplotlib inline

# Load CSV File
csv_path = Path(__file__).resolve().parents[1] / 'measurements' / 'Data' / 'Linear' / 'Linear_64_64.csv'  # Change this to your CSV file path
df = pd.read_csv(csv_path)
data=df['EVM1 POWER Results (W)']


#################################################################################
# Median Filtering to remove outliers
#################################################################################

kernel_size = 11  # adjust based on smoothing needed, e.g., 3, 5, 7
median_filtered = medfilt(data, kernel_size=kernel_size)

df['median_filtered'] = median_filtered


#################################################################################
# Low pass filtering to filter noise
#################################################################################
data = df['median_filtered']

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

fs = 5  # Sampling frequency (Hz) — adjust depending on your time step spacing
cutoff = 0.1  # Cutoff frequency (Hz) — lower means smoother

filtered = lowpass_filter(data, cutoff, fs)

df['lowpass_filtered'] = filtered

#################################################################################
# Averaging to smooth the data further
#################################################################################
df["smoothed"]=df['lowpass_filtered'].rolling(window=30, center=True).mean()

#################################################################################
# Compute means and variances
#################################################################################
# Above threshold
#threshold = 1.3
threshold = threshold_otsu(df['smoothed'].dropna().values)
print("Threshold value:", threshold)

above_threshold = df[df['smoothed'] > threshold]
average_power_active = above_threshold['smoothed'].mean()
variance_active = above_threshold['smoothed'].var()
print(f"Variance (active state, >{threshold} W): {variance_active:.5f} W²")
print(f"Average power (active state, >{threshold} W): {average_power_active:.5f} W")

# Below threshold
below_threshold = df[df['smoothed'] < threshold]
average_power_idle = below_threshold['smoothed'].mean()
variance_idle = below_threshold['smoothed'].var()
print(f"Variance (idle state, <{threshold} W): {variance_idle:.5f} W²")
print(f"Average power (idle state, <{threshold} W): {average_power_idle:.5f} W")

# Final results
print("Final Results: average:", average_power_active -  average_power_idle)
print("Final Results: average:", variance_active -  variance_idle)

# mean_val = df['smoothed'].mean()
# std_val = df['smoothed'].std()
# # threshold = mean_val + 0.5 * std_val 
# threshold = threshold_otsu(df['smoothed'].dropna().values)
# print("Treshold", threshold)
#################################################################################
# Plot
#################################################################################
smoothed = df["smoothed"]
smoothed = smoothed.dropna()
# Align 'Sample' values with the smoothed data index
sample_aligned = df['Sample'].iloc[smoothed.index]

time_sec = df['Sample'] * 0.1 # Assuming each sample corresponds to 0.1 seconds
#time_sec = df['Sample'] / 1000 # Assuming 'Sample' is in milliseconds, convert to seconds
time_sec_smoothed = time_sec.iloc[smoothed.index]

plt.figure(figsize=(20, 5))
plt.plot(time_sec_smoothed, smoothed, linewidth=2.0)
plt.xlim(time_sec_smoothed.min(), time_sec_smoothed.max())
plt.xlabel('Time (s)')
plt.ylabel('Power consumption (W)')
plt.title('Power consumption in Watts (W)')
#plt.legend()
#plt.savefig("high_res_plot.svg",format='svg', dpi=300, bbox_inches='tight', transparent=False)
plt.grid(True)
plt.show()