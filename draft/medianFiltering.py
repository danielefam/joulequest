# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
from pathlib import Path

# Enable inline plotting for Jupyter
#%matplotlib inline

# Load CSV File
csv_path = Path(__file__).resolve().parents[1] / 'measurements' / 'Data' / 'Linear' / 'Linear_64_64.csv'  # Change this to your CSV file path
df = pd.read_csv(csv_path)
data=df['EVM1 POWER Results (W)']

kernel_size = 43  # adjust based on smoothing needed, e.g., 3, 5, 7
median_filtered = medfilt(data, kernel_size=kernel_size)

# Add back to DataFrame (optional)
df['median_filtered'] = median_filtered

smoothed = df['median_filtered'] 
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