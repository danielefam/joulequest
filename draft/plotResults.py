# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Enable inline plotting for Jupyter
#%matplotlib inline

# Load CSV File
csv_path = '/home/frederic/Documents/BANERA/Code/Linear_2048_2048.csv'  # Change this to your CSV file path
df = pd.read_csv(csv_path)

# Smooth the 'EVM1 POWER Results (W)' using a rolling mean
window_size = 1  # Adjust the window size as needed
smoothed = df['EVM1 POWER Results (W)'].rolling(window=window_size, center=True).mean()

smoothed = smoothed.dropna()
# Align 'Sample' values with the smoothed data index
sample_aligned = df['Sample'].iloc[smoothed.index]

time_sec = df['Sample'] * 0.1 # Assuming each sample corresponds to 0.1 seconds
#time_sec = df['Sample'] / 1000 # Assuming 'Sample' is in milliseconds, convert to seconds
time_sec_smoothed = time_sec.iloc[smoothed.index]

plt.figure(figsize=(20, 5))
# plt.plot(time_sec_smoothed, smoothed, label=f'Rolling Mean (window={window_size})',linewidth=2.0)
plt.plot(df["Sample"], smoothed, label=f'Rolling Mean (window={window_size})',linewidth=2.0)
# plt.xlim(time_sec_smoothed.min(), time_sec_smoothed.max())
plt.xlabel('Time (s)')
plt.ylabel('Power consumption (W)')
plt.title('Power consumption in Watts (W)')
#plt.legend()
#plt.savefig("high_res_plot.svg",format='svg', dpi=300, bbox_inches='tight', transparent=False)
plt.grid(True)
plt.show()