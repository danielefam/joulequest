# How Results Are Computed in data_processing.py

In this note, I explain step by step how I compute the final outputs:

- power_avg_W
- power_var_W2
- energy_avg_J
- energy_var_J2

## 1) Signal Preprocessing

I start from the raw power signal x[n], taken from the EVM1 POWER Results (W) column.

1. Median filter

$$
x_m[n] = \mathrm{medfilt}(x[n], k)
$$

where k = kernel_size.

2. Butterworth low-pass filter (using filtfilt)

$$
x_l[n] = \mathrm{filtfilt}(b, a, x_m[n])
$$

where b and a are computed from a Butterworth filter with:

- cutoff frequency f_c = cutoff
- sampling frequency f_s = fs
- filter order = order

3. Centered rolling average

$$
s[n] = \mathrm{rolling\_mean}(x_l[n], W)
$$

where W = window_size.

4. Otsu threshold

$$
T = \mathrm{Otsu}(s[n])
$$

## 2) Active Region Detection

I define the activity mask as:

$$
a[n] =
\begin{cases}
1 & \text{if } s[n] > T \\
0 & \text{otherwise}
\end{cases}
$$

Transitions in a[n] give me the active region boundaries:

- start_i when the signal goes from 0 to 1
- end_i when the signal goes from 1 to 0

For each active region i, I compute its duration:

$$
d_i = (end_i - start_i) \cdot \Delta t
$$

with Delta t = sampling_interval = 0.1 s.

## 3) Per-Region Statistics (Intra-Region)

For each active region i:

1. Active segment mean and variance

$$
\mu_{A,i} = \mathrm{mean}(s[start_i:end_i])
$$

$$
\sigma^2_{A,i} = \mathrm{var}(s[start_i:end_i])
$$

2. Local idle mean and variance (idle before + idle after, when available)

$$
\mu_{I,i} = \mathrm{mean}(idle_i)
$$

$$
\sigma^2_{I,i} = \mathrm{var}(idle_i)
$$

3. Net power offset for that region

$$
P_i = \mu_{A,i} - \mu_{I,i}
$$

4. Net power variance for that region

$$
V_{P,i} = \sigma^2_{A,i} + \sigma^2_{I,i}
$$

5. Region energy (scaled by 100000 in the current implementation)

$$
E_i = P_i \cdot \frac{d_i}{100000}
$$

6. Region energy variance

$$
V_{E,i} = V_{P,i} \cdot \left(\frac{d_i}{100000}\right)^2
$$

## 4) Final Aggregation Across Regions

Let N be the number of detected active regions.

1. Final average power

2. ```math
   ext{power\_avg\_W} = \frac{1}{N}\sum_{i=1}^{N} P_i
   ```
2. Final power variance (sample variance, ddof = 1)

$$
ext{power\_var\_W2} = \mathrm{Var}_{\text{sample}}(P_1, \dots, P_N)
$$

3. Total active duration

$$
D_{tot} = \sum_{i=1}^{N} d_i
$$

4. Final average energy (exactly as implemented)

$$
ext{energy\_avg\_J} = \frac{\sum_{i=1}^{N} d_i \cdot E_i}{D_{tot}}
$$

5. Final energy variance (sample variance, ddof = 1)

$$
ext{energy\_var\_J2} = \mathrm{Var}_{\text{sample}}(E_1, \dots, E_N)
$$

## Technical Note

In the energy_avg_J formula, E_i already includes d_i. This means the final aggregation applies an additional duration weight, so the effective contribution is proportional to d_i^2.
