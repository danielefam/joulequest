# First Week Report

## Summary

During this first week, I focused on collecting new power measurements for the BANERA project and on checking the reliability of the measurement pipeline. My main goal was to gather data for the target models, but while working on the measurements I also discovered a problem in the inference code: in some runs, inference was being executed without disabling autograd properly. Because of that, I decided to repeat part of the measurement campaign so that the new dataset reflects the correct inference behavior.

## What I Did

I collected new measurements on the Jetson Nano for the models stored under the new measurement folder. In parallel, I reviewed the acquisition workflow and the post-processing logic to make sure that the recorded traces are consistent and usable for the final energy report.

While checking the implementation, I noticed that inference should be run without gradient tracking. This is important because the measurements must represent inference only, not training-related overhead. After identifying this issue, I repeated several measurements so that the affected runs are no longer part of the final dataset.

## Why I Redid Some Measurements

The reason I repeated some measurements is that, if autograd is left enabled during inference, the run includes extra overhead that is not representative of a real deployment scenario. Even if the model output is still correct, the power and energy numbers are no longer measuring pure inference cost.

For this reason, I treated the original measurements as unreliable for comparison and collected corrected ones. This gave me a cleaner dataset and reduced the risk of mixing valid and invalid traces in the final analysis.

## Notes on Energy Computation

I also spent time understanding how the energy is computed from the power traces, especially in cases where the measured waveform does not start exactly at the beginning of the active burst or does not end exactly at the end of it.

My conclusion is that these traces are not automatically invalid for energy estimation. If the active portion is still captured correctly, then the energy computed over the detected active interval is still meaningful. In other words, a trace that starts a little late or ends a little early is not necessarily biased in energy, as long as the energy is computed consistently on the portion that is actually recorded.

What changes in those cases is the completeness of the observation, not the physical meaning of the recorded samples. The main limitation is that the duration is partially observed, so those measurements are less suitable when I want to compare exact burst boundaries or verify the full shape of the active window. For this reason, I would keep them, but I would label them carefully and avoid presenting them as perfectly complete traces.

## How I Would Explain It To My Professor

I would say that I collected a first batch of measurements, then found a methodological issue in the inference code and repeated part of the campaign to make the dataset consistent. I also checked how the energy is extracted from the power traces, and I concluded that traces with a missing start or end are still usable for energy estimation if the active region is detected correctly. They are not ideal for boundary analysis, but they do not automatically invalidate the energy value.

## Current Status

At this point I have:

- collected a first set of measurements for the Jetson Nano campaign;
- repeated the affected runs after fixing the inference setup;
- reviewed the energy extraction logic;
- clarified how to interpret traces that are partially cut at the beginning or end.

## Next Steps

In the next phase, I want to consolidate the dataset, separate the corrected measurements from the earlier ones, and continue with the report generation so that the results can be presented in a clean and consistent way.
