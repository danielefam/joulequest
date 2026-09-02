# Energy lookup production and consumption

JouleQuest owns the complete measurement-to-lookup pipeline:

1. acquire hardware power samples;
2. process runs into one summary per hardware/configuration in `measurements/lookup_summaries/summaries/`;
3. validate campaign quality and timing consistency; and
4. aggregate accepted layer measurements into `energy_lookup_table.csv`.

Create the table from one or more summaries belonging to the same hardware and
software configuration:

```bash
python -m processing_report.build_energy_lookup_table \
  measurements/lookup_summaries/summaries/pi5.csv \
  --output measurements/lookup_summaries/energy_lookup_table.csv
```

Never mix boards, runtimes, batches, dtypes, power modes, or acquisition
policies in one table.

## Consumer boundary

JouleGrad is an independent API-only package. It consumes the generated CSV;
it does not import JouleQuest, parse summaries, or provide a lookup-builder
command.

```python
from joulegrad import EnergyEstimator

estimator = EnergyEstimator(
    "measurements/lookup_summaries/energy_lookup_table.csv",
    out_of_range="error",
)
energy_mj = estimator.linear(130, 162)
```

The estimate is a differentiable PyTorch tensor in millijoules per input
sample. Consult JouleGrad's `docs/LOOKUP_SCHEMA.md` and `docs/API.md` for the
consumer contract and interpolation policies.

## Accepted measurements

The producer accepts complete rows with quality `OK` or `REVIEW`, positive
energy, matching detected/expected regions when available, a complete/aligned
clock status when available, and uncertainty within its threshold when those
columns are present.

Recognized model names include Linear, Conv, ResNetConv, Attention, and
RotaryAttention configurations. Repeated coordinates are aggregated into mean,
standard deviation, and measurement count.
