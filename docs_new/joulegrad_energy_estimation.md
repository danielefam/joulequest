# Differentiable energy estimation: moved to joulegrad

The differentiable energy estimator that used to live in this repository's
`energy_estimator/` directory has been extracted into a standalone package:
**joulegrad**.

## Where it lives now

- Sibling checkout: `../joulegrad` (package import name: `joulegrad`,
  distribution name: `joulegrad`).
- Install it into the `banera_pt` environment with:

  ```bash
  conda run -n banera_pt python -m pip install -e ../joulegrad
  ```

- Once published, install a pinned release instead, e.g.
  `pip install "joulegrad @ git+ssh://git@github.com/danielefam/joulegrad.git@v0.1.0"`.

## What this repository still owns

JouleQuest (this repository) captures and processes the hardware measurements:
INA226 acquisition, adaptive burst runs, data processing, and the per-board
`summary.csv` files. joulegrad consumes those summaries; it does not replace
any measurement code here.

## Building a lookup table

The builder moved with the package. From one processed summary per
board/configuration:

```bash
conda run -n banera_pt python -m joulegrad.build_energy_lookup_table \
  measurements/Plot/pi5/summary.csv \
  --output measurements/Plot/pi5/energy_lookup_table.csv
```




After generating the lookup table, query the estimated energy of a layer
directly from the JouleQuest directory. For example, this estimates a linear
layer with 130 input features and 162 output features:

```bash
conda run -n banera_pt python -m joulegrad \
  measurements/Plot/pi5/energy_lookup_table.csv \
  linear 130 162
```

The command prints the estimated energy per inference, for example:

```text
0.128964165565 mJ/inference
```

The same command supports `conv` and `attention` layer queries. Run
`conda run -n banera_pt python -m joulegrad --help` for their arguments.

The strict acceptance policy is unchanged: only `COMPLETE` campaigns with
quality `OK`/`REVIEW`, positive energies, matching cycle counts, and valid
clock-alignment uncertainty. One lookup per board/runtime configuration — never
merged.

## Using the estimate in training

```python
from joulegrad import EnergyLookup, ModelEnergyRegularizer

lookup = EnergyLookup("measurements/Plot/pi5/energy_lookup_table.csv",
                      out_of_range="error")
regularizer = ModelEnergyRegularizer(model, lookup)
energy_mj = regularizer(masks)
loss = task_loss + energy_weight * energy_mj
```

The jouleNAS project (`../jouleNAS`, formerly `ecological_nas`) already
consumes joulegrad through its explicit topology bridges.

## Full documentation

See the joulegrad repository:

- `README.md` — overview and workflow;
- `differentiable_energy_estimator.md` — integration contract;
- `optimization.md` — performance design (prepared grids, device caches,
  batched evaluation).
