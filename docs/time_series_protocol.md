# Time-Series Tabular Protocol

## Research Framing

The project studies LLM-guided NAS for tabular datasets whose features are
derived from time-series data. The LLM is not used as a forecaster. It acts as a
controller that proposes neural tabular architectures and training
hyperparameters under a fixed search budget.

The key claim is:

> On time-series-derived tabular datasets, a dataset-aware LLM-NAS controller can
> select neural architectures that improve average rank against fixed classical
> baselines such as CatBoost and LightGBM, random NAS, and Optuna TPE.

## Dataset Types

The protocol distinguishes two dataset types:

1. **Forecasting-safe time-series tables.** Rows are chronologically ordered.
   Features for target time `t` must be built only from observations before `t`.
   ELEC2 implements this protocol, but is excluded from the reported benchmark
   because its chronological validation and test partitions are single-class.
2. **Time-series-derived classification tables.** Rows are windows or segments
   extracted from signals, for example HAR or ECG5000. These datasets are useful
   for studying temporal signal features in tabular form, but they are not the
   same as a strict forecasting split unless an original temporal/group split is
   available.

For ECG5000, the UCR-provided TRAIN/TEST partition is preserved and validation
is derived only from TRAIN. For HAR, the UCI-provided subject-separated test
partition is preserved and validation is derived only from TRAIN.

For larger raw-signal experiments, HARTH and PAMAP2 are segmented into windows
of 128 sensor readings with stride 32. A window is retained only if at least
80 percent of its timestamps share the resulting activity label. Participants
are assigned to fixed train, validation, and test groups before window
construction, so overlapping windows never cross an evaluation boundary. The
HARTH partition is chosen from activity-label availability only so that its
rare activity is represented in every partition, without inspecting model
results.

- HARTH: 6,461,328 raw readings from 22 participants and two accelerometers.
- PAMAP2: the official Protocol recordings from the 3,850,505-reading archive
  are used, covering 12 activities common to the participant holdout groups.
  Optional-activity recordings are excluded because they introduce activity
  labels absent from some subject groups.

## ELEC2 Preprocessing

ELEC2 is converted into supervised tabular form before any model sees the data.
For each target row `t`, predictors are generated only from past rows:

- lags: `1, 2, 3, 6, 12, 24, 48`;
- rolling means and standard deviations over windows `6, 12, 24, 48`;
- first differences: `x[t-1] - x[t-2]`.

The current timestamp features `x[t]` are not used. Initial rows without enough
history are dropped.

After feature construction, the split is chronological:

- first part: train;
- middle part: validation;
- last part: test.

There is no shuffling and no look-ahead from validation/test into train.

## Preprocessing Fit Rules

All preprocessing state is fit only on the train split:

- numeric imputation uses train medians;
- standard scaling uses train mean and standard deviation;
- quantile transformations, when selected by NAS, are fit on train only;
- categorical vocabularies are built from train only, with out-of-vocabulary
  categories mapped to a reserved value.

## Compared Methods

Every method receives the same train/validation/test split and the same
time-series-derived tabular features:

- CatBoost;
- LightGBM;
- Random NAS over the same neural search space;
- Optuna TPE over the neural search space;
- LLM-NAS batch search;
- optional greedy ensemble over top LLM-NAS trials.

Configuration search, early stopping, and greedy ensemble weights use
validation only. The headline metric used for method comparison is computed
once on the untouched test split after the configuration has been selected.
The LLM prompt receives no test label balance or sampled test records. For
reported LLM-NAS runs, failure of LLM proposal generation must abort the run;
random fallback runs are ablations and may not be reported as LLM-NAS.

## Reporting

Each experiment writes `dataset_summary.json`. For time-series datasets it must
include:

- `split_strategy`;
- `feature_engineering.type`;
- `feature_engineering.target_alignment`;
- lag steps;
- rolling windows;
- whether current timestamp features are used.
- subject partitions for window-classification datasets.

These fields are used to verify that the experiment is leakage-free.
