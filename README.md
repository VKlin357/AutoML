# LLM-NAS: Neural Architecture Search for Tabular Data

Automated neural architecture search system for tabular data. The system searches over six neural network families (MLP, ResMLP, FT-Transformer, GatedTab, AutoInt, TabM) using a language model as the search controller.

## Project Structure

```
src/
  nas_orchestrator.py   - main search loop and prompt building
  models.py             - six architecture families
  train_nn.py           - training loop with multi-fidelity support
  search_space.py       - hyperparameter space definitions
  preprocessing.py      - data preprocessing pipeline
  data.py               - dataset loading (OpenML + built-in)
  ensemble.py           - greedy ensemble (Caruana 2004)
  surrogate.py          - surrogate model for candidate ranking
  multi_fidelity.py     - cheap/medium/full fidelity rungs
  metrics.py            - evaluation metrics
  baselines_automl.py   - CatBoost, LightGBM, Optuna baselines
  run_baselines_core.py - baseline runners
  run_nas_v2.py         - main entry point (NAS)
  run_baselines.py      - entry point for baselines
  run_experiment.py     - multi-dataset experiment runner
  utils.py              - seeding and utilities
gen_figures.py          - figure generation for results
```

## Quick Start

```bash
pip install -r requirements.txt

# Run NAS on Jannis dataset
python src/run_nas_v2.py \
  --openml_id 45021 \
  --task multiclass \
  --out_dir experiments/jannis_s42 \
  --budget 40 \
  --random_warmup 12 \
  --mode batch \
  --batch_n 20 \
  --batch_k 4 \
  --seed 42 \
  --llm_model gpt-4o-mini

# Run baselines (CatBoost + Optuna)
python src/run_baselines.py \
  --openml_id 45021 \
  --task multiclass \
  --out_dir experiments/jannis_baselines \
  --optuna --optuna_trials 40 \
  --seed 42
```

## Datasets

Experiments use five datasets from the RTDL benchmark (Gorishniy et al. 2021):

| Dataset    | OpenML ID | Rows    | Features | Classes | Metric   |
|------------|-----------|---------|----------|---------|----------|
| Volkert    | 41166     | 58,310  | 180      | 10      | Accuracy |
| Jannis     | 45021     | 57,580  | 54       | 2       | Accuracy |
| MiniBooNE  | builtin   | 130,065 | 50       | 2       | Accuracy |
| Helena     | 41169     | 65,196  | 27       | 100     | Accuracy |
| Adult      | 1590      | 48,842  | 14       | 2       | ROC AUC  |

## Requirements

See `requirements.txt`. Main dependencies: PyTorch, scikit-learn, CatBoost, OpenML, Optuna.
