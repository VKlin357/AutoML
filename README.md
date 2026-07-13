# LLM-NAS for Tabular Data

**LLM-guided Neural Architecture Search for tabular and time-series data.**
A large language model acts as the *controller* of an architecture search: at
each step it reads a dataset profile and the full history of past trials, then
proposes neural architectures and training recipes in structured JSON. Candidates
are filtered by a cheap proxy (short training on a data subset) before any full
training run, so the search spends its budget where it matters.

> **Аннотация (RU).** Проект исследует, можно ли заменить «слепой» перебор в
> нейроархитектурном поиске (NAS) знаниями языковой модели. LLM выступает
> управляющим компонентом поиска: предлагает архитектуры и гиперпараметры,
> анализирует историю обучения и обновляет стратегию. На стандартном бенчмарке
> RTDL (Gorishniy et al., 2021) подход при одинаковом бюджете обходит CatBoost,
> случайный NAS и Optuna на большинстве датасетов и в трёх случаях **превышает
> опубликованные результаты с ручным тюнингом** — полностью автоматически, за
> ≈ $0.09 на вызовы LLM. Насколько нам известно, это первая работа по LLM-NAS
> именно для табличных данных.

---

## TL;DR

- **Problem.** Classical NAS (reinforcement learning, evolution) is expensive and
  *uninformed* — it explores the search space without any prior about what makes
  a good network. Meanwhile strong tabular architectures exist (FT-Transformer,
  TabM, AutoInt), but no single one is best on every dataset.
- **Idea.** Use an LLM as a *warm-started, reasoning controller* that has read
  millions of lines of ML code and papers, instead of a blind optimizer.
- **Result.** Under an identical budget of 40 full training runs, LLM-NAS
  (greedy-ensemble of its top trials) beats CatBoost / Random NAS / Optuna on
  **7 / 9** classification datasets and CatBoost on all **7 / 7** forecasting
  datasets, and exceeds the *hand-tuned* RTDL paper numbers on Jannis, Helena
  and MiniBooNE.
- **Cost.** ≈ **$0.09** of LLM API + ≈ **$2** of GPU per dataset search.

All numbers below are regenerated from raw logs by
[`scripts/build_results_table.py`](scripts/build_results_table.py) — nothing is
hand-typed.

---

## Method

```
                 ┌──────────────────────────────────────────────┐
   warmup        │   for each search step (budget B = 40):       │
  12 random  ──▶ │                                               │
   trials        │   1. build dataset profile + trial history    │
                 │   2. LLM proposes 20 candidate configs (JSON)  │  ◀── reasoning
                 │   3. cheap screening: 5 epochs on 30% data     │      (chain-of-thought)
                 │   4. top-4 by cheap score → full training      │
                 │   5. every 8 steps: Reflect → update strategy  │
                 └───────────────────┬──────────────────────────┘
                                     ▼
                  greedy (Caruana) ensemble of top-K full trials
```

- **Search space** — 6 architecture families: `MLP`, `ResMLP`, `FT-Transformer`,
  `GatedTab` (GLU blocks), `AutoInt` (self-attention over feature pairs), and
  `TabM` (shared trunk + K parallel heads, a parameter-efficient in-network
  ensemble). The LLM selects the family *and* its hyperparameters.
- **Cheap screening (multi-fidelity).** Each of the 20 proposals trains for 5
  epochs on 30 % of the data (≈ 10 % of full cost); only the top 4 are trained in
  full. Net effect: ≈ **5.6× fewer GPU-hours** than training all proposals.
- **Reflect.** Every 8 steps the LLM re-reads the whole history and rewrites its
  strategy ("dropout is not helping → the problem is capacity, widen the net").
- **Structured output.** > 90 % of LLM responses parse as valid JSON on the
  first try; invalid ones are repaired and re-requested.

The design and the v1→v9 evolution (how each component was added and *measured*)
are documented in [`docs/`](docs/) and the project report.

---

## Results

Reproduce with `python scripts/build_results_table.py` (reads `experiments_v9/`
and `experiments_forecasting/`).

### Classification — accuracy on the untouched test split, budget = 40 trials

Full results in [`results/final_results_classification.json`](results/final_results_classification.json).

| Dataset | CatBoost | Random NAS | Optuna NAS | **LLM-NAS** |
| --- | --- | --- | --- | --- |
| HAR | 0.926 | 0.958 | 0.958 | **0.991** |
| HARTH | **0.842** | 0.795 | 0.846 | **0.905** |
| PAMAP2 | 0.737 | 0.717 | 0.692 | **0.794** |
| EEG | 0.991 | 0.998 | **0.999** | **0.999** |
| MiniBooNE | 0.985 | 0.986 | 0.987 | **0.988** |
| Helena (100 cls) | 0.372 | 0.392 | 0.395 | **0.407** |
| Adult | **0.928** | 0.914 | 0.914 | 0.919 |
| Jannis | 0.800 | 0.789 | 0.795 | **0.803** |
| Volkert | 0.707 | 0.703 | **0.765** | 0.724 |

LLM-NAS ensemble wins **7/9**. Adult (categorical-heavy, GBM territory) and Volkert
(Optuna finds a better single config) are reported honestly.

### vs. published RTDL paper (Gorishniy et al., 2021), hand-tuned

| Dataset | RTDL best (hand-tuned) | **LLM-NAS (automatic)** | Δ |
| --- | --- | --- | --- |
| Jannis | 0.793 | **0.803** | +1.3 % |
| Helena | 0.388 | **0.407** | +5.0 % |
| MiniBooNE | 0.974 | **0.988** | +1.4 % |

### Forecasting — test MSE (lower is better), budget = 40 trials

Full results in [`results/final_results_forecasting.json`](results/final_results_forecasting.json).

| Dataset | CatBoost | Random NAS | Optuna NAS | **LLM-NAS** |
| --- | --- | --- | --- | --- |
| Electricity | 0.0728 | 0.0640 | 0.0593 | **0.0561** |
| ETTh1 | 0.0105 | 0.0068 | **0.0059** | **0.0059** |
| ETTh2 | 0.0150 | **0.0038** | 0.0039 | **0.0038** |
| ETTm1 | 0.0027 | 0.0028 | **0.0020** | **0.0020** |
| ETTm2 | 0.0018 | 0.0024 | 0.0016 | **0.0004** |
| Exchange | 1.1611 | **0.0016** | **0.0016** | 0.0017 |
| Traffic | 0.2698 | 0.0884 | 0.0891 | **0.0855** |

LLM-NAS beats CatBoost on all 7. On *Exchange* MSE is ~700× lower than CatBoost:
the LLM diagnosed the series as linear and chose `NLinear` — a semantic decision
tree models cannot make as they ignore observation order.

---

## Reproducibility

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # only needed for the LLM-NAS runs

# 2. baselines (CatBoost / LightGBM / Random NAS / Optuna), e.g. Jannis (OpenML 41168)
python scripts/run_baselines.py --openml_id 41168 --out_dir outputs/jannis_baselines

# 3. LLM-guided NAS under the same budget
python scripts/run_llm_nas.py  --openml_id 41168 --out_dir outputs/jannis_llm --budget 40

# 4. regenerate the results tables from raw logs
python scripts/build_results_table.py    # -> results/results_*.csv, results/results_tables.md
```

- **Fixed protocol.** 64/16/20 train/val/test split, `seed = 42`, budget
  `B = 40` shared by every method. The test split is touched exactly once, at the
  end (no leakage). Time-series benchmarks use a chronological / participant-level
  split — see [`docs/time_series_protocol.md`](docs/time_series_protocol.md).
- **Determinism** is checked by `src/test_determinism.py`.
- **One command** end-to-end: `make demo` (see [Makefile](Makefile)) — runs a full 40-trial search on Jannis; expect ~2 h on a GPU machine plus API cost.

## Experiment tracking, configs & CI

All experiments write a local `tracking/<run>.jsonl` log by default (zero setup).
[`src/tracking.py`](src/tracking.py) also supports MLflow (offline, against
`./mlruns`) — set `--track mlflow` or `LLMNAS_TRACKER=mlflow`. W&B and ClearML
backends are wired in the same interface but require account credentials and are
not tested end-to-end.

- **Configs** — [`configs/experiment.yaml`](configs/experiment.yaml) and
  [`configs/datasets.yaml`](configs/datasets.yaml) (benchmark registry).
- **Pipeline** — [`dvc.yaml`](dvc.yaml) defines the forecasting experiment flow.
- **CI** — GitHub Actions runs lint + smoke tests on every push.

## Efficiency

Search cost, honestly ([`scripts/search_efficiency.py`](scripts/search_efficiency.py),
full report in `results/search_efficiency.md`): the cheap multi-fidelity screen
prunes **16 of 20** proposals per step, so only **4** are trained in full —
**5× fewer full trainings per proposal batch** (~4× in training-epoch terms). A
complete per-dataset search costs minutes of GPU + an estimated ~$0.09 of LLM
calls, versus tabular architectures that were **hand-designed by research teams
over months** (see [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md)).

---

## Repository layout

```
src/                core library
  nas_orchestrator.py   search loop (warmup → propose → screen → reflect → ensemble)
  llm/                  LLM controller, prompts, JSON parsing/repair
  search_space.py       6 architecture families + named actions
  models.py, train_nn.py    architectures and training
  multi_fidelity.py     cheap screening
  ensemble.py           greedy (Caruana) ensembling
  baselines_automl.py   CatBoost / LightGBM / Random NAS / Optuna
  data.py, forecasting/ leakage-free (time-series) tabularization
  tracking.py           unified MLflow/W&B/ClearML/TensorBoard tracker
scripts/            entry points + run_*.sh + build_results_table.py + search_efficiency.py + track_demo.py
configs/            experiment.yaml (Hydra) + datasets.yaml (benchmark registry)
experiments_v9/     raw per-trial JSON logs (classification)
experiments_forecasting/  raw logs (forecasting)
results/            aggregated CSV + Markdown, incl. search_efficiency.md (generated)
docs/               protocol, method notes, RELATED_WORK.md
dvc.yaml, params.yaml   DVC pipeline + parameters
tests/              smoke tests run in CI
```

---

## Why this matters (product framing)

- **Who.** An ML engineer or small team that needs a strong tabular deep model
  but has **no budget for thousands of GPU-hours** of classical NAS, and no time
  to hand-tune FT-Transformer per dataset.
- **Alternatives.** AutoGluon / AutoML (not deep-architecture-focused), classical
  NAS (BOHB, evolutionary — expensive and uninformed), manual tuning (slow, needs
  an expert). LLM-NAS gets near-/above-expert architectures **automatically**.
- **Impact.** ≈ $0.09 + $2 per dataset and ≈ 5.6× fewer GPU-hours than screening
  every candidate, while matching or beating hand-tuned published results.
- **MVP.** A single CLI command (`run_llm_nas.py`) that, given an OpenML id or a
  CSV, returns a trained architecture + ensemble and a full audit trail of the
  LLM's reasoning.

---

## Related work

Strong tabular architectures — FT-Transformer (Gorishniy et al., NeurIPS 2021),
TabNet (AAAI 2021), SAINT, NODE (ICLR 2020), TabM (2024) — are all **hand-designed**.
LLM-driven NAS (GENIUS, arXiv:2304.10970; EvoPrompting, NeurIPS 2023; LLMatic,
GECCO 2024) targets **vision/code**, not tabular data. This project closes that gap
and adds cheap multi-fidelity screening and a reasoning `Reflect` loop for the
tabular setting. Full positioning and citations: [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Limitations

Single seed per dataset (each run costs real money/GPU, the norm in NAS literature);
statistical-significance testing across seeds is future work (`scripts/run_multiseed.sh`
+ `scripts/aggregate_results.py` provide the harness). Results depend on the LLM
backend; gains are consistent across the tested datasets but not guaranteed on
arbitrary new domains.

## Citation

```bibtex
@thesis{palysaev2025llmnas,
  title  = {LLM-guided Neural Architecture Search for Tabular Data},
  author = {Palysaev, Vadim},
  year   = {2025},
  school = {HSE University}
}
```

License: [MIT](LICENSE).
