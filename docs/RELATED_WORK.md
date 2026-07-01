# Related work and positioning

This project sits at the intersection of two lines of research: **deep-learning
architectures for tabular data** (which are, to date, *hand-designed* by
researchers) and **LLM-driven neural architecture search** (which so far targets
*vision/code*, not tabular data). Our contribution is to bring an LLM-guided,
reasoning-based search to the tabular domain and show it finds competitive
architectures automatically and cheaply.

## 1. Hand-designed deep architectures for tabular data

Every strong tabular deep model below was designed and tuned *by hand*, over
months of research effort. That is exactly the cost our automated search removes.

| Method | Idea | Reference |
| --- | --- | --- |
| **NODE** | differentiable oblivious decision-tree ensembles | Popov et al., ICLR 2020, arXiv:1909.06312 |
| **TabNet** | sequential attention, instance-wise feature selection | Arik & Pfister, AAAI 2021, arXiv:1908.07442 |
| **TabTransformer** | attention over categorical embeddings + MLP | Huang et al., 2020, arXiv:2012.06678 |
| **FT-Transformer** | feature tokenizer + Transformer; the RTDL benchmark | Gorishniy et al., NeurIPS 2021, arXiv:2106.11959 |
| **SAINT** | row + inter-sample attention, contrastive pre-training | Somepalli et al., 2021, arXiv:2106.01342 |
| **TabPFN** | in-context learning with a pre-trained transformer | Hollmann et al., ICLR 2023, arXiv:2207.01848 |
| **TabM** | parameter-efficient in-network ensembling of MLPs | Gorishniy et al., 2024, arXiv:2410.24210 |

Two survey/benchmark works frame the field: *A Survey on Deep Tabular Learning*
(2024, arXiv:2410.12034) catalogs the architecture zoo, and Grinsztajn et al.,
*Why do tree-based models still outperform deep learning on typical tabular data?*
(NeurIPS 2022, arXiv:2207.08815) shows there is **no universal winner** — the best
model depends on the dataset. This is precisely the motivation for an automatic,
per-dataset architecture search.

**Our search space reuses this literature as building blocks** (MLP, ResMLP,
FT-Transformer, GatedTab, AutoInt, TabM). The LLM does not invent new layers; it
*composes and configures* these families per dataset — the decision that normally
requires a human expert.

## 2. LLMs for neural architecture search

Recent work uses LLMs to drive architecture search, but almost exclusively on
**image/code** benchmarks:

| Method | Role of the LLM | Domain | Reference |
| --- | --- | --- | --- |
| **GENIUS** | GPT-4 as a black-box optimizer proposing/refining nets | vision | Zheng et al., 2023, arXiv:2304.10970 |
| **EvoPrompting** | LLM as evolutionary mutation/crossover over code | vision (MNIST-1D, CLRS) | Chen et al., NeurIPS 2023, arXiv:2302.14838 |
| **LLMatic** | LLM operators + Quality-Diversity optimization | vision (NAS-Bench-201) | Nasir et al., GECCO 2024, arXiv:2306.01102 |
| **OPRO** | LLM as a general optimizer from natural-language traces | generic | Yang et al., ICLR 2024, arXiv:2309.03409 |

**Gap.** None of these target tabular data, and none combine an LLM controller
with a *cheap multi-fidelity screen* tailored to the fast-training tabular regime.
Our method (a) operates on tabular classification **and** forecasting, (b) adds a
5-epoch / 30 %-data proxy screen so the LLM's 20 proposals cost ~10 % of full
training, and (c) uses an explicit `Reflect` step where the LLM rewrites its
search strategy from the trial history.

## 3. Why our result is meaningful (honest framing)

The claim is **not** "a new architecture beats everything." It is:

1. **Automation replaces months of manual design.** FT-Transformer, TabM, SAINT
   etc. are human artifacts. Our LLM-guided search reaches *comparable or better*
   quality per dataset **automatically**, at ≈ \$0.09 LLM + ≈ \$2 GPU per search.
2. **It beats other search methods at equal budget.** Under a fixed budget of 40
   trainings, LLM-guidance directs the search better than Random NAS and Optuna
   TPE on most datasets (see `results/`).
3. **It matches/*exceeds published hand-tuned numbers* on some datasets.** Our
   single-model results reach or surpass the RTDL paper's tuned FT-Transformer on
   Helena and MiniBooNE — i.e., an automatic search rivals expert tuning.
4. **It transfers across domains.** The same controller works on classic tabular,
   high-dimensional sensor data, and multivariate forecasting — including
   *semantically correct* choices (picking a linear NLinear on the near-linear
   Exchange series).

All numbers are regenerated from raw logs by `scripts/build_results_table.py`
and the search-cost comparison by `scripts/search_efficiency.py` — nothing is
hand-entered.

## References

- Popov, Morozov, Babenko. *Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data.* ICLR 2020. arXiv:1909.06312.
- Arik, Pfister. *TabNet: Attentive Interpretable Tabular Learning.* AAAI 2021. arXiv:1908.07442.
- Huang, Khetan, Cvitkovic, Karnin. *TabTransformer: Tabular Data Modeling Using Contextual Embeddings.* 2020. arXiv:2012.06678.
- Gorishniy, Rubachev, Khrulkov, Babenko. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021. arXiv:2106.11959.
- Somepalli et al. *SAINT: Improved Neural Networks for Tabular Data.* 2021. arXiv:2106.01342.
- Hollmann, Müller, Eggensperger, Hutter. *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* ICLR 2023. arXiv:2207.01848.
- Gorishniy, Kotelnikov, Babenko. *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling.* 2024. arXiv:2410.24210.
- Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 2022. arXiv:2207.08815.
- *A Survey on Deep Tabular Learning.* 2024. arXiv:2410.12034.
- Zheng et al. *Can GPT-4 Perform Neural Architecture Search?* 2023. arXiv:2304.10970.
- Chen, Dohan, So. *EvoPrompting: Language Models for Code-Level Neural Architecture Search.* NeurIPS 2023. arXiv:2302.14838.
- Nasir et al. *LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization.* GECCO 2024. arXiv:2306.01102.
- Yang et al. *Large Language Models as Optimizers.* ICLR 2024. arXiv:2309.03409.
