# Current validation practice undermines surgical AI development

This repository contains the code accompanying the paper  **Reinke et al., “Current validation practice undermines surgical AI development”**. It provides fully documented analyses for three experiments that demonstrate common validation pitfalls in surgical AI benchmarking.

---

## Reproducibility and data availability

The analyses in this repository operate on performance results from surgical AI benchmarking benchmarks. These result tables cannot be shared publicly, as participant-level consent for data redistribution was not obtained.

As a consequence:

- All experiments are implemented as Jupyter notebooks.
- Running the notebooks end-to-end requires access to the corresponding anonymized challenge result tables, which can only be granted to reviewers.
- Even without access to these data files, the notebooks display all results, figures, and outputs reported in the paper and allow full inspection of the analysis logic.

No raw images, videos, or annotations are accessed. All analyses are based exclusively on precomputed, tabular performance data.

---

## Repository structure

The repository consists of three Jupyter notebooks, one per experiment:

1. **Experiment 1 – Dependent test samples inflate confidence**  `exp1_dependent-test-samples-inflate-confidence.ipynb`

2. **Experiment 2 – Averages hide critical failures**  `exp2_averages-hide-critical-failures.ipynb`

3. **Experiment 3 – Aggregation choices can flip the winner**  `exp3_aggregation-choices-can-flip-the-winner.ipynb`

Each notebook is self-contained and includes detailed methodological explanations, explicit documentation of data assumptions, all analysis steps required to reproduce the reported figures and conclusions.

---

## Experiment 1: Dependent test samples inflate confidence

This experiment demonstrates how **ignoring hierarchical dependencies** in temporally structured surgical video data leads to **severely underestimated uncertainty**, i.e., overly narrow confidence intervals.

### Core idea

Surgical video data is inherently hierarchical as multiple correlated frames originate from the same patient case (video). We compare two resampling strategies for estimating 95% bootstrap confidence intervals:

- **Naive bootstrap**: resampling individual frames, implicitly assuming independence.
- **Hierarchical bootstrap**: resampling videos/patients first, then frames within each selected video, explicitly accounting for dependencies.

### Tasks and datasets

- **Binary instrument segmentation (RobustMIS 2019)**  
  - 10 challenge submissions  
  - Metrics: Dice Similarity Coefficient (DSC), Normalized Surface Dice (NSD)  
  - Hierarchy: patient/video level (n = 10)

- **Surgical action triplet recognition (CholecT45)**  
  - Precomputed Swin-Base predictions  
  - Metrics: mean Average Precision (mAP), class-weighted mAP, top-5 accuracy  
  - Hierarchy: patient/video level (n = 45)  
  - Cross-validation folds handled separately

---

## Experiment 2: Averages hide critical failures

This experiment shows that **global (non-stratified) aggregation of performance metrics** can conceal **clinically critical failure modes** that only become visible under stratified analysis.

### Core idea

Performance is often summarized as a single global score, implicitly assuming that errors are evenly distributed across conditions. However, rare but safety-critical visual conditions can cause substantial performance drops that are masked by global aggregation.

The experiment contrasts:
- **Non-stratified aggregation**: median performance over all frames.
- **Stratified aggregation**: median performance restricted to frames exhibiting specific visual artifacts.

### Task and dataset

- **Task**: Multi-instance instrument segmentation  
- **Dataset**: RobustMIS 2019 challenge results   
- **Algorithms**: 7 challenge submissions  
- **Metric**: Multi-instance Dice Similarity Coefficient (MI_DSC)

### Artifact-based stratification

Stratification is performed using structured frame-level metadata describing visual artifacts and image properties. Each frame may contain multiple artifacts; subsets are therefore not mutually exclusive. Considered conditions include:

- Blood
- Motion
- Reflections
- Smoke
- Instrument(s) covered by material
- Overexposed instruments
- Underexposed instruments
- Intersecting instruments
- Low-artifact scenes (≤ 1 annotated artifact)

### Uncertainty estimation

Uncertainty of performance differences is estimated using **hierarchical bootstrapping** to calculate confidence intervals (see Experiment 1).

---

## Experiment 3: Aggregation choices can flip the winner

This experiment illustrates how **different, yet reasonable, aggregation strategies** applied to the same fixed results can lead to **substantially different algorithm rankings**, including changes in the apparent winner.

### Core idea

Surgical video analysis data is multi-level (frames, phases, videos). Reported performance scores and rankings depend critically on how results are aggregated across these levels, yet aggregation schemes are often underspecified or omitted in practice.

### Experimental setup

- **Task**: Binary instrument segmentation ()
- **Data**: RobustMIS 2019 challenge results
- **Algorithms**: 10 challenge submissions
- **Metric**: DSC
- **Aggregation operator**: 5th percentile (as used in the original challenge)

Six aggregation strategies are compared, including frame-wise, video-wise, phase-wise, and clinically weighted phase-wise aggregation.

---

## Software dependencies

All notebooks use standard Python libraries for data handling, statistical resampling, metric computation, and visualization. All required imports are explicitly listed at the top of each notebook.

---

## How to run

Each experiment is implemented as a Jupyter notebook.

1. Open the corresponding `.ipynb` file.
2. Run the notebook **top-to-bottom** (`Kernel → Restart & Run All`).

Execution requires access to the underlying challenge result tables. If these files are not available, the notebooks still allow full inspection of the analysis code and display the results reported in the paper.

---

## Citation

If you use this code, please cite:

```bibtex
@article{reinke2025current,
  title={Current validation practice undermines surgical AI development},
  author={Reinke, Annika and Li, Ziying O and Tizabi, Minu D and Andr{\'e}, Pascaline and Knopp, Marcel and Rother, Mika M and Machado, Ines P and Altieri, Maria S and Alapatt, Deepak and Bano, Sophia and others},
  journal={arXiv preprint arXiv:2511.03769},
  year={2025}
}
