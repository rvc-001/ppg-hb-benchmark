# Multiwavelength PPG Hemoglobin Benchmark

Code and diagnostic figures supporting the study *Baseline-Calibrated, Leave-One-Subject-Out Benchmarking of Multiwavelength Photoplethysmography for Non-Invasive Hemoglobin Estimation: A Quantum-Inspired Case Study* (submitted to **Computers in Biology and Medicine**).

This repository contains the analysis and feature extraction code and the diagnostic figures. It does not include the manuscript source, raw participant recordings, intermediate caches, or subject-level prediction files.

## Included files

- `run_comprehensive_evaluation_200hz.py`: extracts features from the public dataset, saves an intermediate cache, and runs the 200 Hz evaluation.
- `bp_features.py` and `bp_quantum_features.py`: signal and quantum-inspired feature extraction used by the analysis.
- `audit_and_diagnostics.py` and `generate_publication_figures.py`: diagnostic and figure-generation scripts. These require the intermediate cache and, for some outputs, the corresponding evaluation results described below.
- `figures/`: diagnostic figures associated with the study.
- `requirements.txt`: Python dependencies.

## Data and setup

The raw PPG dataset is not included. Obtain it from the [Figshare dataset](https://doi.org/10.6084/m9.figshare.22256143.v5) and follow its access and reuse terms. The evaluation script expects the extracted files in `Hb_PPG_Dataset/data_mat/` and the participant information workbook at `Hb_PPG_Dataset/subject information.xlsx`.

Use Python 3.10 or later, install dependencies, and run the evaluation from the repository root:

```bash
python -m pip install -r requirements.txt
python run_comprehensive_evaluation_200hz.py
```

The evaluation writes `features_cache_200hz.pkl` and aggregate results under `final_results_full_demo/`; generated diagnostic plots are written under `final_plots_full_demo/`. These intermediate files are not committed.

The diagnostic scripts use the cache and some also expect per-subject prediction CSVs under `final_results_full_demo/`. They are provided for the analysis workflow and require those inputs to be available before execution.

## Reproducibility and privacy

Do not commit the raw dataset or participant-level records. Review dataset terms and applicable ethics/privacy requirements before sharing any derived participant-level data. The repository intentionally omits subject-level predictions and intermediate caches.
