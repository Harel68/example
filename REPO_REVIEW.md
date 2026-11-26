# Repository review — Spec_Null (before journal submission)

Scope: every file in `Spec_Null-main` was read, including both notebooks cell by cell and the paper PDF (14 pages).
The parts that can run without hours of compute were **actually re-run in a clean environment** (Python 3.13, NumPy 2.5, PyTorch 2.15, Optuna 5.1).

---

> **Second pass (latest `non_linear_eq3.ipynb`):** everything up to Solution C is now reproducible. The details are in `REPRODUCIBILITY_REPORT.md`.
> Resolved: Fig. 3–4 (`scripts/single_null.py`), waveform export (`scripts/export_waveform.py`, bit-exact), fast seeded SER (`scripts/ser_monte_carlo.py`), Fig. 10 plotting (`scripts/make_rmse_figure.py`).
> Still open: Solution C (out of scope), committing the RMSE CSVs, and choosing between regenerated and committed A/B (A2).

## 0. What was verified by re-running

| Paper item | Result of independent re-run | Verdict |
|---|---|---|
| Table I (β weights, lr of A/B) | Recomputed from `optuna_study.db`: A = trial #1259, B = trial #871. **Every percentage matches to 0.1 %.** | ✅ |
| §IV-D "1 344 trials, NSGA-II, pop. 50, β∈[10⁻²,10²], lr∈[10⁻⁴,10⁻¹]" | `optuna_study.db`: 1 344 COMPLETE + 3 RUNNING trials, same distributions | ✅ |
| Optuna trials #1259, #871 | Notebook objective (cell 14, zero start) re-run: objectives equal the DB to **2·10⁻¹¹ / 2·10⁻⁷ dB** | ✅ bit-level |
| Fig. 6 | Notebook cells 17 + 26 (original), and `scripts/make_pareto_figure.py` (same normalization) | ✅ |
| Fig. 3–4 (single null) | `scripts/single_null.py`: anchors match exactly; PSLR/ISLR improvements 0.37/0.42 dB as in the paper; depth −73.77 dB (paper −73.2) | ✅ |
| Table II, rows LFM / Vouras / A / B | Recomputed exactly from whichever `.npz` files are committed | ✅ |
| Fig. 7, 8 | `scripts/make_52null_figures.py` | ✅ |
| Fig. 9 (SER) | `scripts/ser_monte_carlo.py` on the committed waveforms: all 24 points within 1.6 σ of `SER_results_r1.csv` | ✅ |
| Fig. 10 / 14 | `sensing_operation_claude.py` seeded and deterministic; `scripts/make_rmse_figure.py` plots its CSVs | ✅ code · ⚠️ CSVs not committed |
| Committed `simulation_results_{1259,871}.npz` | **Cannot be regenerated** from anything in the repo (A2) | ⚠️ |
| Table II row C, Fig. 11–13, 14(d) | Solution C — out of scope for this pass | — |

---

## A. Blocking issues — fix before the reviewers see the repo

### A1. Code and data that the paper uses but the repo does not contain
| Missing | Needed for | Status |
|---|---|---|
| LPF-aware optimizer + output (Solution C) | Table II row C, Fig. 11–13, 14(d) | open (your friend has it). It should be built through `export_waveform.py` with the filter in the loss, since C = "B, LPF considered" |
| `simulation_results_one_null.npz` + single-null code | Fig. 3–4, "−40 → −73.2 dB" | ✅ `scripts/single_null.py`. Change the paper to −73.8 dB |
| RMSE CSVs (instead of `RMSE_all.xlsb`) | Fig. 10 | run `sensing_operation_claude.py` per waveform and commit `RMSE_1259.csv`, `RMSE_871.csv`; plot with `scripts/make_rmse_figure.py` |
| `pareto_5000_700_PSLR_ISLR_Nulls.txt`, `isac_study_backup.db` | exploratory notebook cells only | delete those cells |

### A2. The committed A/B waveforms have no reproducible origin
* The study ran from a **zero initial phase** (cell 14), and each trial reproduces exactly. The committed `.npz` files do not match their trials (PSLR of A: −38.63 dB vs the trial's −33.86 dB).
  Ruled out: Vouras start (with your exact starting phase rebuilt to 10⁻¹² rad), null-width terms off, 7-digit parameters, longer runs, the other anchor sets, and any other trial in the DB.
  The files are related to these trials (phase coherence 0.59/0.71 vs ≤0.10 for unrelated trials), but came from a run that is not in the repo.
* This makes the paper internally inconsistent: Fig. 6 marks #1259 at PSLR −33.86 dB, while Table II lists −38.63 dB for A.
* **Recommended fix:** commit the waveforms from `scripts/export_waveform.py --trial 1259 / 871` (`regenerated/` holds them) and update the few affected numbers (table in `REPRODUCIBILITY_REPORT.md` §3). All qualitative claims survive.
* The ill-conditioned linearized baseline (condition number 3·10¹³) only affects `s_lin`. It is now stored once in `vouras_linearized_52.npz`. Replacing `inv` by `solve`/`lstsq` in `getphi()` is still good practice, but cannot make the result machine-independent.
* Notebook trap: **cell 2 sets `s_lin` to the plain chirp**, so running the cells in order exports a wrong "linearized" waveform.

### A3. Files that crash or are stale
| File | Problem | Fix |
|---|---|---|
| `PSLR_ISLR_ISAC.py` | `KeyError: 'isac_signal'` — the npz contains the typo keys `isac_siganl`, `vouras_siganl` | fix the key names or delete the script (its function is duplicated elsewhere) |
| `simulation_results.npz` | stale: a different waveform (PSLR −14.47 dB, linearized `s_lin` differs from the 871/1259 files) | delete |
| `SER_and_RMSE_figures.py` | SER values **hard-coded** (copied from the CSVs); RMSE part needs `pyxlsb` + missing `RMSE_all.xlsb` | replaced by `scripts/make_ser_figure.py` + `scripts/make_rmse_figure.py`; delete |
| `figures_making_one_null.py` | input file missing | ✅ run `scripts/single_null.py` first |

### A4. Things that must not be public
* `IEEE_TAES_ISAC_Waveform_Design.pdf` still contains internal advisor comments, e.g. "[AW: Add references considering these applications]" and "[AW: Have you checked this table?? …]" on page 1. Remove the PDF from the repo. Link to arXiv/DOI once available.
* `docx/Frequency_Modulated_Waveforms_for_Integrated_Imaging_and_Communications.pdf` is the published Vouras (IEEE/Asilomar) paper, which is **copyrighted**. Do not redistribute it; cite it.
* `docx/Preliminary_Report_BSc_Final_Project_Template_Harel_and_Ori.pdf` is an internal course report and not needed. The folder name `docx/` is also confusing (it holds PDFs).
* Notebook outputs leak a personal path (`C:\Users\Elad\AppData\...`) and contain error tracebacks, `KeyboardInterrupt`, and `%pip install` logs.

### A5. Paper ↔ code mismatches to correct (in the paper or the code)
1. **Monte-Carlo trial counts (§IV-F).** The paper says 5 × 10⁴ trials for SNR > 10 dB. The stored SER values only correspond to whole error counts with **4 × 10⁵ frames** for the proposed curves (e.g. 1.15625·10⁻⁵ × 48 × 4·10⁵ = 222 errors exactly; with 5·10⁴ it would be 27.75). The linearized curve used `iteration` (5 × 10³) instead of `iteration_i` from 10 dB upward. Fix the text: "3·10³ (<10 dB), 5·10³ (10 dB), 4·10⁵ (>10 dB) frames for the proposed waveforms".
2. **Trial IDs.** The repo is organised by trial number (871, 1259) but the paper never says A = #1259, B = #871. Fig. 14 text even says "Proposed #871 LPF Considered". Add the trial IDs to Table I.
3. **Fig. 14 text.** "As shown in Fig. 14(c) for the filter-agnostic Solution C" should be **Solution B**.
4. **Fig. 10 axis.** The code's SNR is **per sample** (`noise_power = Ps·10^(−SNR/10)`), and the CRLB in the code uses `E/N0 = N·SNR`. The figure axis says ‖x_s‖²/N₀, which would be 10·log₁₀(600) = **27.8 dB** higher than the per-sample value. Make the label and Eq. (11) consistent.
5. **Fig. 10 receiver filter.** `sensing_operation_claude.py` applies the `[0, B]` LPF before the RMSE Monte Carlo, but §IV-G does not mention it and §IV-H introduces the filter as a new element. Confirm which signals Fig. 10 used and say so.
6. **Two PSLR/ISLR definitions.** Table II columns 1–2 use the optimizer's fixed ±1/B mainlobe; columns 3–4 use a mainlobe widened to the first nulls. Unfiltered ISLR differs by up to 0.1 dB between them (−9.83 vs −9.76 dB for the chirp). State the definition used for each column.
7. **Single-null depth.** Paper: −73.2 dB; old README: −74.91 dB. The reproducible value is **−73.8 dB** (`scripts/single_null.py`; linearized −39.8 dB).
8. **Initial condition of the Pareto search** is the zero phase (verified by exact re-runs). §IV-D should say so; the paper mentions the Vouras start only for the single null.
9. **Null-width anchors** are 0.7 % / 1 % off "exactly one" at the start: they were computed with the `(1/Tⁿ)∫…t dt` form, while the loss uses `t_norm = t/max(t)`.

---

## B. Duplicate code (same logic written several times)

| Logic | Copies | Note |
|---|---|---|
| System parameters `T, B, fs, N, t, ψ, chirp` | **10** (`version1`, `OFDM_2`, `non_linear_perf`, `figures_making_one_null`, `PSLR_ISLR_ISAC`, `sensing_operation`, `sensing_operation_claude`, both notebooks, `test_ofdm_visualization_full`) | Two time grids are used (`linspace(…, endpoint=False)` vs `arange(0, T, 1/fs)`); they agree for N = 600 only by luck of floating point. Two chirp formulas, `πbt²` and `2π(b/2)t²`. → one `config.py` |
| PSLR / ISLR | **7** (`version1.calculate_mf_and_pslr`, `PSLR_ISLR_ISAC` ×2, `sensing_operation.analyze_signal`, `sensing_operation_claude`, notebook torch, notebook numpy cell 19) | **Three different mainlobe definitions** (±1/B, first-null, ±3×(3-dB width)) → one `metrics.py` with a NumPy and a Torch version of the same definition |
| `configure_ieee_style()` | 5 | → `plot_style.py` |
| Matched filter / PSD | 6+ (`helper.apply_matched_filter`, `calc_mf_db`, `calc_psd`, inline FFT code) | → `metrics.py` |
| OFDM `channel` / `receiver` / `SER_Comp` | 2 verbatim copies (both notebooks) | → `comms.py` |
| Null-width loss integrals | copy-pasted in 6 notebook cells (with `t` vs `t_norm` variants — see A2) | → one function |
| Guard / pilot / DC index lists | 6 places | → constants in `config.py` |
| `getphi()` | duplicates the module-level code of `version1.py` line for line | keep only the function |
| `matrix_inverse` | `helper_functions.py` and `null_width_control.py` | and it should be `solve`, not `inv` (A2) |
| Sensing evaluation | `sensing_operation.py` (old, slow, no interpolation, wrong SNR grid) vs `sensing_operation_claude.py` (the one used) | delete the old one |

## C. Dead or irrelevant code / files

| Item | Status |
|---|---|
| `__pycache__/` (includes `.pyc` of deleted modules `null_depth_ver_2`, `nulling_ver2`) | delete, add `.gitignore` |
| `optimal_phasor.npy`, `optimal_phasor_{zero,random,vouras}.npy` + `non_linear_perf.py` | 3-null (0.2/0.3/0.4 MHz) exploration, not in the paper → `archive/` or delete. Note: `optimal_phasor.npy` ≠ `optimal_phasor_vouras.npy` despite the name |
| `sensing_operation.py` | superseded (B) |
| `calculate_roc_for_signal___`, `calculate_mf_pslr_islr_numpy_`, `monte_carlo_roc_rmse_heavy`, `matched_filter_bank`, `plot_ambiguity_function_3d`, `create_c`, `create_s` | never called |
| `non_linear_eq3.ipynb` cells 2, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21, 22 | earlier Optuna variants and plotting attempts. Cell 11 crashed; cell 13 reuses the name `"beta_islr"` for four different parameters (a bug); cell 12 adds a manual "+6.08 dB PSLR correction" to old results; cell 15 takes `log10` of values that are already in dB → NaN tables in the outputs. **Delete before release.** Reviewers will read them. |
| `ofdm_creation.ipynb` cells 7, 9, 10, 13 | experiments. Cell 10 uses `SER_VOURAS`/`SER_val_vouras` from earlier hidden state |
| `OFDM_2.py` bottom | `test_ofdm_visualization_full()` runs **at import time**: every `import OFDM_2` opens a plot and overwrites the global `T`. Move it under `if __name__ == "__main__":` |
| `version1.py` top | computes and prints a 0.4 MHz null at **import time** (the "Null at around 0.40 MHz" line in the notebooks). Move it under `__main__` |
| `null_depth_control.py`, `null_width_control.py` | relevant only to the linearized baseline (App. A/B); keep, but in a `baselines/` folder |

## D. Efficiency — measured hot spots (all replacements verified to give identical numbers)

| Hot spot | Now | Vectorized | Speed-up | Impact |
|---|---|---|---|---|
| `OFDM()` + `OFDM_demodulate()` per frame (Python loop over 64 subcarriers, `np.exp` recomputed every call) | 3.02 ms | 19.9 µs (precompute 64×600 basis `E` once; `X @ E`, `R @ E.conj().T`) | **×152** | SER for one waveform (8.2·10⁵ frames): ≈ 45 min → about a minute |
| `QPSK_modulation()` (loop + `np.array_equal`) | 381 ms / 192k bits | `((1-2b₀) + 1j(1-2b₁))/√2` | **×231** | |
| `inner_product_mat()` (double Python loop of `trapezoid`) | 97 ms | `A.T @ (A * w[:,None])`, `w` = trapezoid weights | **×98** | |
| Null-width loss: 2 × 52 `torch.trapezoid` calls in list comprehensions per iteration | 5.9 ms each | `torch.trapezoid(z * (ph*t_norm)[:,None], x=t_norm, dim=0)` | **×10** | ≈ 60 % of each Adam iteration (≈ 12 of 20 ms) → each trial ≈ 100 s → ≈ 45 s, full Pareto search ≈ 2× faster |
| SER Monte Carlo | one frame at a time | batch 5 000 frames as a matrix (noise `(F,600)`, demod `R @ E^H`) | — | lets 4·10⁵-frame points run in seconds |
| RNG | legacy `np.random.seed` in comms, `default_rng` in sensing | use `np.random.default_rng(seed)` everywhere and pass it explicitly | — | reproducible, order-independent |

Smaller items: `build_D` builds a dense (N−1)×N matrix just to take differences (`np.diff` is enough). `sensing_operation_claude.py` writes the CSV name `RMSE_1259_new.csv` but its docstring says `RMSE_871.csv`, and its ambiguity title is hard-coded to "(871)". `version1.py` labels a degree plot "(rad)".

## E. Readability for international reviewers
* Translate the Hebrew comments and docstrings (`helper_functions.py`, `figures_making_one_null.py`, `SER_and_RMSE_figures.py`, `sensing_operation_claude.py`, both notebooks).
* Rename files after what they do: `OFDM_2.py` → `ofdm.py`, `version1.py` → `linearized_baseline.py`, `non_linear_eq3.ipynb` → `optimize_waveforms.ipynb`, `ofdm_creation.ipynb` → `communication_ser.ipynb`, `sensing_operation_claude.py` → `sensing_evaluation.py` (a file name should describe its content, not the tool used to write it). The README file names must be updated together with any rename.
* Notebooks have **no Markdown cells** and were executed out of order (execution counts 2, 18, –, 4, 14, 15, 198, …). Add a heading cell per section, strip outputs (`nbstripout`), and check that "Restart & Run All" works.

## F. Suggested target layout

```
Spec_Null/
├── README.md  LICENSE  requirements.txt  .gitignore  Makefile
├── spec_null/                  # importable, no side effects on import
│   ├── config.py               # T, B, fs, N, N_FFT, OFDM layout, c, target range
│   ├── signals.py              # chirp, OFDM, QPSK              (from OFDM_2.py)
│   ├── metrics.py              # ONE PSLR/ISLR (numpy + torch), PSD, matched filter, null energy
│   ├── linearized.py           # getphi, CG depth control, width control
│   ├── optimizer.py            # loss terms + Adam loop          (from notebook cells 3/4/14)
│   ├── comms.py                # channel, receiver, vectorized SER Monte Carlo
│   ├── sensing.py              # LPF, RMSE/ROC Monte Carlo, CRLB, ambiguity
│   └── plot_style.py
├── experiments/
│   ├── 01_pareto_search.py     # → results/optuna_study.db
│   ├── 02_export_waveform.py   # --trial 1259 → results/simulation_results_1259.npz
│   ├── 03_single_null.py  04_lpf_aware.py
│   ├── 05_ser.py               # → results/SER_results_r1.csv
│   └── 06_sensing.py           # → results/RMSE_*.csv, Table II
├── figures/  make_fig3_4.py … make_fig14.py  make_all.py
├── results/                    # committed reference artifacts (npz, db, csv)
└── notebooks/                  # optional, clean, annotated walkthrough
```
A `Makefile` with `make verify` (Level 1, minutes) and `make all` (everything) is what reviewers love most.

## G. Priority checklist

**P0 — before submission**
- [ ] Remove the manuscript PDF with "[AW: …]" comments, the Vouras PDF, and the BSc report (A4)
- [ ] Commit the LPF-aware optimizer + output (Solution C) and the RMSE CSVs (A1)
- [x] Single-null run + `simulation_results_one_null.npz` → `scripts/single_null.py` (copy from `regenerated/`)
- [x] Export script that turns any trial into its waveform, bit-exact → `scripts/export_waveform.py`
- [ ] Commit the regenerated `simulation_results_{1259,871}.npz` and update the affected paper numbers (A2, REPRODUCIBILITY_REPORT §3)
- [x] Store the linearized baseline once → `vouras_linearized_52.npz`
- [ ] Delete `simulation_results.npz`; fix or delete `PSLR_ISLR_ISAC.py` (A3)
- [ ] Clean both notebooks: delete exploratory cells, strip outputs, Restart & Run All (C)
- [ ] Fix the paper text: MC counts, zero initial phase, trial IDs in Table I, Fig. 14 "Solution C"→"B", Fig. 10 axis/filter, single null −73.8 dB (A5)
- [ ] Add `LICENSE`, `requirements.txt`, `.gitignore`; copy the new `README.md`, `scripts/`, `docs/images/`, `vouras_linearized_52.npz` and the `regenerated/` data files into the repo

**P1 — strongly recommended**
- [ ] One `config.py`, one `metrics.py`, one `plot_style.py` (B)
- [ ] No side effects on import (`OFDM_2.py`, `version1.py`) (C)
- [ ] Rename files, translate comments (E)

**P2 — nice to have**
- [ ] Vectorize OFDM/SER and the null-width loss (D)
- [ ] `Makefile` with `make verify`
