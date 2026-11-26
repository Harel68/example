# Reproducibility report — everything up to Solution C

*Second pass, using the latest `non_linear_eq3.ipynb`. Solution C (LPF-aware) is deliberately out of scope.*

## 1. Bottom line

| Stage | Status | How it was checked |
|---|---|---|
| Pareto search → `optuna_study.db` (Fig. 6, Table I) | ✅ **bit-level reproducible** | Re-ran trials #1259 and #871 with the notebook's own objective (cell 14): objectives equal the database to **2·10⁻¹¹ dB** and **2·10⁻⁷ dB**, on a different OS and PyTorch version |
| Export of A and B as waveforms | ✅ with the new `scripts/export_waveform.py`<br>⚠️ **not** for the currently committed `.npz` files | The script is bit-identical to the notebook objective and reruns give identical files. The committed files were made by a run that is not in the repo (§2) |
| Single null (Fig. 3–4) | ✅ new `scripts/single_null.py` | Anchors in the notebook equal the loss terms at the start **exactly**. PSLR/ISLR improve by **0.37 / 0.42 dB, exactly as in the paper**. Null depth **−73.77 dB** (paper: −73.2 dB) |
| SER (Fig. 9) | ✅ new `scripts/ser_monte_carlo.py` (2 min instead of ~2.5 h) | Same transmitter/receiver as the notebook to 7·10⁻¹⁵. On the committed waveforms all 24 points agree with `SER_results_r1.csv` within 1.6 σ. Seeded, so two runs are byte-identical |
| Table II (no filter / LPF) | ✅ | Recomputed exactly from whichever waveform files are committed |
| RMSE (Fig. 10), ambiguity (Fig. 14 a–c) | ✅ code is seeded and deterministic · ⚠️ RMSE CSVs not committed | New `scripts/make_rmse_figure.py` plots from the CSVs the sensing script writes (replaces the missing `RMSE_all.xlsb`) |

## 2. The one real problem: the committed A/B waveforms have no reproducible origin

The study was run from a **zero initial phase** (the latest cell 14: `phi_start = torch.zeros(N)`). Re-running a trial reproduces the database exactly.
The committed `simulation_results_1259.npz` / `_871.npz` are **not** those trials' waveforms:

| | null energy | PSLR | ISLR |
|---|---|---|---|
| Optuna DB, trial #1259 = exact re-run | −17.51 dB | −33.86 dB | −11.50 dB |
| committed `simulation_results_1259.npz` (paper's "A") | −16.92 dB | −38.63 dB | −15.83 dB |
| Optuna DB, trial #871 = exact re-run | 36.14 dB | −39.64 dB | −20.21 dB |
| committed `simulation_results_871.npz` (paper's "B") | 36.27 dB | −40.32 dB | −19.97 dB |

What was ruled out, each by re-running:
* Vouras start (with your machine's exact starting phase rebuilt from the stored `s_lin` to 10⁻¹² rad), with null-width terms on and off.
* Parameters rounded to 7 digits as in the old cell 4.
* Longer runs (checkpoints every 1 000 iterations up to 36 000).
* The two other anchor sets commented in cell 4.
* Any other trial: no trial in the database has those objective values.

The committed files *are* related to these trials. Their phase coherence with the zero-start re-run of the same trial is 0.59 (#1259) and 0.71 (#871), versus 0.03–0.10 between unrelated trials. So they came from these parameters through a modified run that is not in the repo.

This also makes the paper **internally inconsistent today**: Fig. 6 (cell 26) marks #1259 at its database value (PSLR −33.86 dB), while Table II reports −38.63 dB for "solution A".

## 3. Recommendation: switch A and B to the exactly reproducible export

`scripts/export_waveform.py --trial 1259` and `--trial 871` regenerate A and B. Each takes ~70 s, checks itself against the database, and is bit-identical between runs.
Every qualitative claim of the paper survives. These are the numbers that change:

| Paper item | Now in paper | With regenerated A/B |
|---|---|---|
| Table II, Opt. A — PSLR / ISLR | −38.63 / −15.83 dB | **−33.86 / −11.50 dB** |
| Table II, Opt. A — with LPF | −1.50 / 1.48 dB | **−1.80 / 1.52 dB** |
| Table II, Opt. B — PSLR / ISLR | −40.32 / −19.97 dB | **−39.64 / −20.21 dB** |
| Table II, Opt. B — with LPF | −2.83 / 0.04 dB | **−2.85 / 0.20 dB** |
| §IV-H text: "B's PSLR rises from … to …, ISLR from … to …" | −40.32→−2.83, −19.97→0.04 | −39.64→−2.85, −20.21→0.20 |
| SER at 14 dB, A | 7.29·10⁻⁷ | **5.73·10⁻⁷** (11 errors in 1.92·10⁷ symbols) |
| SER at 14 dB, B | 1.16·10⁻⁵ | **8.54·10⁻⁶** |
| SER at 14 dB, linearized | 1.33·10⁻² | 1.37·10⁻² (new seeded run; statistically the same) |
| "over four orders of magnitude" (abstract, §IV-F) | ✔ | ✔ still true (1.37·10⁻² / 5.73·10⁻⁷ ≈ 2.4·10⁴) |
| Single null depth | −73.2 dB | **−73.8 dB** (reproducible value) |
| Fig. 7, 8, 9, 14 | — | regenerate with the scripts (already done for 7, 8, 9: `regenerated/figures/`) |
| Fig. 10 | — | rerun `sensing_operation_claude.py` (≈4.4 h per waveform). A quick check (10⁵ trials, 10 SNR points) shows the same curves: asymptotes within 2 %, A even slightly better in the threshold region |
| Fig. 6, Table I | — | unchanged (they already describe the database trials) |
| C vs B improvement (18.35 / 17.63 dB) | — | depends on C. The paper defines C as "B, LPF considered", so C should be built through the same export path |

**Alternative:** keep the committed files and have the person who made them commit the exact code/settings that produced them. Only do this if that run can be found; otherwise the waveform stage of the paper stays unreproducible.

## 4. Paper text corrections (needed either way)

1. **Initial condition of the Pareto search: zero phase.** The paper states the Vouras start only for the single null (correct) and never states it for the 52-null study. Add it to §IV-D. The loss *anchors* are evaluated at the Vouras phase (checked exactly), which is what §IV-A says.
2. **Monte-Carlo frames:** 3·10³ (< 10 dB), 5·10³ (10 dB), 4·10⁵ (> 10 dB) for the proposed waveforms, and 5·10³ from 10 dB up for the linearized one. The paper says 5·10⁴.
3. **Trial IDs in Table I:** A = #1259, B = #871.
4. Fig. 14 text: "filter-agnostic Solution C" → B. Fig. 10 axis label vs per-sample SNR (see REPO_REVIEW A5).
5. Minor: the two null-width anchors are not exactly "one" at the start (ratio (600/599)² and (600/599)³). They were computed with the old `(1/Tⁿ)∫…t dt` form while the loss uses `t_norm = t/max(t)`. Either recompute them or ignore it; it only rescales β_deriv1/β_deriv2 by 0.7 % / 1 %.

## 5. Notebook traps found in this pass
* **Cell 2 overwrites `s_lin` with the plain chirp** (`s_lin = s_base * exp(1j*0)`). Running cells 0→6 in order therefore exports a "linearized" waveform that is the unmodified chirp. The export script avoids this by reading `s_lin` from `vouras_linearized_52.npz`.
* Cell 14 as committed runs `population_size=2, n_trials=2` in memory; the paper's study needs `population_size=50, n_trials=1344, storage="sqlite:///optuna_study.db"`.
* Cell 4's current β values belong to the 2-trial test study (trial 0), not to A or B.
* `vouras_linearized_52.npz` (new) stores the linearized waveform once. Its 104×104 Gram matrix has condition number 3·10¹³, so recomputing it on another machine changes it at the 10⁻⁶ level. Only the Vouras baseline is affected by this; the Pareto search is not, because it starts from zero.

## 6. Commands — full reproduction up to Solution C

```bash
python scripts/export_waveform.py --trial 1259        # A   (~70 s, prints PASS)
python scripts/export_waveform.py --trial 871         # B   (~70 s, prints PASS)
python scripts/single_null.py                         # single null (~2 min)
python scripts/ser_monte_carlo.py                     # SER_r1.csv (~2 min)
python scripts/make_pareto_figure.py                  # Fig. 6 + Table I
python scripts/make_52null_figures.py                 # Fig. 7, 8 + Table II (no filter)
python scripts/make_ser_figure.py                     # Fig. 9
python figures_making_one_null.py                     # Fig. 3, 4
python sensing_operation_claude.py                    # Table II (LPF), Fig. 14, RMSE CSV   (set FILENAME / RMSE_CSV_FILENAME per waveform)
python scripts/make_rmse_figure.py --a RMSE_1259.csv --b RMSE_871.csv     # Fig. 10
```
The outputs of every step except the two long RMSE runs are in `regenerated/`.
