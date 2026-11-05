# PAIR-CI replication package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23976940.svg)](https://doi.org/10.5281/zenodo.23976940)

Code, results and table builders for **PAIR-CI: Calibrated Conditional
Independence Testing for Causal Discovery with Incomplete Data** (Thomas S.
Robinson and Ranjit Lall, NeurIPS 2026). Repository:
<https://github.com/tsrobinson/pair-ci>; archived on Zenodo at
<https://doi.org/10.5281/zenodo.23976940> (all versions; v1.0.0 is
10.5281/zenodo.23976941). The same files cover the submitted
version; [Changes since the submitted version](#changes-since-the-submitted-version)
lists what differs. To cite the paper or this code, see `CITATION.cff`.

## Licences and third-party material

The package is released under the MIT License (`LICENSE`), with the exceptions
listed in `NOTICE`; each exception carries an SPDX header.

- `src/baselines_new.py` (FZ-Rubin baseline) ports `micd::gaussMItest` (R package
  micd, GPL (>= 3)) and is therefore GPL-3.0-or-later (`LICENSES/GPL-3.0.txt`).
- `src/notears_linear.py` is modified from the NOTEARS reference code
  (Apache-2.0, `LICENSES/Apache-2.0.txt`); the changes are listed in its header.
- **MissDAG baseline: not redistributed.** `src/missdag_gaussian.py` is derived from
  the authors' public code (https://github.com/ErdunGAO/MissDAG), whose repository
  specifies no licence, so it is excluded from the supplementary material and the
  public repository. Without it, `run_missdag_gaussian` raises an `ImportError`;
  the MissDAG rows of the released result CSVs were produced with it. To rerun
  MissDAG, obtain the original code from its authors.
- **Sachs data: not redistributed.** Run `bash data/fetch_sachs_data.sh` once to
  download the bnlearn copy of the Sachs et al. (2005) data and check it against the
  file used in the paper (853 x 11, value checksum). The ALARM and HAILFINDER
  experiments use only the published graph structures (hard-coded edge lists); all
  their data are simulated.
- The release is built from an explicit list (`release/MANIFEST.txt`) with
  `bash release/make_release.sh OUT.zip`; nothing outside the list is shipped.

## Quick start

```bash
# 1. Environment and data
conda env create -f environment.yml
conda activate causal_ci
python check_env.py                # checks that every dependency imports
bash data/fetch_sachs_data.sh      # Sachs data are not redistributed (see Licences)

# 2. Copy the package to the cluster
scp -r pair-ci_replication/ user@hpc:/scratch/user/

# 3. Submit everything
cd /scratch/user/pair-ci_replication/
chmod +x reproduce_all.sh
./reproduce_all.sh --dry-run       # list the 153 jobs without submitting
./reproduce_all.sh                 # submit (cluster default partition)

# 4. Rebuild every table, figure and checked number from the CSVs
#    (no cluster needed; minutes on a laptop)
python analysis/build_tables.py --output-dir tables/
python analysis/check_rebuttal_numbers.py --results-dir results/
```

The released `results/` holds every CSV the paper uses, so step 4 alone
reproduces the paper's tables and quoted numbers; steps 2–3 regenerate the CSVs
themselves, subject to the caveats under
[Changes since the submitted version](#changes-since-the-submitted-version).
Do not run the smoke test (`--quick`) in this folder: it writes to the same
`results/*.csv` paths (see [Smoke test](#smoke-test)).

## What gets reproduced

Paper references are to the camera-ready: Tables 1–3 are in §5, appendices are
given by letter. Each command is the one that produced the released CSV, taken
from `reproduce_all.sh` or from the `slurm/` script named. A "builder" is
`analysis/build_tables.py --table <name>`, which writes `tables/tab_<name>.tex`;
`tables/TABLE_NOTES.md` describes each fragment.

PAIR-CI's learner depends on the benchmark. On graphs with linear edges (ER
p = 10/20/30 linear, ALARM linear) it uses ridge regression
(`Ridge(alpha=1e-6)`, `src/ci_oracle_linear.py`); on nonlinear edges it uses
forests (random forests in the general variant, ExtraTrees in the fast
variant), as it does on the real Sachs data unless `--learner ridge` is given.

### Experiments in the submitted version

| Script (`experiments/`) | Purpose | Command (as run) | Output (`results/`) | Paper |
|---|---|---|---|---|
| `calibration.py` | Standalone CI-test calibration grid: PAIR-CI, Fisher's Z (complete-case, imputed vote, Rubin), GCM, KCI | `slurm/submit_calibration.sh` (fast methods 100 reps; KCI n=1,000, signals 0/0.6, 50 reps; merge) | `ci_calibration_fast.csv`, `ci_calibration_kci.csv` → `ci_calibration_results.csv` | Table 1 (both panels); App. A (Table `calibration_by_dgp`); App. R.2 (mechanism × n × \|X\|); App. D (power plots); App. L |
| `caching_benchmark.py`, `caching_validation.py` | Cached vs per-query imputation (p = 5; p = 5, 10, 20) | `slurm/submit_caching.sh` | `caching_benchmark_results.csv`, `caching_validation_large_p.csv` | §3.2; App. C (from `caching_validation_large_p.csv`; no analysis script reads `caching_benchmark_results.csv`) |
| `graph_recovery_p10.py` | ER graph recovery, p = 10 (general variant) | `--mechanism M --dgp D --n_graphs 10 --n_datasets 20` | `p10_{dgp}_{mech}.csv` → `p10_{dgp}_results.csv` (`analysis/combine_p10.py`) | Table 2; App. F |
| `graph_recovery_p20.py` | ER graph recovery, p = 20 (fast variant) | as above | `p20_{dgp}_{mech}.csv` → `p20_full_results.csv` (`analysis/combine_p20.py`) | Table 2; App. F |
| `graph_recovery_p30.py` | ER graph recovery, p = 30 (fast variant) | as above | `p30_{dgp}_{mech}.csv` → `p30_full_results.csv` (`analysis/combine_p30.py`) | Table 2; App. F |
| `alarm.py` | ALARM network, linear / nonlinear edges | `--n_obs 1000 --n_reps 20 --dgp {linear,nonlinear}` | `alarm_results.csv`, `alarm_nonlinear_results.csv` | Table 3; App. H |
| `hailfinder.py` | HAILFINDER network (p = 56, nonlinear), 10 per-cell jobs | `--n_obs 1000 --n_reps 20 --dgp nonlinear --rate R --mechanism M` | `hailfinder_{rate}_{mech}.csv` (per-cell intermediates, not shipped) → `hailfinder_results.csv` (`analysis/combine_hailfinder.py`) | Table 3; App. I |
| `sachs.py` | Sachs protein network, real flow-cytometry data | `--n_reps 20 --data data/sachs_data.csv` | `sachs_results.csv` | App. J |
| `sachs_nonlinear.py` | Sachs DAG with synthetic nonlinear edges, 10 per-cell jobs | `--n_obs 1000 --n_reps 20 --dgp nonlinear --rate R --mechanism M` | `sachs_nonlinear_{rate}_{mech}.csv` (per-cell intermediates, not shipped) → `sachs_nonlinear_results.csv` (`analysis/combine_sachs_nonlinear.py`) | App. J |
| `sensitivity.py` | Hyperparameter grid (M, K, trees, leaf) at p = 10, nonlinear, MAR | `--n_graphs 10 --n_datasets 20` | `sensitivity_results.csv` | App. E.1; App. T.2 |
| `robustness.py` | Missingness rate × sample size sweep at p = 10 | `--mechanism M --miss_rate R --n_obs N --n_graphs 10 --n_datasets 20` | `robustness_{mech}_{rate}_{n}.csv` → `robustness_results.csv` (`analysis/combine_robustness.py`) | App. F |
| `alpha_sensitivity.py` | PAIR-CI α sweep at p = 10 | `--n_graphs 10 --n_datasets 20` | `alpha_sensitivity_results.csv` | App. T.1 (frontier); App. T.2 |
| `variance_comparison.py` | Naive vs Bayle within-imputation variance | `--mode both --n_reps 50 --n_graphs 10 --n_datasets 20` | `variance_comparison_calibration.csv`, `variance_comparison_graph.csv` | App. K |
| `early_stopping_calibration.py` | Early-stopping FPR check, fast variant | `--n_reps 100` | `early_stopping_calibration.csv` | App. P |
| `imputation_degradation.py` | MICE vs mean vs marginal imputation | `--n_reps 200 --n_obs 500 --dim_x 5 --seed 42` | `imputation_degradation.csv` | App. Q |
| `new_baselines.py` | FZ-Rubin (micd-pooled) and MissDAG baselines | `--benchmark {erdos,alarm,hailfinder,sachs} ...` (see `reproduce_all.sh`) | `newbase_p{p}_{dgp}_{mech}.csv`, `newbase_alarm_{linear,nonlinear}.csv`, `newbase_hailfinder_nonlinear.csv`, `newbase_sachs_{linear,nonlinear}.csv` | Tables 2–3; App. F, H, I, J |
| `m_sensitivity.py` | Power vs number of imputations M | `--n_reps 50 --Ms 3 5 10 20 --K 5 ...` | `m_sensitivity_results.csv` | App. M; App. T.2 |
| `learner_comparison.py` | RF vs ExtraTrees vs LightGBM vs XGBoost | `--n_reps 50 --learners rf extra_trees lgbm lgbm_tuned xgboost ...` | `learner_comparison_results.csv` | App. O; App. T.2 |
| `adversarial_kappa.py` | Adversarial DGPs stressing the κ-cancellation | `--n_reps 100 --dgps hub chain dense_block weak_hub branch_separator hub_nonlinear ...` (`reproduce_all.sh`; `slurm/submit_adversarial_kappa.sh` asks for 200, see [Known inconsistencies](#known-inconsistencies-kept-as-released)) | `adversarial_kappa_results.csv` | App. N |

### Experiments added for the camera-ready

The `slurm/` targets named in the command column are listed under
[Submitting single experiments](#submitting-single-experiments).

| Script (`experiments/`) | Purpose | Command (as run) | Output (`results/`) | Paper |
|---|---|---|---|---|
| `mvpc_baseline.py` | MVPC (Tu et al.) via causal-learn 0.1.3.7: PC-stable search + `MV_Crtn_Fisher_Z` correction, on the p-sweep and network datasets | `--benchmark erdos --p P --dgp D --mechanism M --n_graphs 10 --n_datasets 20`; `--benchmark alarm --dgp D --n_reps 20`; `--benchmark hailfinder --dgp nonlinear --n_reps 20 --rep_timeout 3600` (`slurm/submit_rebuttal.sh h1`); `--smoke` for a 1-minute API check | `mvpc_p{p}_{dgp}_{mech}.csv`, `mvpc_alarm_{linear,nonlinear}.csv`, `mvpc_hailfinder_nonlinear.csv` | Tables 2–3 (MVPC rows); App. F, H, I; App. T.1 (bootstrap) |
| `mvpc_sachs.py` | MVPC on the Sachs benchmarks, replaying `sachs.py` (real data) and the per-cell `sachs_nonlinear.py` streams; test-wise deletion rerun as an identity check | `--dgp real`; `--dgp nonlinear` | `mvpc_sachs_{real,nonlinear}.csv` | App. J (MVPC column) |
| `mvpc_robustness.py` | MVPC on the p = 10 rate × n sweep, replaying `robustness.py`'s stream; test-wise deletion rerun as an identity check | `--n_jobs 6` | `mvpc_robustness.csv` | App. F (MVPC rows of both sweep tables) |
| `alpha_sweep_baselines.py` | α sweeps of the tunable baselines on the main runs' datasets (HAILFINDER: matched in distribution, caveat (iv)) | `--benchmark erdos --n_graphs 10 --n_datasets 20`; `--benchmark alarm --dgp D --n_reps 20 --rates 0.2 --mechs mar mnar mixed`; HAILFINDER with `--alphas 0.01 0.05 0.10 0.20` (`submit_rebuttal.sh h2`) | `alpha_sweep_baselines_p10.csv`, `alpha_sweep_baselines_alarm_{linear,nonlinear}.csv`, `alpha_sweep_baselines_hailfinder.csv` | App. T.1 (Figure `pr_frontier`, builder `sparsity_matched`); App. T.2 |
| `kci_gcm_audit.py` | Complete-data audit of GCM (trees × residuals) and KCI (scaling, bandwidth, null) configurations | `--n_jobs 16 --output results/kci_gcm_audit.csv` (`submit_rebuttal.sh h3`) | `kci_gcm_audit.csv` | Table 1 (tuned-KCI row); App. R.1 (builder `kci_gcm_audit`) |
| `gcm_ols.py` | GCM with a correctly specified OLS learner | `--n_jobs 8 --output results/gcm_ols_audit.csv` (`slurm/submit_anticipation.sh a2`) | `gcm_ols_audit.csv` (n = 500 × \|X\| = 10 cells not run, as the paper states) | App. R.1 (builder `gcm_ols`) |
| `imputer_quality.py` | Direct imputer accuracy: R², NRMSE, corr(δ, Y), corr(δ, Z) | `--benchmark calibration --n_jobs 16`; `--benchmark erdos --n_jobs 16` (`submit_rebuttal.sh h4`). The released calibration file is the rerun on the corrected generator from `slurm/submit_diagnostics.sh imputer`, written as `imputer_quality_calibration_v2.csv` and renamed | `imputer_quality_calibration.csv`; `imputer_quality_erdos.csv` (not used by any builder or the paper) | App. S.3 (builder `imputer_quality`) |
| `learner_robustness.py` | Calibration under weak or misspecified learners (RF-100, shallow RF, stumps, ridge, k-NN) | `--output results/learner_robustness.csv` (defaults: 200 reps/cell; `submit_rebuttal.sh h7`) | `learner_robustness.csv` | App. S.2 (builder `learner_robustness`) |
| `variance_decomposition.py` | T vs Var(μ̂): 250 datasets × 4 pipeline runs per cell | `--output results/variance_decomposition.csv` (`submit_rebuttal.sh h5`) | `variance_decomposition.csv` | App. R.3 (builder `variance_ratio`) |
| `test_variable_missingness.py` | Missingness on the test variables Z / Y: imputed-labels vs labels-observed rule | `--output results/test_variable_missingness.csv` (defaults: 250 reps/cell; `submit_rebuttal.sh h6`) | `test_variable_missingness.csv` | §3.2; App. T.3 (builder `testvar_missingness`) |
| `calibration.py` (MAR × \|X\| = 2 rerun) | Replacement rows after the generator fix | `--mechanisms mar --dim_xs 2 --n_reps 100 --methods ours fisher_z gcm fisher_z_rubin --seed 4242`; KCI slice `--seed 4243` (`slurm/submit_diagnostics.sh mar2`) | `mar_dim2_rerun.csv`, `mar_dim2_rerun_kci.csv` (merged into `ci_calibration_*.csv`) | Table 1 (MAR column); App. V |
| `search_check_baselines.py` | Test-wise deletion, FZ-vote and complete case under the simplified search and under PC-stable, on Table 3's 20%-MNAR cells, with identical data and seeds | `--network {alarm_nonlinear,hailfinder}` (`slurm/submit_search_check.sh baselines`) | `search_check/rerun_alarm_nl.csv`, `search_check/rerun_hailf.csv` | App. E ("PC algorithm with cached imputations") |

### Supplementary: run or built, not reported in the paper

Diagnostics kept for provenance. `./reproduce_all.sh --supplementary` submits
the four experiments marked †; the rest run locally.

| Script | Purpose | Command (as run) | Output (`results/`) | Status |
|---|---|---|---|---|
| `experiments/placebo_diagnostics.py` † | Placebo quality at \|X\| = 10 (oracle / deployed kNN / marginal / fixed-k arms) | `--dgp D --output results/placebo_diag_{dgp}.csv` (`slurm/submit_diagnostics.sh placebo`) | `placebo_diag_{linear_gaussian,post_nonlinear,latent_confounder}.csv` | builder `placebo_diag` exists; not in the paper |
| `experiments/pooling_validity.py` † | Dependence-valid pooling (Fisher, twice-median, twice-mean, Bonferroni, Licht–Rubin) for imputed GCM/KCI | `--n_jobs 16` (`slurm/submit_anticipation.sh a1`) | `pooling_validity.csv` (`--smoke` writes `pooling_validity_smoke.csv`) | no builder; not in the paper |
| `experiments/heavy_missingness_p10.py` † | p = 10 with all non-root variables incomplete | `--dgp D --mechanism M` (`submit_anticipation.sh a3`) | `heavy_miss_p10_{dgp}_{mech}.csv` | no builder; not in the paper |
| `experiments/search_check_pairci.py` † | PAIR-CI itself under the PC-stable search (complement of the search check) | `--network N --rep R` (`slurm/submit_search_check.sh pairci`, 40 jobs) | `search_check/pairci_stdsearch_{network}_rep{NN}.csv` | **never run**; optional |
| `experiments/sachs.py --learner ridge` | Sachs, real data, with a ridge learner in place of the default tree ensemble | 20 replicates; not in `reproduce_all.sh` | `sachs_results_ridge.csv` | no builder |
| `analysis/search_oracle_check.py` | Perfect-oracle runs of the in-house search: the two Remark (i) counterexamples (App. A.6), ER DAGs, true ALARM/HAILFINDER | `--part {examples,er,networks,all}`, `--smoke` | prints only | the counterexamples are stated in App. A.6; the ER/network rates are not in the paper |
| builder `calibration_full_by_dgp` | Per-DGP mechanism × n × \|X\| grid | `python analysis/build_tables.py --table calibration_full_by_dgp` | `tables/tab_calibration_full_by_dgp.tex` | built, not `\input` |
| `experiments/k_compare.py` | One-off K = 5 vs K = 10 check on the worst calibration cell | `python experiments/k_compare.py` (no CLI; prints only) | none | provenance only; the paper's K row comes from `sensitivity.py` |

## Analysis and table-building workflow

Everything below runs locally from the released CSVs (pandas, scipy,
matplotlib; seconds to minutes), in dependency order:

```bash
# Table fragments: every tables/tab_*.tex except the bootstrap table
python analysis/build_tables.py --output-dir tables/        # --table <name> for one; --print to stdout

# Paired, graph-clustered bootstrap CIs (B = 10,000, seed 20260828)
#   -> results/paired_bootstrap_contrasts.csv, tables/tab_bootstrap_contrasts.tex
python analysis/paired_bootstrap.py --results-dir results --output-dir tables

# Precision-recall frontier (App. T.1) -> figures/pr_frontier.{pdf,png}, results/pr_frontier_points.csv
python analysis/pr_frontier.py --results-dir results --output figures/pr_frontier.pdf

# kappa trace (App. N): replay the adversarial cells -> results/kappa_trace.csv (~5 min),
# then the figure with its fitted line -> figures/kappa_fpr_trace.{pdf,png}, results/kappa_fit.csv
python analysis/compute_kappa_trace.py
python analysis/plot_kappa_trace.py

# Power plots (App. D) and skeleton precision-recall scatter (App. G)
python analysis/generate_figures.py --results-dir results/
python analysis/plot_pr_scatter.py --results-dir results/ --output figures/pr_scatter.pdf

# Worked example for Proposition 1 (App. U): closed form vs Monte Carlo
python analysis/worked_example_closed_form.py                # conditional-mean imputation
python analysis/worked_example_closed_form.py --draw_noise   # posterior-draw imputation

# Checks (non-zero exit on failure)
python analysis/search_check_summary.py                      # App. E search-check numbers
python analysis/check_rebuttal_numbers.py --results-dir results/   # 201 numbers quoted in the authors' responses during review
python analysis/search_oracle_check.py --smoke               # perfect-oracle counterexamples (App. A.6)
```

`analysis/tables_revision.py` holds the builders added for the camera-ready;
`build_tables.py` merges its registry, so it is not run on its own.
`analysis/extract_paper_numbers.py` is the submitted version's number dump. It
still runs, but `build_tables.py` and `check_rebuttal_numbers.py` supersede it.

Checked on the released CSVs (4 Nov 2026): `build_tables.py` regenerates every
builder fragment in the camera-ready identically, and `paired_bootstrap.py`
does the same for `tab_bootstrap_contrasts.tex` and
`results/paired_bootstrap_contrasts.csv`; `pr_frontier_points.csv` and
`kappa_fit.csv` also reproduce exactly. The paper's `tab_shd_combined.tex` and
`tab_degradation_combined.tex` are not generated: each stacks two builder
fragments as Panels A and B (`tab_shd_linear` over `tab_shd_nonlinear`;
`tab_degradation_fpr` over `tab_degradation_power`). `check_rebuttal_numbers.py`
passes 201/201, and `search_check_summary.py` reproduces every search-check
number in App. E.

## Changes since the submitted version

App. V of the paper summarises these.

- **MAR × |X| = 2 generator fix (July 2026).** In the standalone calibration
  grid, `src/dgps.py::inject_missingness_calibration` had no fully observed
  driver at |X| = 2 under MAR. It degenerated to one-column self-masking
  (MNAR mislabelled as MAR), and the second column was never masked. The
  corrected generator masks one column, driven by the other. Cells with
  |X| ≥ 5 are bit-identical, and the graph experiments (external drivers)
  were never affected. The affected cells were rerun (100 replicates per
  cell; `mar_dim2_rerun.csv`, `mar_dim2_rerun_kci.csv`, seeds 4242/4243) and
  merged:
  - `ci_calibration_fast.csv` and `ci_calibration_kci.csv`: mar × dim2 rows replaced; row counts unchanged;
  - `ci_calibration_results.csv`: regenerated as their concatenation;
  - `imputer_quality_calibration.csv`: rerun in full on the fixed generator; all other cells identical;
  - `variance_decomposition.csv`: the four mar × dim2 cells were not rerun but
    relabelled `mechanism = mnar_1col`, the mechanism they actually realised.

  The pre-fix originals are not shipped. Effect on Table 1's MAR column:
  FZ-Rubin 0.113 → 0.058; PAIR-CI unchanged at 0.021.
- **Reproducibility caveats.**
  (i) The corrected generator consumes the random-number stream differently
  in that stratum, so regenerating the calibration grid from one seed
  (`slurm/submit_calibration.sh`, `reproduce_all.sh`) matches the released
  files downstream of the affected configurations in distribution, not bit
  for bit. The same holds for `variance_decomposition.py`, which uses one
  sequential generator; a rerun also yields genuine-MAR cells, labelled
  `mar`, where the released file has the four `mnar_1col` cells.
  `imputer_quality.py` seeds per cell and reproduces exactly.
  (ii) MVPC's correction step draws from NumPy's unseeded global generator in
  its authors' implementation (`np.random.shuffle` in
  `causallearn/utils/PCUtils/Helper.py::get_predictor_ws`, causal-learn 0.1.3.7),
  so MVPC's outputs reproduce in distribution but not bit for bit, even where
  its input datasets are identical to the other methods' (see (iv)).
  (iii) FZ-vote's `IterativeImputer` (posterior draws) is not bit-identical
  across platforms and BLAS builds, even with identical seeds. Rerunning
  Table 3's 20%-MNAR cells for the search check gave the released per-replicate
  SHD exactly for test-wise deletion and complete case, but for FZ-vote it
  matched in only 6/20 (ALARM-nonlinear) and 1/20 (HAILFINDER) replicates
  (|ΔSHD| ≤ 5 and ≤ 9; medians 55.5 → 56.0 and 122.0 → 121.5).
  (iv) The released HAILFINDER and Sachs-nonlinear runs of PAIR-CI and the
  in-house baselines were one job per rate × mechanism cell. A per-cell job skips the other cells' masking draws, so it
  reproduces a single-job run's datasets only in the 10% complete and 10% MAR
  cells; the other cells get different, identically distributed datasets.
  `new_baselines.py`, `mvpc_baseline.py` and `alpha_sweep_baselines.py` replay
  the single-job stream. Their datasets therefore match PAIR-CI's in
  distribution only, not replicate by replicate, for FZ-Rubin on HAILFINDER and
  Sachs-nonlinear, MissDAG on Sachs-nonlinear, MVPC on HAILFINDER, and the
  HAILFINDER α sweeps. Everywhere else (the ER grids, ALARM, real Sachs) the
  baselines see the same datasets as PAIR-CI, as does the search check, which
  replays the per-cell streams.
  After a from-scratch regeneration, `check_rebuttal_numbers.py` and
  `search_check_summary.py` can fail at their rounding tolerances through
  Monte Carlo noise alone. On the released CSVs both pass.
- **New experiments**: MVPC, the baseline α sweeps, the GCM/KCI audit, GCM-OLS,
  imputer quality, learner robustness, variance decomposition, test-variable
  missingness and the search check, listed in the tables above and run by
  `reproduce_all.sh`.
- **Simplified PC search and the search check.** All in-house methods (PAIR-CI
  and the deletion and imputation baselines) share `src/pc_integration.py::_pc_skeleton`.
  It departs from the standard PC adjacency search (Kalisch & Bühlmann 2007,
  Alg. 1) by drawing conditioning sets from one endpoint's (the lower index's)
  neighbourhood only and by stopping at the first level at which no edge is
  removed (App. E, Algorithm `alg:pc`). MVPC instead runs causal-learn's
  PC-stable search followed by its correction step. `experiments/search_check_baselines.py`
  reruns test-wise deletion and FZ-vote under the PC-stable search on the Table 3
  data. Their median SHD falls by 7 and 7.5 on ALARM-nonlinear and by 24 and
  18.5 on HAILFINDER, and spurious adjacencies fall in every replicate. PAIR-CI,
  still on the simplified search, keeps a lower SHD in 20/20 paired replicates
  against both re-searched baselines (`python analysis/search_check_summary.py`).
  Against MVPC, its median SHD and F1 are better in both cells; per replicate it
  wins 20/20 on ALARM-nonlinear and 14/19 on HAILFINDER (datasets matched in
  distribution there; one MVPC replicate exceeded the 1-hour cap).
  PAIR-CI itself under PC-stable (`experiments/search_check_pairci.py`) was not run.
- **Paired bootstrap** intervals were recomputed (`analysis/paired_bootstrap.py`,
  seed 20260828) and differ from those quoted during the review discussion by
  at most 0.02 at either endpoint.
- **SLURM scripts.** The submitted `slurm/*.sh` passed a non-existent `default`
  partition and activated conda inside `--wrap` payloads, which run under
  `/bin/sh`, where activation fails silently. Some were `#SBATCH` scripts that
  located the package via `$0`, which under `sbatch` points at slurmd's spooled
  copy. All are now `bash` submit wrappers: they locate the env's python at
  submit time, pin BLAS threads, and use the cluster's default partition unless
  `PARTITION=<name>` is set. Job parameters are unchanged. `reproduce_all.sh`
  has the same fix (`--partition` is optional).

## Compute and runtime

Runtimes below are the per-row `runtime` columns the scripts log (wall-clock
time inside the method call), summed per job. Linear-edge graph runs are much
faster than nonlinear ones because PAIR-CI uses the ridge learner there. The graph and network
experiments run their replicates one after another, so for them the sum is
close to the job's wall time; for jobs that run tests in parallel processes
(`--n_jobs`) it is not. "Not logged" means the script records no runtime.
Jobs have 16 CPUs unless stated; walltime is the limit `reproduce_all.sh`
requests.

| Experiment | Jobs | Logged runtime per job | Total | Walltime |
|---|---|---|---|---|
| `calibration.py`: fast methods, KCI, merge | 3 | not logged | | 5 d, 7 d, 12 h |
| `caching_benchmark.py` + `caching_validation.py` | 1 | 1.9 h for the benchmark; validation not logged | | 5 d |
| `graph_recovery_p10.py`, linear / nonlinear | 4 + 4 | 1.3–1.5 h / 44–48 h | 6 h / 184 h | 5 d |
| `graph_recovery_p20.py`, linear / nonlinear | 4 + 4 | 2.7–3.0 h / 42–46 h | 11 h / 176 h | 10 d |
| `graph_recovery_p30.py`, linear / nonlinear | 4 + 4 | 6.6–7.2 h / 96–120 h | 27 h / 438 h | 10 d |
| `robustness.py` | 18 | 36–49 h | 756 h | 5 d |
| `sachs.py` | 1 | 33 h | | 5 d |
| `sachs_nonlinear.py`, per cell + merge | 10 + 1 | 1.5–1.9 h | 17 h | 2 d |
| `sensitivity.py` | 1 | 113 h | | 10 d |
| `alpha_sensitivity.py` | 1 | 108 h | | 7 d |
| `alarm.py`, linear / nonlinear | 1 + 1 | 5.3 h / 119 h | | 10 d |
| `hailfinder.py`, per cell + merge | 10 + 1 | 34–51 h | 390 h | 2 d (see [Known inconsistencies](#known-inconsistencies-kept-as-released)) |
| `early_stopping_calibration.py` | 1 | not logged | | 12 h |
| `imputation_degradation.py` | 1 | not logged | | 2 d |
| `variance_comparison.py` | 1 | 65 h for the graph part; calibration part not logged | | 5 d |
| `new_baselines.py`, ER sweep | 24 | 0.2 h (p=10), 1.2–1.9 h (p=20), 3.2–5.8 h (p=30) | 46 h | 5 d |
| `new_baselines.py`, ALARM / HAILFINDER / Sachs | 2 + 1 + 2 | 3.1, 5.8 h / 2.1 h / 1.6, 0.3 h | 13 h | 2 d (HAILFINDER 5 d) |
| `m_sensitivity.py` | 1 | not logged | | 2 d |
| `learner_comparison.py` | 1 | 52 h | | 2 d (see [Known inconsistencies](#known-inconsistencies-kept-as-released)) |
| `adversarial_kappa.py` | 1 | not logged | | 2 d |
| *Added for the camera-ready:* | | | | |
| `mvpc_baseline.py`, ER p=10/20/30 | 24 (2 CPUs) | median 0.03 / 0.41 / 1.08 s per replicate | 0.7 h | 2 d |
| `mvpc_baseline.py`, ALARM ×2 / HAILFINDER | 2 + 1 (2 CPUs) | median 0.83 s / 3.0 s per replicate | | 2 d |
| `alpha_sweep_baselines.py`, p=10 / ALARM ×2 / HAILFINDER | 1 + 2 + 1 (8 CPUs) | 0.6 h / 1.2, 1.4 h / 6.5 h | 9.7 h | 2 d |
| `kci_gcm_audit.py` | 1 (16 processes) | 297 h summed over GCM tests (median 1.9 s/test), 4 h over KCI tests | | 2 d |
| `gcm_ols.py` | 1 (8 CPUs) | not logged | | 12 h |
| `imputer_quality.py` | 2 | not logged | | 12 h |
| `variance_decomposition.py`, `test_variable_missingness.py`, `learner_robustness.py` | 1 each | not logged | | 2 d |
| `search_check_baselines.py`, ALARM-nonlinear / HAILFINDER | 1 + 1 (2 CPUs) | 378 s / 3,901 s | | 12 h / 2 d |
| supplementary: `placebo_diagnostics.py` | 3 | 4.4–5.1 h | 14.0 h | 2 d |
| supplementary: `heavy_missingness_p10.py` | 4 (4 CPUs) | 0.2 h (linear), 4.0–4.1 h (nonlinear) | 8.5 h | 2 d |
| supplementary: `pooling_validity.py` | 1 | not logged | | 2 d |
| supplementary: `search_check_pairci.py` | 40 (8 CPUs) | never run | | 12 h |

MVPC replicates are capped at 30 min (the script's default `--rep_timeout`)
on the ER sweep and ALARM and at 1 h on HAILFINDER. Two HAILFINDER replicates
(20% and 40% MNAR) did not finish within the cap, and one ALARM-nonlinear
replicate (10%, mixed) failed.

Summed over jobs, the submitted version's experiments logged about 2,560 h,
not counting the experiments that log no runtime; the camera-ready additions
are small by comparison, apart from the GCM/KCI audit. With full parallelism
a complete run takes about five days, set by the longest single jobs:
`alarm.py --dgp nonlinear` (119 h; 200 replicates at a median of about 30 min
each), the slowest `graph_recovery_p30.py` nonlinear cells (up to 120 h),
`sensitivity.py` (113 h) and `alpha_sensitivity.py` (108 h). HAILFINDER runs
as ten per-cell jobs, the slowest taking 51 h.

## Running on the cluster

### Requirements

- **Scheduler**: SLURM. Jobs go to the cluster's default partition; choose
  another with `--partition <name>` (`reproduce_all.sh`) or `PARTITION=<name>`
  (`slurm/*.sh`).
- **Conda environment**: `causal_ci`, created by `reproduce_all.sh` from
  `environment.yml` if it does not exist. Jobs get the env's python through
  `PATH`, located at submit time; set `PYBIN=/path/to/python` to override
  (not honoured by three of the `slurm/` scripts, see
  [Known inconsistencies](#known-inconsistencies-kept-as-released)).
  `analysis/search_oracle_check.py` needs networkx ≥ 3.3, which causal-learn
  pulls in unpinned.
- **CPUs**: 16 per job for most jobs; 2 for MVPC and the search check, 8 for
  the α sweeps, GCM-OLS and the PAIR-CI search check, 4 for the
  heavy-missingness runs. No GPU.
- **Memory**: 4 GB for most jobs; 8 GB for p = 30, HAILFINDER, the
  `new_baselines.py` network runs on ALARM and HAILFINDER, the network
  MVPC and α-sweep runs, the GCM/KCI audit, the HAILFINDER search check and
  the supplementary pooling and PAIR-CI search-check runs.
- **Walltime**: `reproduce_all.sh` uses five classes, `TIME_SHORT` (12 h),
  `TIME_MEDIUM` (2 d), `TIME_LONG` (5 d), `TIME_XLONG` (7 d) and
  `TIME_XXLONG` (10 d); the compute table gives each experiment's. Under
  `--quick` all are 30 min to 1 h.

### Submitting single experiments

The `slurm/` scripts are submit wrappers: run them with `bash`, not `sbatch`.

```bash
bash slurm/submit_sachs.sh                     # just the Sachs experiment
PARTITION=compute bash slurm/submit_p20.sh     # p=20 jobs on a named partition
DRY_RUN=1 bash slurm/submit_p20.sh             # print the sbatch calls without submitting

# Camera-ready experiments, by target
bash slurm/submit_rebuttal.sh h1 h2    # h1 MVPC, h2 baseline α sweeps, h3 GCM/KCI audit, h4 imputer quality,
                                       # h5 variance decomposition, h6 test-variable missingness, h7 learner
                                       # robustness; with no target, h1-h6
bash slurm/submit_anticipation.sh a2   # a2 GCM-OLS; a1 (pooling) and a3 (heavy missingness) are supplementary
bash slurm/submit_diagnostics.sh mar2  # MAR x |X|=2 replacement rows; also `imputer`, `placebo`
bash slurm/submit_search_check.sh      # search check, baselines; add `pairci` for the optional PAIR-CI arm
```

### Logs and failures

Experiment commands discard stderr (`2>/dev/null`). scikit-learn's deprecation
and convergence warnings cannot all be silenced through Python's `warnings`
module, and on long runs they reached ~100 GB of logs. Most experiment scripts
catch errors per method and replicate, print them to the job's stdout log
(`logs/slurm_<job>_<id>.out`) and record the row as failed. A crash outside
those loops loses its traceback with stderr; rerun the job's command by hand,
without `2>/dev/null`, to see it.

There is no resume flag. Jobs are independent, so resubmit failed ones with
the matching `slurm/` script, or copy their `sbatch` line from
`./reproduce_all.sh --dry-run`. The combine and build jobs wait on the
original job IDs, so after a resubmission run the `analysis/combine_*.py`
scripts and the [analysis workflow](#analysis-and-table-building-workflow) by
hand.

## Smoke test

To check that the pipeline runs end to end, use a copy of the package with an
empty `results/`. `--quick` writes to the same `results/*.csv` paths and
refuses to overwrite the released results unless `--force` is given:

```bash
rsync -a --exclude results/ --exclude logs/ ./ ../pairci_smoke/
cd ../pairci_smoke && mkdir results
./reproduce_all.sh --quick    # 1 graph, 2 datasets, reduced reps; each script's own --quick/--smoke mode
```

Every script and analysis step runs, with too few replicates to match the
paper. The search check has no reduced mode and is skipped under `--quick`, so
`search_check_summary.py` fails for want of its CSVs, as does
`check_rebuttal_numbers.py` on smoke output; the build job prints a notice and
continues.

## DGPs and shared datasets

Shared DGP and missingness functions live in `src/dgps.py`. The exceptions:
`caching_benchmark.py` uses a small fixed p = 5 graph of its own;
`adversarial_kappa.py` defines its six adversarial DGPs and its own MNAR
injector; `alarm.py` and `hailfinder.py` define the true DAGs and their own
data samplers; `placebo_diagnostics.py` carries copies of three `dgps.py`
generators that also return the conditional mean (kept in sync by hand); and
`test_variable_missingness.py` adds its own masking of Z and Y to the
`dgps.py` generators. The baselines and the search check replay the
experiment scripts' random-number streams; caveat (iv) under
[Changes since the submitted version](#changes-since-the-submitted-version)
lists where that gives the same datasets as PAIR-CI and where only matched
ones.

## Directory structure

```
pair-ci_replication/
├── README.md, LICENSE, NOTICE, LICENSES/   # licences: see above
├── reproduce_all.sh            # submits every job (see Quick start)
├── environment.yml             # conda environment (causal_ci)
├── check_env.py                # checks that the environment imports cleanly
│
├── src/                        # core library
│   ├── ci_oracle.py            # GeneralCITest class (PAIR-CI, general variant)
│   ├── ci_oracle_fast.py       # FastCITest class (ExtraTrees, early stopping)
│   ├── ci_oracle_linear.py     # ridge-learner variant (linear-edge graphs)
│   ├── pc_integration.py       # simplified PC search and in-house baselines
│   ├── conditional_permutation.py  # kNN local permutation
│   ├── baselines_new.py        # FZ-Rubin (micd-pooled) and MissDAG-Gaussian baselines
│   ├── notears_linear.py       # NOTEARS solver used in MissDAG's M-step
│   └── dgps.py                 # shared DGP and missingness functions
│                               # (missdag_gaussian.py is not shipped; see Licences)
│
├── experiments/                # one script per experiment (tables above)
│   ├── calibration.py, caching_benchmark.py, caching_validation.py
│   ├── graph_recovery_p10.py, graph_recovery_p20.py, graph_recovery_p30.py
│   ├── alarm.py, hailfinder.py, sachs.py, sachs_nonlinear.py
│   ├── sensitivity.py, robustness.py, alpha_sensitivity.py
│   ├── variance_comparison.py, early_stopping_calibration.py, imputation_degradation.py
│   ├── new_baselines.py, m_sensitivity.py, learner_comparison.py, adversarial_kappa.py
│   ├── mvpc_baseline.py, alpha_sweep_baselines.py           # camera-ready
│   ├── kci_gcm_audit.py, gcm_ols.py, imputer_quality.py     # camera-ready
│   ├── learner_robustness.py, variance_decomposition.py     # camera-ready
│   ├── test_variable_missingness.py, search_check_baselines.py  # camera-ready
│   ├── placebo_diagnostics.py, pooling_validity.py          # supplementary
│   ├── heavy_missingness_p10.py, search_check_pairci.py     # supplementary
│   └── k_compare.py            # one-off K=5 vs K=10 check (not in the pipeline)
│
├── analysis/                   # post-processing, figures, tables, checks
│   ├── combine_p10.py, combine_p20.py, combine_p30.py, combine_robustness.py
│   ├── combine_hailfinder.py, combine_sachs_nonlinear.py    # merge per-cell CSVs
│   ├── build_tables.py         # every LaTeX table fragment (with the tables_revision.py registry)
│   ├── tables_revision.py      # builders added for the camera-ready
│   ├── paired_bootstrap.py     # paired cluster-bootstrap contrasts
│   ├── pr_frontier.py          # PR frontier figure and points
│   ├── compute_kappa_trace.py, plot_kappa_trace.py  # kappa trace and Figure kappa_fpr_trace
│   ├── generate_figures.py     # power plots
│   ├── plot_pr_scatter.py      # precision-recall scatter
│   ├── worked_example_closed_form.py  # App. U closed form vs Monte Carlo
│   ├── check_rebuttal_numbers.py      # asserts the 201 numbers quoted during review
│   ├── search_check_summary.py        # asserts the App. E search-check numbers
│   ├── search_oracle_check.py         # perfect-oracle checks of the search
│   ├── extract_paper_numbers.py       # submitted version's number dump (legacy)
│   └── plot_kill_switch.py     # plotting helpers used by combine_p10.py
│
├── slurm/                      # per-experiment submit wrappers (run with bash)
│   ├── _common.sh              # shared submit-time setup (sourced)
│   ├── submit_*.sh             # one per experiment family
│   └── submit_rebuttal.sh, submit_anticipation.sh, submit_diagnostics.sh, submit_search_check.sh
│
├── data/
│   ├── fetch_sachs_data.sh     # downloads and checks the Sachs data (not redistributed)
│   └── sachs_data.csv          # Sachs protein signalling (853 obs x 11 vars; written by the fetch script)
│
├── release/                    # MANIFEST.txt and make_release.sh
├── results/                    # output CSVs (released; regenerated by the scripts)
│   └── search_check/           # search-check outputs
├── tables/                     # LaTeX fragments (TABLE_NOTES.md: definitions, caveats)
├── figures/                    # output figures (flat layout)
└── logs/                       # SLURM job logs (created on first submission)
```

## Known inconsistencies kept as released

- The fourth orientation rule in `src/pc_integration.py::_orient_edges`,
  labelled R4, is not Meek's R4: it checks the opposite adjacency conditions.
  With a perfect CI oracle it changes no orientation on any DAG used in the
  paper (the index-ordered ER graphs, ALARM, HAILFINDER), but it can on other
  graphs (about 0.5% of randomly relabelled ER DAGs at p = 10 and 3% at
  p = 20). It affects total SHD only, not skeleton metrics, and not MVPC or
  MissDAG. It is kept so that the released results reproduce.
- `slurm/submit_adversarial_kappa.sh` requests 200 replicates; the released
  `adversarial_kappa_results.csv` has 100 per cell, as run by `reproduce_all.sh`.
- `slurm/submit_hailfinder.sh` and `slurm/submit_sachs_nonlinear.sh` run the
  network as one job. `reproduce_all.sh`, like the released run, splits it
  into 10 per-cell jobs plus a merge; the two match only in the 10% complete
  and 10% MAR cells (caveat (iv) under
  [Changes since the submitted version](#changes-since-the-submitted-version)).
- Two experiments logged more runtime than `reproduce_all.sh` allows them:
  the two slowest HAILFINDER cells (49 h and 51 h against `TIME_MEDIUM`,
  2 days) and `learner_comparison.py` (52 h against 2 days;
  `slurm/submit_learner_comparison.sh` asks for 1 day). On nodes no faster
  than ours, raise `TIME_MEDIUM` or resubmit these jobs with a longer `--time`.
- `slurm/submit_rebuttal.sh`, `submit_anticipation.sh` and
  `submit_diagnostics.sh` do not source `slurm/_common.sh`. They honour
  `PARTITION` but ignore `DRY_RUN`, `PYBIN` and `CONDA_ENV`, so they always
  submit, with the `causal_ci` env. `./reproduce_all.sh --dry-run` shows the
  same commands.
