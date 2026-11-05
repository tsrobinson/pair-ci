# Table fragments: contents and caption notes

Each `tables/tab_<name>.tex` is written by `python analysis/build_tables.py`
(one fragment with `--table <name>`), except `tab_bootstrap_contrasts.tex`,
which `analysis/paired_bootstrap.py` writes. Each entry gives where the paper
uses the fragment, what it reports, and the caveats its caption should carry.
Unless stated otherwise, rates are rejection rates at α = 0.05 and red marks
values above α. Fragments marked *not \input* are built but not used by the
camera-ready. `analysis/check_rebuttal_numbers.py` checks 201 quoted numbers
against the same loaders.

## Table 1: standalone calibration

**calibration** (Panel A). FPR at signal 0, mean over the three DGPs,
n ∈ {500, 1000, 2000, 5000} and |X| ∈ {2, 5, 10} (n = 500 × |X| = 10 is not in
the grid), 30% missingness. 100 replicates per cell for PAIR-CI, Fisher-Z and
GCM; 50 for default KCI, which was run at n = 1,000 only. The Complete column
shows complete-case variants only: imputed variants are defined only under
missingness, since on complete data the M = 5 imputations coincide and Fisher
pooling of identical p-values would inflate KCI from 0.13–0.17 to 0.278.
"KCI (tuned)†" (standardised inputs, median-heuristic bandwidth, spectral null)
comes from the complete-data audit grid (`kci_gcm_audit.csv`: n ∈ {500, 1000},
200 replicates per cell), where default KCI gives 0.144; disaggregated in
`kci_gcm_audit`. The MAR column uses the CSVs merged after the MAR × |X| = 2
generator fix, so the caption should point to App. V.

**calibration_power_compact** (Panel B). PAIR-CI FPR (signal 0) by mechanism
× n and power (signal 0.3) under MNAR, pooled over DGP and |X|. Caveats:
(i) the n = 500 row pools over |X| ∈ {2, 5} only; (ii) the learner subsamples
to 2,000 rows (`max_subsample`), which is why power at n = 5,000 does not
exceed power at n = 2,000. FPR stays below α at this grain but not at
mechanism × n × |X|, where some |X| = 10 cells exceed it (`calibration_full`).

**calibration_power** (not \input). As Panel B, with power for all three
mechanisms.

## Table 2: synthetic graph recovery

**scaling_combined** (Panel A). Per method, p ∈ {10, 20, 30} and edge type:
total SHD, skeleton SHD in parentheses, and skeleton F1. Each value is the
median across mechanisms (complete, MAR, MNAR, mixed) of per-mechanism medians
over 10 graphs × 20 datasets, n = 1,000. Bold = lowest constraint-based SHD
(ties within 0.5 both bold). MVPC‡ runs in its authors' PC implementation
(causal-learn, `MV_Crtn_Fisher_Z`, the only correction released); all other
constraint-based rows share our PC search. MissDAG† outputs a DAG, so its total
SHD is not comparable (gray); its skeleton SHD is.

**scaling_pre** (Panel B). Skeleton precision / recall, median output edges in
parentheses, with Panel A's aggregation.

**scaling**, **scaling_f1** (not \input). Panel A's SHD and F1 as separate
tables.

## Table 3: network topologies

**realworld**. ALARM (linear and nonlinear edges) and HAILFINDER at 20%
missingness per incomplete column, MNAR, 20 replicates: median SHD, skeleton
precision, recall, F1 and median output edges. Edge counts for the PAIR-CI and
original-baseline runs are back-derived from precision, recall and the true
edge count (46 / 66), because those runs logged no tp/fp; MVPC and FZ-Rubin
log tp/fp directly. On HAILFINDER, FZ-Rubin and MVPC see datasets matched to
PAIR-CI's in distribution, not the same datasets (README, reproducibility
caveat (iv)). MVPC did not finish within the 1-hour cap on one
HAILFINDER replicate at MNAR-20% (median over 19; one more at MNAR-40%, in the
appendix tables). Complete-case HAILFINDER MNAR is the empty graph (F1 = 0,
gray †). Sixteen columns: the paper wraps it in `\resizebox`.

**realworld_by_mech** (not \input). The same metrics for MAR, MNAR and mixed.

Ranges at the right grain: on HAILFINDER at 20%, the parametric pipelines
output 89–112 edges, 57–64% of them false (range over MAR, MNAR and mixed). On
ALARM-nonlinear, PAIR-CI's F1 is 0.79–0.80 and recall 0.67–0.70 across
mechanisms, against F1 0.49–0.65 for the four original baselines and up to
0.69 for MVPC (MAR, mixed).

## Appendix: calibration and assumptions

**calibration_by_dgp** (App. A). FPR at signal 0 by DGP × mechanism (MAR,
MNAR) for every method, averaged over n and |X|, 100 replicates per cell.

**calibration_full** (App. R.2). PAIR-CI FPR by mechanism × n × |X|, pooled
over the three DGPs (300 tests per cell; MC SE ≈ 0.013 at 0.05). Maximum 0.077
(MNAR, n = 2,000, |X| = 10).

**calibration_by_n**, **calibration_by_dimx**, **calibration_full_by_dgp** (not
\input). PAIR-CI FPR by mechanism × n, by mechanism × |X| (maximum 0.067), and
by DGP × mechanism × n × |X| (100 tests per cell; MC SE ≈ 0.02 at 0.05). In
the last, linear-Gaussian complete-data |X| = 10 reaches 0.16 at n = 5,000,
the regime that `placebo_diag` traces to the placebo generator; any summary
should point there rather than describe these cells as below nominal.

**power** (App. L). PAIR-CI power by DGP × signal (0.3, 0.6, 1.0) ×
n ∈ {500, 1000, 2000}, averaged over MAR and MNAR at 30% missingness,
|X| = 5, 100 replicates per cell.

**kci_gcm_audit** (App. R.1). Complete data. GCM: trees ∈ {50, 100, 400} ×
{5-fold cross-fitted, honest OOB} residuals, 400 replicates per cell over
n ∈ {500, …, 5000} × |X| ∈ {2, 5, 10}. KCI: default (raw inputs, empirical
width, gamma null) → standardised → + median width → + spectral null, 200
replicates per cell at n ∈ {500, 1000} only (O(n³)), hence "--" at
n ≥ 2,000. The power column pools signal 0.3. RCIT is not in the pinned
causal-learn 0.1.3.7; the caption should say why it is excluded (same kernel
family, so the KCI audit bounds it).

**gcm_ols** (App. R.1). GCM with OLS regressions on complete data, 300
replicates per cell; the n = 500 × |X| = 10 cells were not run. The last
column gives pooled FPR and power per DGP.

**variance_ratio** (App. R.3). 24 cells (2 DGPs × {complete, MAR, MNAR, MNAR
(1 col.)} × n ∈ {500, 1000} × |X| ∈ {2, 5}), 250 datasets per cell and 4
pipeline runs per dataset. Raw ratio = mean T over all runs / variance of μ̂
over the 250 independent datasets (run 0); robust ratio = median T over all
runs / MAD² of μ̂ over all runs (MAD scaled by 1.4826). Raw > 1 in 21/24 cells,
maximum 5.63; robust 2.96–6.11, all > 1. The three raw ratios below 1 are the
heavy-tailed cells; the one cell above nominal (linear-Gaussian MNAR,
n = 1,000, |X| = 5, FPR 0.072) has raw ratio 0.75. FPR under N(0,1) and under
t_ν differs by at most 0.012, so the critical value is not the source of the
conservatism. The four MNAR (1 col.) cells are the pre-fix MAR × |X| = 2
generator's actual output (one-column self-masking), relabelled, not rerun.

**learner_robustness** (App. S.2). 36,000 tests: 5 learners (RF-100, shallow
RF, stumps, ridge, k-NN) × 2 DGPs (linear-Gaussian, post-nonlinear) × 3
mechanisms × n ∈ {1000, 2000} × |X| = 5 × signals {0, 0.3, 0.6} × 200
replicates. Ridge is misspecified on the post-nonlinear DGP by design; k-NN
with fixed k sits outside the stability assumption (A4). The stump FPR of
0.022 is pooled over mechanisms; its power of 0.07–0.12 is the per-DGP range.

**imputer_quality** (App. S.3). Calibration grid; rows are DGP × mechanism ×
rate, columns n; pooled over |X| and signal. R² compares the across-M mean
imputation on the missing entries with the missing subpopulation's own
variance, so negative values mean worse than that subpopulation's mean, as
expected under self-masking MNAR. NRMSE is per posterior draw against the
column's marginal SD (`nrmse_marginal`; medians 1.16 MAR, 1.42 MNAR), above 1
by design because proper imputation adds posterior noise. The correlations are
per-draw medians. Mechanism-level R² medians (MAR 0.19; MNAR −2.5; per DGP
−1.5 to −5.6) are row-level medians of the mean-imputation rows (`m == -1`).
corr(δ, Z) = −0.44 in latent-confounder MNAR is the per-draw value; the mean
imputation gives −0.52.

**testvar_missingness** (App. T.3). 30% missingness on Z, on Y, on both, or
on X and Z; n = 1,000, |X| = 5, 250 replicates × 2 DGPs × 2 mechanisms ×
2 rules. Under the deployed imputed-labels rule, FPR is 0.000–0.068 over the
16 null cells (worst: linear-Gaussian MNAR on X and Z), and power exceeds the
labels-observed rule's (0.35 against 0.28 at signal 0.3).

**degradation_fpr**, **degradation_power** (App. Q). MICE, mean and marginal
imputation inside PAIR-CI: rejection rate at signal 0 and at signal 0.6, by
mechanism, averaged over the linear-Gaussian and post-nonlinear DGPs; 200
replicates, n = 500, |X| = 5. Power cells for mean and marginal imputation
under MAR and MNAR are gray †: with inflated FPR they cannot be read as power.
The paper's `tab_degradation_combined.tex` stacks the two fragments as Panels
A and B.

**placebo_diag** (not \input). Complete data, |X| = 10, n ∈ {1000, 2000,
5000}, 150 paired replicates per cell, with datasets and internal seeds held
fixed across arms. The oracle arm (exact draw from P(Y | X)) has FPR 0.000 and
mean t ≈ 0 in all nine DGP × n cells; the deployed kNN generator behaves like
unconditional permutation at |X| = 10, and no fixed k ∈ {8, 32, 128} helps.
Latent-confounder is calibrated in every arm (its X-dependencies do not
overlap). The learner's effective n caps at 2,000.

## Appendix: graph recovery

**shd_linear**, **shd_nonlinear** (App. F). p = 10, n = 1,000: median total
SHD (skeleton SHD) by method × mechanism; half-unit medians are printed as
such. Complete means no values masked. The
paper's `tab_shd_combined.tex` stacks the two fragments as Panels A and B.

**graph_full_{linear,nonlinear}_p{10,20,30}** (App. F). Full skeleton metrics
per mechanism (SHD, skeleton SHD, precision, recall, F1, output edges),
medians over 10 graphs × 20 datasets.

**robustness_mar**, **robustness_mnar** (App. F). Median SHD at p = 10,
nonlinear edges, MAR or MNAR, by n ∈ {500, 1000, 2000} × missingness rate
{10, 30, 50}%. The MVPC row comes from `mvpc_robustness.csv` (same datasets).

**alarm_full**, **alarm_full_nonlinear** (App. H), **hailfinder_full**,
**hailfinder_f1** (App. I). The submitted rate × mechanism layouts (median
total SHD with skeleton SHD in parentheses; F1 for `hailfinder_f1`) with an
MVPC column. The 0% (Complete) row is stored as rate 0.1, mechanism `complete`
in the CSVs; no values are masked. On HAILFINDER, complete-case cells with
median F1 = 0 (empty skeleton) are gray.

**alarm_edges**, **alarm_nonlinear_edges**, **hailfinder_edges** (App. H, I).
Median output edges on the same rate × mechanism grid (46 / 66 true edges).

**network_metrics_{alarm_linear,alarm_nonlinear,hailfinder}** (App. H, I).
Full skeleton metrics per mechanism at 20% missingness, with the number of
replicates in each median and, for MVPC, the replicates that did not finish
within the cap (DNF).

**sachs_linear**, **sachs_nonlinear** (App. J). Rate × mechanism layouts,
median total SHD (skeleton SHD), 20 replicates, 11 nodes, 17 edges.
`sachs_linear` is built from the real flow-cytometry data (`sachs_results.csv`,
`newbase_sachs_linear.csv`), not simulated linear data; `sachs_nonlinear` uses
the Sachs DAG with synthetic nonlinear edges, where the FZ-Rubin and MissDAG
columns come from datasets matched to PAIR-CI's in distribution only. The MVPC
columns (`mvpc_sachs_{real,nonlinear}.csv`) use identical datasets.

## Appendix: operating points and design choices

**sparsity_matched** (App. T.1). For each tunable baseline, the swept α whose
median edge count is closest to PAIR-CI's at α = 0.05; the last column gives
the best F1 over the sweep and the α attaining it. MVPC sits at its native
operating point. Sweep grids: ALARM and p = 10 α ∈ {0.001, 0.01, 0.05, 0.1,
0.15, 0.2, 0.3}; HAILFINDER α ∈ {0.01, 0.05, 0.1, 0.2}. Test-wise deletion
bottoms out at 71 edges on HAILFINDER; complete case collapses to the empty
graph at every α.

**bootstrap_contrasts** (App. T.1). Paired difference PAIR-CI − baseline (SHD:
negative favours PAIR-CI; F1: positive favours PAIR-CI), mean over datasets,
MAR, MNAR and mixed pooled. Percentile 95% intervals from B = 10,000 resamples
of clusters (graphs on the ER grids, 10 per cell; replicates on the networks,
20), seed 20260828. Gray = interval covers 0. ∗ = MVPC on HAILFINDER, paired by
replicate index only (datasets matched in distribution, not identical). The
FZ-Rubin contrast on HAILFINDER is paired the same way, for the same reason,
but `paired_bootstrap.py` does not mark it.
Main results: at p = 10 the gains are small but significant (−0.74 [−1.04,
−0.42] linear, −0.75 [−1.24, −0.27] nonlinear, against test-wise deletion);
MVPC beats PAIR-CI on linear SHD at p ≥ 20 (+2.4, +5.2) and on ALARM-linear
(+4.5); on the ER nonlinear grids PAIR-CI wins on SHD by 4.7–16 but its
skeleton F1 is at or slightly below test-wise deletion's and FZ's (−0.01 to
−0.04) and below MVPC's (−0.02 to −0.05), reflecting its sparse operating
point; on the networks PAIR-CI wins on both SHD and F1 against every baseline
except MVPC on ALARM-linear.

**sensitivity_summary** (App. T.2). One row per design choice, the rest held at
the defaults. M from `m_sensitivity` (n ∈ {500, 1000}, |X| = 5, K = 5); K and
the joint grid from `sensitivity_results` (p = 10, nonlinear, MAR; 11 PAIR-CI
configurations, median SHD 10–11); learner from `learner_comparison`
(n ∈ {500, 1000}); α from PAIR-CI's own p = 10 sweep. The `k_compare.py`
diagnostic (K = 5 vs 10 on the worst calibration cell) prints only; rerun it
to a CSV before quoting its number.

## Figures

**kappa_fpr_trace** (App. N). x-axis: the three-factor product
κ̂²_imp |κ̂_Y| |κ̂_Z| (Definition 7 convention, κ_imp squared); the submitted
version plotted κ_Y κ_Z. Spearman 0.89 over 48 cells; median product 0.002 in
calibrated cells against 0.18 in inflated ones; log-log slope of excess FPR on
the 25 inflated cells 0.69 ± 0.15 (OLS SE). Statistics in
`results/kappa_fit.csv`.

**pr_frontier** (App. T.1). Panels: p = 10 nonlinear MNAR (with PAIR-CI's own
α sweep), ALARM-nonlinear MNAR 20%, HAILFINDER MNAR 20%. Iso-F1 contours; edge
counts annotated at the sparse end of each baseline sweep. Points in
`results/pr_frontier_points.csv`.
