#
# from __future__ import annotations
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import kruskal, mannwhitneyu
#
#
# PEARSON_SUMMARY_PKL = Path(
#     "./prior_localization_sessionfit_output/pearson_summary_VISa_VISp_PL.pkl"
# )
#
# GROUP_ORDER = ["fast", "normal", "slow"]
#
#
# METRICS = [
#     ("r_real", "Pearson r (real)", "VISa_VISp_PL_pearson_r_real_sig_n.pdf"),
#     ("r_corr", "Corrected Pearson r = r_real − mean(r_pseudo)", "VISa_VISp_PL_r_corr_sig_n.pdf"),
#     ("z_corr", r"z = (r_real − mean(r_pseudo)) / std(r_pseudo)", "VISa_VISp_PL_z_corr_sig_n_eq.pdf"),
#     ("r2_corr", "Corrected R²", "VISa_VISp_PL_r2_corr_sig_n.pdf"),
# ]
#
# OUT_DIR = Path("./prior_localization_sessionfit_output/plots_all_rois")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# SUMMARY_CSV = OUT_DIR / "summary_VISa_VISp_PL_selected_metrics.csv"
#
#
# MIN_N_PER_GROUP = 2
# ALPHA = 0.05
# DO_KW_GATE = True
#
# TITLE_FONTSIZE = 12
# TITLE_PAD = 18
# N_LABEL_PAD_FRAC = 0.14
# BRACKET_BASE_FRAC = 0.22
# BRACKET_STEP_FRAC = 0.12
# BRACKET_H_FRAC = 0.04
#
#
# with open(PEARSON_SUMMARY_PKL, "rb") as f:
#     rows = pickle.load(f)
#
# df = pd.DataFrame(rows)
#
# print("Loaded:", PEARSON_SUMMARY_PKL)
# print("Columns:", df.columns.tolist())
#
# required = ["roi", "group", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]
# for c in required:
#     if c not in df.columns:
#         raise RuntimeError(f"Missing column '{c}'")
#
# df["roi"] = df["roi"].astype(str)
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
#
# for col in ["r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df["r_corr"] = df["r_real"] - df["r_fake_mean"]
#
# rois = sorted(df["roi"].dropna().unique())
# print("ROIs:", rois)
#
#
# summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
# summary = (
#     df.groupby(["roi", "group"], observed=True)[summary_cols]
#       .agg(["count", "mean", "median", "std"])
# )
# summary.columns = ["_".join(c) for c in summary.columns]
# summary = summary.reset_index()
# summary.to_csv(SUMMARY_CSV, index=False)
# print("Saved summary CSV:", SUMMARY_CSV)
#
#
# def bh_fdr(pvals):
#     pvals = np.asarray(pvals, float)
#     m = len(pvals)
#     if m == 0:
#         return pvals
#     order = np.argsort(pvals)
#     ranked = pvals[order]
#     q = ranked * m / (np.arange(1, m + 1))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0, 1)
#     out = np.empty_like(q)
#     out[order] = q
#     return out
#
#
# def p_to_stars(p):
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""
#
#
# def add_sig_bracket(ax, x1, x2, y, text, h):
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
#     ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)
#
#
# def add_pairwise_sig(ax, vals_by_group):
#
#     if DO_KW_GATE:
#         groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
#         if len(groups_for_kw) < 2:
#             return
#         try:
#             _, p_kw = kruskal(*groups_for_kw)
#         except Exception:
#             return
#         if not np.isfinite(p_kw) or p_kw >= ALPHA:
#             return
#         ax.text(
#             0.02, 0.98, f"KW p={p_kw:.3g}",
#             transform=ax.transAxes, ha="left", va="top", fontsize=9
#         )
#
#     pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
#     raw_ps, valid_pairs = [], []
#
#     for a, b in pairs:
#         va, vb = vals_by_group[a], vals_by_group[b]
#         if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
#             try:
#                 _, p = mannwhitneyu(va, vb, alternative="two-sided")
#             except Exception:
#                 continue
#             raw_ps.append(float(p))
#             valid_pairs.append((a, b))
#
#     if not raw_ps:
#         return
#
#     qvals = bh_fdr(raw_ps)
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     base_y = y_max + BRACKET_BASE_FRAC * yr
#     step = BRACKET_STEP_FRAC * yr
#     h = BRACKET_H_FRAC * yr
#
#     x = {"fast": 1, "normal": 2, "slow": 3}
#     layer = 0
#
#     for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
#         if q < ALPHA:
#             label = p_to_stars(q) or f"q={q:.3f}"
#             y = base_y + layer * step
#             add_sig_bracket(ax, x[a], x[b], y, label, h)
#             layer += 1
#
#
# def add_n_labels(ax, vals_by_group):
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     y_n = y_max + N_LABEL_PAD_FRAC * yr
#     for j, g in enumerate(GROUP_ORDER, start=1):
#         n = len(vals_by_group[g])
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)
#
#     top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
#     cur_lo, cur_hi = ax.get_ylim()
#     ax.set_ylim(cur_lo, max(cur_hi, top))
#
#
# def plot_metric(metric: str, ylabel: str, out_pdf: Path):
#     if metric not in df.columns:
#         raise RuntimeError(f"Metric '{metric}' not found in df columns")
#
#     ncols = 3
#     nrows = int(np.ceil(len(rois) / ncols))
#
#     with PdfPages(out_pdf) as pdf:
#         fig, axes = plt.subplots(
#             nrows=nrows, ncols=ncols,
#             figsize=(5.2 * ncols, 3.9 * nrows),
#             sharey=False
#         )
#         axes = np.array(axes).reshape(-1)
#
#         rng = np.random.default_rng(0)
#
#         for i, roi in enumerate(rois):
#             ax = axes[i]
#             sub = df[df["roi"] == roi]
#
#             vals_by_group = {}
#             data = []
#             for g in GROUP_ORDER:
#                 v = sub.loc[sub["group"] == g, metric].to_numpy()
#                 v = v[np.isfinite(v)]
#                 vals_by_group[g] = v
#                 data.append(v)
#
#             ax.boxplot(
#                 data, positions=[1, 2, 3],
#                 widths=0.6, showfliers=False
#             )
#
#             for j, v in enumerate(data, start=1):
#                 if len(v):
#                     ax.scatter(
#                         j + rng.uniform(-0.15, 0.15, size=len(v)),
#                         v, s=18, alpha=0.6
#                     )
#
#             ax.axhline(0, linestyle="--", linewidth=1)
#             ax.set_xticks([1, 2, 3])
#             ax.set_xticklabels(GROUP_ORDER)
#             ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#             add_n_labels(ax, vals_by_group)
#             add_pairwise_sig(ax, vals_by_group)
#
#         for k in range(len(rois), len(axes)):
#             axes[k].axis("off")
#
#         fig.suptitle(f"{ylabel} by RT group", fontsize=14, y=0.995)
#         fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)
#
#         fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])
#
#         pdf.savefig(fig)
#         plt.close(fig)
#
#     print("Saved:", out_pdf)
#
#
# for col, ylabel, fname in METRICS:
#     plot_metric(col, ylabel, OUT_DIR / fname)
#
# print("Outputs in:", OUT_DIR)
# print("Summary CSV:", SUMMARY_CSV)


# from __future__ import annotations
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import kruskal, mannwhitneyu
#
#
# # =========================
# # INPUT PKLS
# # =========================
# PKL_FILES = [
#     Path("./prior_localization_sessionfit_output/pearson_summary_VISa_VISp_PL.pkl"),
#     Path("./prior_localization_sessionfit_output/pearson_summary_MOs_ACAd_MOp_ORBvl.pkl"),
# ]
#
# # Tag used in output filenames
# TAG = "ALL_ROIS_from_two_pkls"
#
# GROUP_ORDER = ["fast", "normal", "slow"]
#
# # Metrics to plot: (df_column, plot_ylabel, output_pdf_name)
# METRICS = [
#     ("r_real", "Pearson r (real)", f"{TAG}_pearson_r_real_sig_n.pdf"),
#     ("r_corr", "Corrected Pearson r = r_real − mean(r_pseudo)", f"{TAG}_r_corr_sig_n.pdf"),
#     ("z_corr", r"z = (r_real − mean(r_pseudo)) / std(r_pseudo)", f"{TAG}_z_corr_sig_n_eq.pdf"),
#     ("r2_corr", "Corrected R²", f"{TAG}_r2_corr_sig_n.pdf"),
# ]
#
# OUT_DIR = Path("./prior_localization_sessionfit_output/plots_all_rois")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# SUMMARY_CSV = OUT_DIR / f"summary_{TAG}_selected_metrics.csv"
#
# # =========================
# # FILTERING RULES
# # =========================
# MIN_TRIALS_FOR_METRICS = 10   # <-- your requirement: if subgroup has <10 trials, do not show it
# MIN_N_PER_GROUP = 2          # for pairwise tests to run
# ALPHA = 0.05
# DO_KW_GATE = True
#
# # Figure layout
# TITLE_FONTSIZE = 12
# TITLE_PAD = 18
# N_LABEL_PAD_FRAC = 0.14
# BRACKET_BASE_FRAC = 0.22
# BRACKET_STEP_FRAC = 0.12
# BRACKET_H_FRAC = 0.04
#
# # Where to save unioned EIDs from the PKLs
# SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output" / "roi_all"
# DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)
# EIDS_OUT_TXT = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.txt"
# EIDS_OUT_PKL = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.pkl"
#
#
# # =========================
# # Load + combine PKLs
# # =========================
# all_rows = []
# for p in PKL_FILES:
#     if not p.exists():
#         raise FileNotFoundError(f"Missing PKL: {p}")
#     with open(p, "rb") as f:
#         rows = pickle.load(f)
#     if not isinstance(rows, (list, tuple)):
#         raise RuntimeError(f"Expected list/tuple in {p}, got {type(rows)}")
#     print(f"Loaded {len(rows)} rows from: {p}")
#     all_rows.extend(rows)
#
# df = pd.DataFrame(all_rows)
#
# print("Combined rows:", len(df))
# print("Columns:", df.columns.tolist())
#
# # Required columns
# required = ["eid", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]
# for c in required:
#     if c not in df.columns:
#         raise RuntimeError(f"Missing column '{c}' in combined dataframe")
#
# # Clean up types
# df["eid"] = df["eid"].astype(str)
# df["roi"] = df["roi"].astype(str)
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
#
# # numeric columns
# for col in ["n_trials_group_used", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # derived metric
# df["r_corr"] = df["r_real"] - df["r_fake_mean"]
#
# # =========================
# # Filter out subgroup points with <10 trials
# # =========================
# before = len(df)
# df = df[df["n_trials_group_used"] >= MIN_TRIALS_FOR_METRICS].copy()
# after = len(df)
# print(f"Filtered by n_trials_group_used >= {MIN_TRIALS_FOR_METRICS}: {before} -> {after}")
#
# # ROIs present after filtering
# rois = sorted(df["roi"].dropna().unique())
# print("ROIs (after filtering):", rois)
# if len(rois) == 0:
#     raise RuntimeError("No ROIs left after filtering. Maybe MIN_TRIALS_FOR_METRICS too strict or data missing.")
#
#
# # =========================
# # Save summary CSV
# # =========================
# summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
# summary = (
#     df.groupby(["roi", "group"], observed=True)[summary_cols]
#       .agg(["count", "mean", "median", "std"])
# )
# summary.columns = ["_".join(c) for c in summary.columns]
# summary = summary.reset_index()
# summary.to_csv(SUMMARY_CSV, index=False)
# print("Saved summary CSV:", SUMMARY_CSV)
#
#
# # =========================
# # Save unioned EIDs from the PKLs (dedupe)
# # =========================
# eids_union = sorted(set(df["eid"].dropna().astype(str).tolist()))
# EIDS_OUT_TXT.write_text("\n".join(eids_union) + "\n")
# with open(EIDS_OUT_PKL, "wb") as f:
#     pickle.dump(eids_union, f)
#
# print(f"[EIDS] Unioned {len(eids_union)} unique EIDs from filtered rows")
# print("Saved:", EIDS_OUT_TXT)
# print("Saved:", EIDS_OUT_PKL)
#
#
# # =========================
# # Stats helpers (same as your current script)
# # =========================
# def bh_fdr(pvals):
#     pvals = np.asarray(pvals, float)
#     m = len(pvals)
#     if m == 0:
#         return pvals
#     order = np.argsort(pvals)
#     ranked = pvals[order]
#     q = ranked * m / (np.arange(1, m + 1))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0, 1)
#     out = np.empty_like(q)
#     out[order] = q
#     return out
#
#
# def p_to_stars(p):
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""
#
#
# def add_sig_bracket(ax, x1, x2, y, text, h):
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
#     ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)
#
#
# def add_pairwise_sig(ax, vals_by_group):
#
#     if DO_KW_GATE:
#         groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
#         if len(groups_for_kw) < 2:
#             return
#         try:
#             _, p_kw = kruskal(*groups_for_kw)
#         except Exception:
#             return
#         if not np.isfinite(p_kw) or p_kw >= ALPHA:
#             return
#         ax.text(
#             0.02, 0.98, f"KW p={p_kw:.3g}",
#             transform=ax.transAxes, ha="left", va="top", fontsize=9
#         )
#
#     pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
#     raw_ps, valid_pairs = [], []
#
#     for a, b in pairs:
#         va, vb = vals_by_group[a], vals_by_group[b]
#         if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
#             try:
#                 _, p = mannwhitneyu(va, vb, alternative="two-sided")
#             except Exception:
#                 continue
#             raw_ps.append(float(p))
#             valid_pairs.append((a, b))
#
#     if not raw_ps:
#         return
#
#     qvals = bh_fdr(raw_ps)
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     base_y = y_max + BRACKET_BASE_FRAC * yr
#     step = BRACKET_STEP_FRAC * yr
#     h = BRACKET_H_FRAC * yr
#
#     x = {"fast": 1, "normal": 2, "slow": 3}
#     layer = 0
#
#     for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
#         if q < ALPHA:
#             label = p_to_stars(q) or f"q={q:.3f}"
#             y = base_y + layer * step
#             add_sig_bracket(ax, x[a], x[b], y, label, h)
#             layer += 1
#
#
# def add_n_labels(ax, vals_by_group):
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     y_n = y_max + N_LABEL_PAD_FRAC * yr
#     for j, g in enumerate(GROUP_ORDER, start=1):
#         n = len(vals_by_group[g])
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)
#
#     top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
#     cur_lo, cur_hi = ax.get_ylim()
#     ax.set_ylim(cur_lo, max(cur_hi, top))
#
#
# # =========================
# # Plotting
# # =========================
# def plot_metric(metric: str, ylabel: str, out_pdf: Path):
#     if metric not in df.columns:
#         raise RuntimeError(f"Metric '{metric}' not found in df columns")
#
#     ncols = 3
#     nrows = int(np.ceil(len(rois) / ncols))
#
#     with PdfPages(out_pdf) as pdf:
#         fig, axes = plt.subplots(
#             nrows=nrows, ncols=ncols,
#             figsize=(5.2 * ncols, 3.9 * nrows),
#             sharey=False
#         )
#         axes = np.array(axes).reshape(-1)
#
#         rng = np.random.default_rng(0)
#
#         for i, roi in enumerate(rois):
#             ax = axes[i]
#             sub = df[df["roi"] == roi]
#
#             vals_by_group = {}
#             data = []
#             for g in GROUP_ORDER:
#                 v = sub.loc[sub["group"] == g, metric].to_numpy()
#                 v = v[np.isfinite(v)]
#                 vals_by_group[g] = v
#                 data.append(v)
#
#             # If all groups are empty, blank panel
#             if sum(len(v) for v in data) == 0:
#                 ax.axis("off")
#                 continue
#
#             ax.boxplot(
#                 data, positions=[1, 2, 3],
#                 widths=0.6, showfliers=False
#             )
#
#             for j, v in enumerate(data, start=1):
#                 if len(v):
#                     ax.scatter(
#                         j + rng.uniform(-0.15, 0.15, size=len(v)),
#                         v, s=18, alpha=0.6
#                     )
#
#             ax.axhline(0, linestyle="--", linewidth=1)
#             ax.set_xticks([1, 2, 3])
#             ax.set_xticklabels(GROUP_ORDER)
#             ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#             add_n_labels(ax, vals_by_group)
#             add_pairwise_sig(ax, vals_by_group)
#
#         for k in range(len(rois), len(axes)):
#             axes[k].axis("off")
#
#         fig.suptitle(f"{ylabel} by RT group",
#                      fontsize=14, y=0.995)
#         fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)
#
#         fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])
#
#         pdf.savefig(fig)
#         plt.close(fig)
#
#     print("Saved:", out_pdf)
#
#
# for col, ylabel, fname in METRICS:
#     plot_metric(col, ylabel, OUT_DIR / fname)
#
# print("Outputs in:", OUT_DIR)
# print("Summary CSV:", SUMMARY_CSV)
# print("Union EIDs:", EIDS_OUT_TXT, "and", EIDS_OUT_PKL)
#
# from __future__ import annotations
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import kruskal, mannwhitneyu, wilcoxon
#
#
# # =========================
# # INPUT PKLS
# # =========================
# PKL_FILES = [
#     Path("./prior_localization_sessionfit_output/pearson_summary_VISa_VISp_PL.pkl"),
#     Path("./prior_localization_sessionfit_output/pearson_summary_MOs_ACAd_MOp_ORBvl.pkl"),
# ]
#
# # Tag used in output filenames
# TAG = "ALL_ROIS_from_two_pkls"
#
# GROUP_ORDER = ["fast", "normal", "slow"]
#
# # Metrics to plot: (df_column, plot_ylabel, output_pdf_name)
# METRICS = [
#     ("r_real", "Pearson r (real)", f"{TAG}_pearson_r_real_sig_n.pdf"),
#     ("r_corr", "Corrected Pearson r = r_real − mean(r_pseudo)", f"{TAG}_r_corr_sig_n.pdf"),
#     ("z_corr", r"z = (r_real − mean(r_pseudo)) / std(r_pseudo)", f"{TAG}_z_corr_sig_n_eq.pdf"),
#     ("r2_corr", "Corrected R²", f"{TAG}_r2_corr_sig_n.pdf"),
# ]
#
# OUT_DIR = Path("./prior_localization_sessionfit_output/plots_all_rois")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# SUMMARY_CSV = OUT_DIR / f"summary_{TAG}_selected_metrics.csv"
#
# # =========================
# # FILTERING RULES
# # =========================
# MIN_TRIALS_FOR_METRICS = 10   # if subgroup has <10 trials, do not show it
# MIN_N_PER_GROUP = 2          # for pairwise tests to run
# ALPHA = 0.05
# DO_KW_GATE = True
#
# # For one-sample "is metric > 0" tests (Wilcoxon)
# MIN_N_FOR_GT0_TEST = 4       # require at least this many session-points per ROI×group
#
# # Figure layout
# TITLE_FONTSIZE = 12
# TITLE_PAD = 18
# N_LABEL_PAD_FRAC = 0.14
# BRACKET_BASE_FRAC = 0.22
# BRACKET_STEP_FRAC = 0.12
# BRACKET_H_FRAC = 0.04
#
# # Where to save unioned EIDs from the PKLs
# SCRIPT_DIR = Path(__file__).resolve().parent
# DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output" / "roi_all"
# DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)
# EIDS_OUT_TXT = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.txt"
# EIDS_OUT_PKL = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.pkl"
#
#
# # =========================
# # Load + combine PKLs
# # =========================
# all_rows = []
# for p in PKL_FILES:
#     if not p.exists():
#         raise FileNotFoundError(f"Missing PKL: {p}")
#     with open(p, "rb") as f:
#         rows = pickle.load(f)
#     if not isinstance(rows, (list, tuple)):
#         raise RuntimeError(f"Expected list/tuple in {p}, got {type(rows)}")
#     print(f"Loaded {len(rows)} rows from: {p}")
#     all_rows.extend(rows)
#
# df = pd.DataFrame(all_rows)
#
# print("Combined rows:", len(df))
# print("Columns:", df.columns.tolist())
#
# # Required columns
# required = ["eid", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]
# for c in required:
#     if c not in df.columns:
#         raise RuntimeError(f"Missing column '{c}' in combined dataframe")
#
# # Clean up types
# df["eid"] = df["eid"].astype(str)
# df["roi"] = df["roi"].astype(str)
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
#
# # numeric columns
# for col in ["n_trials_group_used", "r_real", "r_fake_mean", "r_fake_std", "z_corr", "r2_corr"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # derived metric
# df["r_corr"] = df["r_real"] - df["r_fake_mean"]
#
# # =========================
# # Filter out subgroup points with <10 trials
# # =========================
# before = len(df)
# df = df[df["n_trials_group_used"] >= MIN_TRIALS_FOR_METRICS].copy()
# after = len(df)
# print(f"Filtered by n_trials_group_used >= {MIN_TRIALS_FOR_METRICS}: {before} -> {after}")
#
# # ROIs present after filtering
# rois = sorted(df["roi"].dropna().unique())
# print("ROIs (after filtering):", rois)
# if len(rois) == 0:
#     raise RuntimeError("No ROIs left after filtering. Maybe MIN_TRIALS_FOR_METRICS too strict or data missing.")
#
#
# # =========================
# # Save summary CSV
# # =========================
# summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
# summary = (
#     df.groupby(["roi", "group"], observed=True)[summary_cols]
#       .agg(["count", "mean", "median", "std"])
# )
# summary.columns = ["_".join(c) for c in summary.columns]
# summary = summary.reset_index()
# summary.to_csv(SUMMARY_CSV, index=False)
# print("Saved summary CSV:", SUMMARY_CSV)
#
#
# # =========================
# # Save unioned EIDs from the PKLs (dedupe)
# # =========================
# eids_union = sorted(set(df["eid"].dropna().astype(str).tolist()))
# EIDS_OUT_TXT.write_text("\n".join(eids_union) + "\n")
# with open(EIDS_OUT_PKL, "wb") as f:
#     pickle.dump(eids_union, f)
#
# print(f"[EIDS] Unioned {len(eids_union)} unique EIDs from filtered rows")
# print("Saved:", EIDS_OUT_TXT)
# print("Saved:", EIDS_OUT_PKL)
#
#
# # =========================
# # Stats helpers
# # =========================
# def bh_fdr(pvals):
#     pvals = np.asarray(pvals, float)
#     m = len(pvals)
#     if m == 0:
#         return pvals
#     order = np.argsort(pvals)
#     ranked = pvals[order]
#     q = ranked * m / (np.arange(1, m + 1))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0, 1)
#     out = np.empty_like(q)
#     out[order] = q
#     return out
#
#
# def p_to_stars(p):
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""
#
#
# def add_sig_bracket(ax, x1, x2, y, text, h):
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
#     ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)
#
#
# def add_pairwise_sig(ax, vals_by_group):
#
#     if DO_KW_GATE:
#         groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
#         if len(groups_for_kw) < 2:
#             return
#         try:
#             _, p_kw = kruskal(*groups_for_kw)
#         except Exception:
#             return
#         if not np.isfinite(p_kw) or p_kw >= ALPHA:
#             return
#         ax.text(
#             0.02, 0.98, f"KW p={p_kw:.3g}",
#             transform=ax.transAxes, ha="left", va="top", fontsize=9
#         )
#
#     pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
#     raw_ps, valid_pairs = [], []
#
#     for a, b in pairs:
#         va, vb = vals_by_group[a], vals_by_group[b]
#         if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
#             try:
#                 _, p = mannwhitneyu(va, vb, alternative="two-sided")
#             except Exception:
#                 continue
#             raw_ps.append(float(p))
#             valid_pairs.append((a, b))
#
#     if not raw_ps:
#         return
#
#     qvals = bh_fdr(raw_ps)
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     base_y = y_max + BRACKET_BASE_FRAC * yr
#     step = BRACKET_STEP_FRAC * yr
#     h = BRACKET_H_FRAC * yr
#
#     x = {"fast": 1, "normal": 2, "slow": 3}
#     layer = 0
#
#     for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
#         if q < ALPHA:
#             label = p_to_stars(q) or f"q={q:.3f}"
#             y = base_y + layer * step
#             add_sig_bracket(ax, x[a], x[b], y, label, h)
#             layer += 1
#
#
# def add_n_labels(ax, vals_by_group):
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     y_n = y_max + N_LABEL_PAD_FRAC * yr
#     for j, g in enumerate(GROUP_ORDER, start=1):
#         n = len(vals_by_group[g])
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)
#
#     top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
#     cur_lo, cur_hi = ax.get_ylim()
#     ax.set_ylim(cur_lo, max(cur_hi, top))
#
#
# # =========================
# # NEW: One-sample test corrected R² > 0, ROI × RT subgroup
# # =========================
# def wilcoxon_gt0(values, min_n=MIN_N_FOR_GT0_TEST):
#     """
#     One-sample Wilcoxon signed-rank test:
#       H0: median(values) == 0
#       H1: median(values) > 0  (one-sided)
#     Returns p-value (float) or NaN if not testable.
#     """
#     v = np.asarray(values, float)
#     v = v[np.isfinite(v)]
#     if v.size < min_n:
#         return np.nan
#
#     # If all values are exactly 0, Wilcoxon is not defined
#     if np.allclose(v, 0):
#         return np.nan
#
#     try:
#         # zero_method="wilcox" drops zero-differences
#         _, p = wilcoxon(v, zero_method="wilcox", alternative="greater")
#         return float(p)
#     except Exception:
#         return np.nan
#
#
# # Build significance table for corrected R² > 0
# sig_rows = []
# for roi in rois:
#     for grp in GROUP_ORDER:
#         sub = df[(df["roi"] == roi) & (df["group"] == grp)]
#         vals = sub["r2_corr"].to_numpy()
#
#         n = int(np.sum(np.isfinite(vals)))
#         med = float(np.nanmedian(vals)) if n > 0 else np.nan
#         p = wilcoxon_gt0(vals)
#
#         sig_rows.append({
#             "roi": roi,
#             "group": grp,
#             "n_sessions": n,
#             "median_corrected_R2": med,
#             "p_corrected_R2_gt0": p,
#         })
#
# sig_df = pd.DataFrame(sig_rows)
# #sig_df["q_r2_corr_gt0"] = bh_fdr(sig_df["p_r2_corr_gt0"].values)
#
# SIG_CSV = OUT_DIR / f"r2_corr_gt0_tests_{TAG}.csv"
# sig_df.to_csv(SIG_CSV, index=False)
#
# print("\nCorrected R² > 0 (Wilcoxon, one-side)")
# print(sig_df.sort_values(["roi", "group"]).to_string(index=False))
# print("Saved:", SIG_CSV)
#
#
# # =========================
# # Plotting
# # =========================
# def plot_metric(metric: str, ylabel: str, out_pdf: Path):
#     if metric not in df.columns:
#         raise RuntimeError(f"Metric '{metric}' not found in df columns")
#
#     ncols = 3
#     nrows = int(np.ceil(len(rois) / ncols))
#
#     with PdfPages(out_pdf) as pdf:
#         fig, axes = plt.subplots(
#             nrows=nrows, ncols=ncols,
#             figsize=(5.2 * ncols, 3.9 * nrows),
#             sharey=False
#         )
#         axes = np.array(axes).reshape(-1)
#
#         rng = np.random.default_rng(0)
#
#         for i, roi in enumerate(rois):
#             ax = axes[i]
#             sub = df[df["roi"] == roi]
#
#             vals_by_group = {}
#             data = []
#             for g in GROUP_ORDER:
#                 v = sub.loc[sub["group"] == g, metric].to_numpy()
#                 v = v[np.isfinite(v)]
#                 vals_by_group[g] = v
#                 data.append(v)
#
#             # If all groups are empty, blank panel
#             if sum(len(v) for v in data) == 0:
#                 ax.axis("off")
#                 continue
#
#             ax.boxplot(
#                 data, positions=[1, 2, 3],
#                 widths=0.6, showfliers=False
#             )
#
#             for j, v in enumerate(data, start=1):
#                 if len(v):
#                     ax.scatter(
#                         j + rng.uniform(-0.15, 0.15, size=len(v)),
#                         v, s=18, alpha=0.6
#                     )
#
#             ax.axhline(0, linestyle="--", linewidth=1)
#             ax.set_xticks([1, 2, 3])
#             ax.set_xticklabels(GROUP_ORDER)
#             ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#             add_n_labels(ax, vals_by_group)
#             add_pairwise_sig(ax, vals_by_group)
#
#         for k in range(len(rois), len(axes)):
#             axes[k].axis("off")
#
#         fig.suptitle(f"{ylabel} by RT group",
#                      fontsize=14, y=0.995)
#         fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)
#
#         fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])
#
#         pdf.savefig(fig)
#         plt.close(fig)
#
#     print("Saved:", out_pdf)
#
#
# for col, ylabel, fname in METRICS:
#     plot_metric(col, ylabel, OUT_DIR / fname)
#
# print("\nOutputs in:", OUT_DIR)
# print("Summary CSV:", SUMMARY_CSV)
# print("Union EIDs:", EIDS_OUT_TXT, "and", EIDS_OUT_PKL)


# from __future__ import annotations
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import kruskal, mannwhitneyu, wilcoxon
#
# # ============================================================
# # PLOT STYLE (match your ggplot style)
# # ============================================================
# plt.style.use("ggplot")
# plt.rcParams.update(
#     {
#         "font.size": 12,
#         "axes.titlesize": 13,
#         "axes.labelsize": 13,
#         "legend.fontsize": 10,
#         "figure.titlesize": 14,
#         "savefig.bbox": "tight",
#     }
# )
#
# # =========================
# # INPUT PKL (UPDATED)
# # =========================
# PKL_FILES = [
#     Path(
#         "prior_localization_sessionfit_output_behav_sessions/"
#         "pearson_summary_BEHAV_SESSIONS_UPDATED_ROIS_"
#         "ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_SSp-tr_SSp-bfd_SSp-ll_SSp-ul_"
#         "SSp-n_SSs_VISp_VISpl_AUDp.pkl"
#     )
# ]
#
# # Tag used in output filenames
# TAG = "BEHAV_SESSIONS_UPDATED_ROIS_single_pkl"
#
# GROUP_ORDER = ["fast", "normal", "slow"]
#
# # Metrics to plot: (df_column, plot_ylabel, output_pdf_name)
# METRICS = [
#     ("r_real", "Pearson r (real)", f"{TAG}_pearson_r_real_sig_n.pdf"),
#     ("r_corr", "Corrected Pearson r", f"{TAG}_r_corr_sig_n.pdf"),
#     ("z_corr", "Z-score", f"{TAG}_z_corr_sig_n.pdf"),
#     ("r2_corr", "Corrected R2", f"{TAG}_r2_corr_sig_n.pdf"),
# ]
#
# OUT_DIR = Path("prior_localization_sessionfit_output_behav_sessions/plots_all_rois")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# SUMMARY_CSV = OUT_DIR / f"summary_{TAG}_selected_metrics.csv"
#
# # =========================
# # FILTERING RULES
# # =========================
# MIN_TRIALS_FOR_METRICS = 10   # if subgroup has <10 trials, do not show it
# MIN_N_PER_GROUP = 2          # for pairwise tests to run
# ALPHA = 0.05
# DO_KW_GATE = True
#
# # For one-sample "is metric > 0" tests (Wilcoxon)
# MIN_N_FOR_GT0_TEST = 4       # require at least this many session-points per ROI×group
#
# # Figure layout
# TITLE_FONTSIZE = 12
# TITLE_PAD = 18
# N_LABEL_PAD_FRAC = 0.14
# BRACKET_BASE_FRAC = 0.22
# BRACKET_STEP_FRAC = 0.12
# BRACKET_H_FRAC = 0.04
#
# # Where to save unioned EIDs from the PKLs
# # (avoid __file__ issues in notebooks)
# SCRIPT_DIR = Path.cwd()
# DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output_behav_sessions" / "roi_all"
# DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)
# EIDS_OUT_TXT = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.txt"
# EIDS_OUT_PKL = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.pkl"
#
#
# # =========================
# # Load + combine PKLs
# # =========================
# all_rows = []
# for p in PKL_FILES:
#     if not p.exists():
#         raise FileNotFoundError(f"Missing PKL: {p}")
#     with open(p, "rb") as f:
#         rows = pickle.load(f)
#     if not isinstance(rows, (list, tuple)):
#         raise RuntimeError(f"Expected list/tuple in {p}, got {type(rows)}")
#     print(f"Loaded {len(rows)} rows from: {p}")
#     all_rows.extend(rows)
#
# df = pd.DataFrame(all_rows)
#
# print("Combined rows:", len(df))
# print("Columns:", df.columns.tolist())
#
# # Required columns
# required = ["eid", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]
# missing = [c for c in required if c not in df.columns]
# if missing:
#     raise RuntimeError(f"Missing columns in combined dataframe: {missing}")
#
# # Clean up types
# df["eid"] = df["eid"].astype(str)
# df["roi"] = df["roi"].astype(str)
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
#
# # numeric columns
# for col in ["n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # derived metric
# df["r_corr"] = pd.to_numeric(df["r_real"], errors="coerce") - pd.to_numeric(df["r_fake_mean"], errors="coerce")
#
# # =========================
# # Filter out subgroup points with < MIN_TRIALS_FOR_METRICS trials
# # =========================
# before = len(df)
# df = df[df["n_trials_group_used"] >= MIN_TRIALS_FOR_METRICS].copy()
# after = len(df)
# print(f"Filtered by n_trials_group_used >= {MIN_TRIALS_FOR_METRICS}: {before} -> {after}")
#
# # ROIs present after filtering
# rois = sorted(df["roi"].dropna().unique())
# print("ROIs (after filtering):", rois)
# if len(rois) == 0:
#     raise RuntimeError("No ROIs left after filtering. Maybe MIN_TRIALS_FOR_METRICS too strict or data missing.")
#
# # =========================
# # Save summary CSV
# # =========================
# summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
# summary = (
#     df.groupby(["roi", "group"], observed=True)[summary_cols]
#       .agg(["count", "mean", "median", "std"])
# )
# summary.columns = ["_".join(c) for c in summary.columns]
# summary = summary.reset_index()
# summary.to_csv(SUMMARY_CSV, index=False)
# print("Saved summary CSV:", SUMMARY_CSV)
#
# # =========================
# # Save unioned EIDs from the PKL (dedupe)
# # =========================
# eids_union = sorted(set(df["eid"].dropna().astype(str).tolist()))
# EIDS_OUT_TXT.write_text("\n".join(eids_union) + "\n")
# with open(EIDS_OUT_PKL, "wb") as f:
#     pickle.dump(eids_union, f)
#
# print(f"[EIDS] Unioned {len(eids_union)} unique EIDs from filtered rows")
# print("Saved:", EIDS_OUT_TXT)
# print("Saved:", EIDS_OUT_PKL)
#
# # =========================
# # Stats helpers
# # =========================
# def bh_fdr(pvals):
#     pvals = np.asarray(pvals, float)
#     m = len(pvals)
#     if m == 0:
#         return pvals
#     order = np.argsort(pvals)
#     ranked = pvals[order]
#     q = ranked * m / (np.arange(1, m + 1))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0, 1)
#     out = np.empty_like(q)
#     out[order] = q
#     return out
#
# def p_to_stars(p):
#     if not np.isfinite(p):
#         return ""
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""
#
# def add_sig_bracket(ax, x1, x2, y, text, h):
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
#     ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)
#
# def add_pairwise_sig(ax, vals_by_group):
#
#     if DO_KW_GATE:
#         groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
#         if len(groups_for_kw) < 2:
#             return
#         try:
#             _, p_kw = kruskal(*groups_for_kw)
#         except Exception:
#             return
#         if not np.isfinite(p_kw) or p_kw >= ALPHA:
#             return
#         ax.text(
#             0.02, 0.98, f"p={p_kw:.3g}",
#             transform=ax.transAxes, ha="left", va="top", fontsize=9
#         )
#
#     pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
#     raw_ps, valid_pairs = [], []
#
#     for a, b in pairs:
#         va, vb = vals_by_group[a], vals_by_group[b]
#         if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
#             try:
#                 _, p = mannwhitneyu(va, vb, alternative="two-sided")
#             except Exception:
#                 continue
#             raw_ps.append(float(p))
#             valid_pairs.append((a, b))
#
#     if not raw_ps:
#         return
#
#     qvals = bh_fdr(raw_ps)
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     base_y = y_max + BRACKET_BASE_FRAC * yr
#     step = BRACKET_STEP_FRAC * yr
#     h = BRACKET_H_FRAC * yr
#
#     x = {"fast": 1, "normal": 2, "slow": 3}
#     layer = 0
#
#     for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
#         if q < ALPHA:
#             label = p_to_stars(q) or f"q={q:.3f}"
#             y = base_y + layer * step
#             add_sig_bracket(ax, x[a], x[b], y, label, h)
#             layer += 1
#
# def add_n_labels(ax, vals_by_group):
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     y_n = y_max + N_LABEL_PAD_FRAC * yr
#     for j, g in enumerate(GROUP_ORDER, start=1):
#         n = len(vals_by_group[g])
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)
#
#     top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
#     cur_lo, cur_hi = ax.get_ylim()
#     ax.set_ylim(cur_lo, max(cur_hi, top))
#
# # =========================
# # One-sample test: r2_corr > 0, ROI × RT subgroup (Wilcoxon, one-sided)
# # =========================
# def wilcoxon_gt0(values, min_n=MIN_N_FOR_GT0_TEST):
#     v = np.asarray(values, float)
#     v = v[np.isfinite(v)]
#     if v.size < min_n:
#         return np.nan
#     if np.allclose(v, 0):
#         return np.nan
#     try:
#         _, p = wilcoxon(v, zero_method="wilcox", alternative="greater")
#         return float(p)
#     except Exception:
#         return np.nan
#
# sig_rows = []
# for roi in rois:
#     for grp in GROUP_ORDER:
#         sub = df[(df["roi"] == roi) & (df["group"] == grp)]
#         vals = pd.to_numeric(sub["r2_corr"], errors="coerce").to_numpy()
#
#         n = int(np.sum(np.isfinite(vals)))
#         med = float(np.nanmedian(vals)) if n > 0 else np.nan
#         p = wilcoxon_gt0(vals)
#
#         sig_rows.append(
#             {
#                 "roi": roi,
#                 "group": grp,
#                 "n_sessions": n,
#                 "median_corrected_R2": med,
#                 "p_corrected_R2_gt0": p,
#             }
#         )
#
# sig_df = pd.DataFrame(sig_rows)
# SIG_CSV = OUT_DIR / f"r2_corr_gt0_tests_{TAG}.csv"
# sig_df.to_csv(SIG_CSV, index=False)
#
# print("\nCorrected R2 > 0 (Wilcoxon, one-sided)")
# print(sig_df.sort_values(["roi", "group"]).to_string(index=False))
# print("Saved:", SIG_CSV)
#
# # =========================
# # Plotting
# # =========================
# def plot_metric(metric: str, ylabel: str, out_pdf: Path):
#     if metric not in df.columns:
#         raise RuntimeError(f"Metric '{metric}' not found in df columns")
#
#     ncols = 3
#     nrows = int(np.ceil(len(rois) / ncols))
#
#     with PdfPages(out_pdf) as pdf:
#         fig, axes = plt.subplots(
#             nrows=nrows,
#             ncols=ncols,
#             figsize=(5.2 * ncols, 3.9 * nrows),
#             sharey=False,
#         )
#         axes = np.array(axes).reshape(-1)
#
#         rng = np.random.default_rng(0)
#
#         for i, roi in enumerate(rois):
#             ax = axes[i]
#             sub = df[df["roi"] == roi]
#
#             vals_by_group = {}
#             data = []
#             for g in GROUP_ORDER:
#                 v = sub.loc[sub["group"] == g, metric].to_numpy()
#                 v = pd.to_numeric(v, errors="coerce")
#                 v = v[np.isfinite(v)]
#                 vals_by_group[g] = v
#                 data.append(v)
#
#             # If all groups are empty, blank panel
#             if sum(len(v) for v in data) == 0:
#                 ax.axis("off")
#                 continue
#
#             ax.boxplot(
#                 data,
#                 positions=[1, 2, 3],
#                 widths=0.6,
#                 showfliers=False,
#             )
#
#             for j, v in enumerate(data, start=1):
#                 if len(v):
#                     ax.scatter(
#                         j + rng.uniform(-0.15, 0.15, size=len(v)),
#                         v,
#                         s=18,
#                         alpha=0.6,
#                     )
#
#             ax.axhline(0, linestyle="--", linewidth=1)
#             ax.set_xticks([1, 2, 3])
#             ax.set_xticklabels(GROUP_ORDER)
#             ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#
#             add_n_labels(ax, vals_by_group)
#             add_pairwise_sig(ax, vals_by_group)
#
#         for k in range(len(rois), len(axes)):
#             axes[k].axis("off")
#
#         fig.suptitle(f"{ylabel} by RT group", fontsize=14, y=0.995)
#         fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)
#
#         fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])
#
#         pdf.savefig(fig)
#         plt.close(fig)
#
#     print("Saved:", out_pdf)
#
# for col, ylabel, fname in METRICS:
#     plot_metric(col, ylabel, OUT_DIR / fname)
#
# print("\nOutputs in:", OUT_DIR)
# print("Summary CSV:", SUMMARY_CSV)
# print("Union EIDs:", EIDS_OUT_TXT, "and", EIDS_OUT_PKL)

#
# from __future__ import annotations
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from scipy.stats import kruskal, mannwhitneyu, wilcoxon
#
# # ============================================================
# # PLOT STYLE (match your ggplot style)
# # ============================================================
# plt.style.use("ggplot")
# plt.rcParams.update(
#     {
#         "font.size": 12,
#         "axes.titlesize": 13,
#         "axes.labelsize": 13,
#         "legend.fontsize": 10,
#         "figure.titlesize": 14,
#         "savefig.bbox": "tight",
#     }
# )
#
# # =========================
# # INPUT PKL (UPDATED)
# # =========================
# PKL_FILES = [
#     Path(
#         "prior_localization_sessionfit_output_behav_sessions/"
#         "pearson_summary_BEHAV_SESSIONS_UPDATED_ROIS_"
#         "ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_SSp-tr_SSp-bfd_SSp-ll_SSp-ul_"
#         "SSp-n_SSs_VISp_VISpl_AUDp.pkl"
#     )
# ]
#
# # Tag used in output filenames
# TAG = "BEHAV_SESSIONS_UPDATED_ROIS_single_pkl"
#
# GROUP_ORDER = ["fast", "normal", "slow"]
#
# # Advisor convention: early/normal/late colors = y/k/m
# GROUP_COLORS = {
#     "fast": "y",     # early RT
#     "normal": "k",   # normal RT
#     "slow": "m",     # late RT
# }
#
# # Metrics to plot: (df_column, plot_ylabel, output_pdf_name)
# METRICS = [
#     ("r_real", "Pearson r (real)", f"{TAG}_pearson_r_real_sig_n.pdf"),
#     ("r_corr", "Corrected Pearson r", f"{TAG}_r_corr_sig_n.pdf"),
#     ("z_corr", "Z-score", f"{TAG}_z_corr_sig_n.pdf"),
#     ("r2_corr", "Corrected R2", f"{TAG}_r2_corr_sig_n.pdf"),
# ]
#
# OUT_DIR = Path("prior_localization_sessionfit_output_behav_sessions/plots_all_rois")
# OUT_DIR.mkdir(parents=True, exist_ok=True)
#
# SUMMARY_CSV = OUT_DIR / f"summary_{TAG}_selected_metrics.csv"
#
# # =========================
# # FILTERING RULES
# # =========================
# MIN_TRIALS_FOR_METRICS = 10   # if subgroup has <10 trials, do not show it
# MIN_N_PER_GROUP = 2          # for pairwise tests to run
# ALPHA = 0.05
# DO_KW_GATE = True
#
# # For one-sample "is metric > 0" tests (Wilcoxon)
# MIN_N_FOR_GT0_TEST = 4       # require at least this many session-points per ROI×group
#
# # Figure layout
# TITLE_FONTSIZE = 12
# TITLE_PAD = 18
# N_LABEL_PAD_FRAC = 0.14
# BRACKET_BASE_FRAC = 0.22
# BRACKET_STEP_FRAC = 0.12
# BRACKET_H_FRAC = 0.04
#
# # Where to save unioned EIDs from the PKLs
# # (avoid __file__ issues in notebooks)
# SCRIPT_DIR = Path.cwd()
# DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output_behav_sessions" / "roi_all"
# DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)
# EIDS_OUT_TXT = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.txt"
# EIDS_OUT_PKL = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.pkl"
#
#
# # =========================
# # Load + combine PKLs
# # =========================
# all_rows = []
# for p in PKL_FILES:
#     if not p.exists():
#         raise FileNotFoundError(f"Missing PKL: {p}")
#     with open(p, "rb") as f:
#         rows = pickle.load(f)
#     if not isinstance(rows, (list, tuple)):
#         raise RuntimeError(f"Expected list/tuple in {p}, got {type(rows)}")
#     print(f"Loaded {len(rows)} rows from: {p}")
#     all_rows.extend(rows)
#
# df = pd.DataFrame(all_rows)
#
# print("Combined rows:", len(df))
# print("Columns:", df.columns.tolist())
#
# # Required columns
# required = ["eid", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]
# missing = [c for c in required if c not in df.columns]
# if missing:
#     raise RuntimeError(f"Missing columns in combined dataframe: {missing}")
#
# # Clean up types
# df["eid"] = df["eid"].astype(str)
# df["roi"] = df["roi"].astype(str)
# df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)
#
# # numeric columns
# for col in ["n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # derived metric
# df["r_corr"] = pd.to_numeric(df["r_real"], errors="coerce") - pd.to_numeric(df["r_fake_mean"], errors="coerce")
#
# # =========================
# # Filter out subgroup points with < MIN_TRIALS_FOR_METRICS trials
# # =========================
# before = len(df)
# df = df[df["n_trials_group_used"] >= MIN_TRIALS_FOR_METRICS].copy()
# after = len(df)
# print(f"Filtered by n_trials_group_used >= {MIN_TRIALS_FOR_METRICS}: {before} -> {after}")
#
# # ROIs present after filtering
# rois = sorted(df["roi"].dropna().unique())
# print("ROIs (after filtering):", rois)
# if len(rois) == 0:
#     raise RuntimeError("No ROIs left after filtering. Maybe MIN_TRIALS_FOR_METRICS too strict or data missing.")
#
# # =========================
# # Save summary CSV
# # =========================
# summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
# summary = (
#     df.groupby(["roi", "group"], observed=True)[summary_cols]
#       .agg(["count", "mean", "median", "std"])
# )
# summary.columns = ["_".join(c) for c in summary.columns]
# summary = summary.reset_index()
# summary.to_csv(SUMMARY_CSV, index=False)
# print("Saved summary CSV:", SUMMARY_CSV)
#
# # =========================
# # Save unioned EIDs from the PKL (dedupe)
# # =========================
# eids_union = sorted(set(df["eid"].dropna().astype(str).tolist()))
# EIDS_OUT_TXT.write_text("\n".join(eids_union) + "\n")
# with open(EIDS_OUT_PKL, "wb") as f:
#     pickle.dump(eids_union, f)
#
# print(f"[EIDS] Unioned {len(eids_union)} unique EIDs from filtered rows")
# print("Saved:", EIDS_OUT_TXT)
# print("Saved:", EIDS_OUT_PKL)
#
# # =========================
# # Stats helpers
# # =========================
# def bh_fdr(pvals):
#     pvals = np.asarray(pvals, float)
#     m = len(pvals)
#     if m == 0:
#         return pvals
#     order = np.argsort(pvals)
#     ranked = pvals[order]
#     q = ranked * m / (np.arange(1, m + 1))
#     q = np.minimum.accumulate(q[::-1])[::-1]
#     q = np.clip(q, 0, 1)
#     out = np.empty_like(q)
#     out[order] = q
#     return out
#
# def p_to_stars(p):
#     if not np.isfinite(p):
#         return ""
#     if p < 0.001:
#         return "***"
#     if p < 0.01:
#         return "**"
#     if p < 0.05:
#         return "*"
#     return ""
#
# def add_sig_bracket(ax, x1, x2, y, text, h):
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
#     ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)
#
# def add_pairwise_sig(ax, vals_by_group):
#
#     if DO_KW_GATE:
#         groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
#         if len(groups_for_kw) < 2:
#             return
#         try:
#             _, p_kw = kruskal(*groups_for_kw)
#         except Exception:
#             return
#         if not np.isfinite(p_kw) or p_kw >= ALPHA:
#             return
#         ax.text(
#             0.02, 0.98, f"p={p_kw:.3g}",
#             transform=ax.transAxes, ha="left", va="top", fontsize=9
#         )
#
#     pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
#     raw_ps, valid_pairs = [], []
#
#     for a, b in pairs:
#         va, vb = vals_by_group[a], vals_by_group[b]
#         if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
#             try:
#                 _, p = mannwhitneyu(va, vb, alternative="two-sided")
#             except Exception:
#                 continue
#             raw_ps.append(float(p))
#             valid_pairs.append((a, b))
#
#     if not raw_ps:
#         return
#
#     qvals = bh_fdr(raw_ps)
#
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     base_y = y_max + BRACKET_BASE_FRAC * yr
#     step = BRACKET_STEP_FRAC * yr
#     h = BRACKET_H_FRAC * yr
#
#     x = {"fast": 1, "normal": 2, "slow": 3}
#     layer = 0
#
#     for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
#         if q < ALPHA:
#             label = p_to_stars(q) or f"q={q:.3f}"
#             y = base_y + layer * step
#             add_sig_bracket(ax, x[a], x[b], y, label, h)
#             layer += 1
#
# def add_n_labels(ax, vals_by_group):
#     all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
#         if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
#     y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
#     yr = max(1e-6, y_max - y_min)
#
#     y_n = y_max + N_LABEL_PAD_FRAC * yr
#     for j, g in enumerate(GROUP_ORDER, start=1):
#         n = len(vals_by_group[g])
#         ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)
#
#     top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
#     cur_lo, cur_hi = ax.get_ylim()
#     ax.set_ylim(cur_lo, max(cur_hi, top))
#
# # =========================
# # One-sample test: r2_corr > 0, ROI × RT subgroup (Wilcoxon, one-sided)
# # =========================
# def wilcoxon_gt0(values, min_n=MIN_N_FOR_GT0_TEST):
#     v = np.asarray(values, float)
#     v = v[np.isfinite(v)]
#     if v.size < min_n:
#         return np.nan
#     if np.allclose(v, 0):
#         return np.nan
#     try:
#         _, p = wilcoxon(v, zero_method="wilcox", alternative="greater")
#         return float(p)
#     except Exception:
#         return np.nan
#
# sig_rows = []
# for roi in rois:
#     for grp in GROUP_ORDER:
#         sub = df[(df["roi"] == roi) & (df["group"] == grp)]
#         vals = pd.to_numeric(sub["r2_corr"], errors="coerce").to_numpy()
#
#         n = int(np.sum(np.isfinite(vals)))
#         med = float(np.nanmedian(vals)) if n > 0 else np.nan
#         p = wilcoxon_gt0(vals)
#
#         sig_rows.append(
#             {
#                 "roi": roi,
#                 "group": grp,
#                 "n_sessions": n,
#                 "median_corrected_R2": med,
#                 "p_corrected_R2_gt0": p,
#             }
#         )
#
# sig_df = pd.DataFrame(sig_rows)
# SIG_CSV = OUT_DIR / f"r2_corr_gt0_tests_{TAG}.csv"
# sig_df.to_csv(SIG_CSV, index=False)
#
# print("\nCorrected R2 > 0 (Wilcoxon, one-sided)")
# print(sig_df.sort_values(["roi", "group"]).to_string(index=False))
# print("Saved:", SIG_CSV)
#
# # =========================
# # Plotting
# #   Changes requested:
# #     1) REMOVE y=0 dashed line
# #     2) Color-code groups: fast='y', normal='k', slow='m'
# # =========================
# def plot_metric(metric: str, ylabel: str, out_pdf: Path):
#     if metric not in df.columns:
#         raise RuntimeError(f"Metric '{metric}' not found in df columns")
#
#     ncols = 3
#     nrows = int(np.ceil(len(rois) / ncols))
#
#     with PdfPages(out_pdf) as pdf:
#         fig, axes = plt.subplots(
#             nrows=nrows,
#             ncols=ncols,
#             figsize=(5.2 * ncols, 3.9 * nrows),
#             sharey=False,
#         )
#         axes = np.array(axes).reshape(-1)
#
#         rng = np.random.default_rng(0)
#
#         for i, roi in enumerate(rois):
#             ax = axes[i]
#             sub = df[df["roi"] == roi]
#
#             vals_by_group = {}
#             data = []
#             colors = []
#
#             for g in GROUP_ORDER:
#                 v = sub.loc[sub["group"] == g, metric].to_numpy()
#                 v = pd.to_numeric(v, errors="coerce")
#                 v = v[np.isfinite(v)]
#                 vals_by_group[g] = v
#                 data.append(v)
#                 colors.append(GROUP_COLORS[g])
#
#             # If all groups are empty, blank panel
#             if sum(len(v) for v in data) == 0:
#                 ax.axis("off")
#                 continue
#
#             # --- boxplot (colored by group) ---
#             bp = ax.boxplot(
#                 data,
#                 positions=[1, 2, 3],
#                 widths=0.6,
#                 showfliers=False,
#                 patch_artist=True,
#                 medianprops=dict(color="white", linewidth=1.4),
#                 whiskerprops=dict(color="black", linewidth=1.0),
#                 capprops=dict(color="black", linewidth=1.0),
#                 boxprops=dict(edgecolor="black", linewidth=1.0),
#             )
#
#             for patch, c in zip(bp["boxes"], colors):
#                 patch.set_facecolor(c)
#                 patch.set_alpha(0.85)
#
#             # --- scatter (colored by group) ---
#             for j, (g, v) in enumerate(zip(GROUP_ORDER, data), start=1):
#                 if len(v):
#                     ax.scatter(
#                         j + rng.uniform(-0.15, 0.15, size=len(v)),
#                         v,
#                         s=18,
#                         alpha=0.65,
#                         color=GROUP_COLORS[g],
#                         edgecolors="none",
#                     )
#
#             # REMOVED per request:
#             # ax.axhline(0, linestyle="--", linewidth=1)
#
#             ax.set_xticks([1, 2, 3])
#             ax.set_xticklabels(GROUP_ORDER)
#             ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)
#
#             add_n_labels(ax, vals_by_group)
#             add_pairwise_sig(ax, vals_by_group)
#
#         for k in range(len(rois), len(axes)):
#             axes[k].axis("off")
#
#         fig.suptitle(f"{ylabel} by RT group", fontsize=14, y=0.995)
#         fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)
#
#         fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])
#
#         pdf.savefig(fig)
#         plt.close(fig)
#
#     print("Saved:", out_pdf)
#
# for col, ylabel, fname in METRICS:
#     plot_metric(col, ylabel, OUT_DIR / fname)
#
# print("\nOutputs in:", OUT_DIR)
# print("Summary CSV:", SUMMARY_CSV)
# print("Union EIDs:", EIDS_OUT_TXT, "and", EIDS_OUT_PKL)

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

# ============================================================
# PLOT STYLE (match your ggplot style)
# ============================================================
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "savefig.bbox": "tight",
    }
)

# =========================
# INPUT PKL (UPDATED)
# =========================
PKL_FILES = [
    Path(
        "prior_localization_sessionfit_output_behav_sessions/"
        "pearson_summary_BEHAV_SESSIONS_UPDATED_ROIS_"
        "ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_SSp-tr_SSp-bfd_SSp-ll_SSp-ul_"
        "SSp-n_SSs_VISp_VISpl_AUDp.pkl"
    )
]

# Tag used in output filenames
TAG = "BEHAV_SESSIONS_UPDATED_ROIS_single_pkl"

GROUP_ORDER = ["fast", "normal", "slow"]

# Advisor convention: early/normal/late dot colors = y/k/m
GROUP_COLORS = {
    "fast": "y",     # early RT
    "normal": "k",   # normal RT
    "slow": "m",     # late RT
}

# Metrics to plot: (df_column, plot_ylabel, output_pdf_name)
METRICS = [
    ("r_real", "Pearson r (real)", f"{TAG}_pearson_r_real_sig_n.pdf"),
    ("r_corr", "Corrected Pearson r", f"{TAG}_r_corr_sig_n.pdf"),
    ("z_corr", "Z-score", f"{TAG}_z_corr_sig_n.pdf"),
    ("r2_corr", "Corrected R2", f"{TAG}_r2_corr_sig_n.pdf"),
]

OUT_DIR = Path("prior_localization_sessionfit_output_behav_sessions/plots_all_rois")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUT_DIR / f"summary_{TAG}_selected_metrics.csv"

# =========================
# FILTERING RULES
# =========================
MIN_TRIALS_FOR_METRICS = 10   # if subgroup has <10 trials, do not show it
MIN_N_PER_GROUP = 2          # for pairwise tests to run
ALPHA = 0.05
DO_KW_GATE = True

# For one-sample "is metric > 0" tests (Wilcoxon)
MIN_N_FOR_GT0_TEST = 4       # require at least this many session-points per ROI×group

# Figure layout
TITLE_FONTSIZE = 12
TITLE_PAD = 18
N_LABEL_PAD_FRAC = 0.14
BRACKET_BASE_FRAC = 0.22
BRACKET_STEP_FRAC = 0.12
BRACKET_H_FRAC = 0.04

# Where to save unioned EIDs from the PKLs
# (avoid __file__ issues in notebooks)
SCRIPT_DIR = Path.cwd()
DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output_behav_sessions" / "roi_all"
DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)
EIDS_OUT_TXT = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.txt"
EIDS_OUT_PKL = DEFAULT_EID_DIR / f"eids_union_FROM_PKLS_{TAG}.pkl"


# =========================
# Load + combine PKLs
# =========================
all_rows = []
for p in PKL_FILES:
    if not p.exists():
        raise FileNotFoundError(f"Missing PKL: {p}")
    with open(p, "rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, (list, tuple)):
        raise RuntimeError(f"Expected list/tuple in {p}, got {type(rows)}")
    print(f"Loaded {len(rows)} rows from: {p}")
    all_rows.extend(rows)

df = pd.DataFrame(all_rows)

print("Combined rows:", len(df))
print("Columns:", df.columns.tolist())

# Required columns
required = ["eid", "roi", "group", "n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in combined dataframe: {missing}")

# Clean up types
df["eid"] = df["eid"].astype(str)
df["roi"] = df["roi"].astype(str)
df["group"] = pd.Categorical(df["group"], categories=GROUP_ORDER, ordered=True)

# numeric columns
for col in ["n_trials_group_used", "r_real", "r_fake_mean", "z_corr", "r2_corr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# derived metric
df["r_corr"] = pd.to_numeric(df["r_real"], errors="coerce") - pd.to_numeric(df["r_fake_mean"], errors="coerce")

# =========================
# Filter out subgroup points with < MIN_TRIALS_FOR_METRICS trials
# =========================
before = len(df)
df = df[df["n_trials_group_used"] >= MIN_TRIALS_FOR_METRICS].copy()
after = len(df)
print(f"Filtered by n_trials_group_used >= {MIN_TRIALS_FOR_METRICS}: {before} -> {after}")

# ROIs present after filtering
rois = sorted(df["roi"].dropna().unique())
print("ROIs (after filtering):", rois)
if len(rois) == 0:
    raise RuntimeError("No ROIs left after filtering. Maybe MIN_TRIALS_FOR_METRICS too strict or data missing.")

# =========================
# Save summary CSV
# =========================
summary_cols = ["r_real", "r_corr", "z_corr", "r2_corr"]
summary = (
    df.groupby(["roi", "group"], observed=True)[summary_cols]
      .agg(["count", "mean", "median", "std"])
)
summary.columns = ["_".join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv(SUMMARY_CSV, index=False)
print("Saved summary CSV:", SUMMARY_CSV)

# =========================
# Save unioned EIDs from the PKL (dedupe)
# =========================
eids_union = sorted(set(df["eid"].dropna().astype(str).tolist()))
EIDS_OUT_TXT.write_text("\n".join(eids_union) + "\n")
with open(EIDS_OUT_PKL, "wb") as f:
    pickle.dump(eids_union, f)

print(f"[EIDS] Unioned {len(eids_union)} unique EIDs from filtered rows")
print("Saved:", EIDS_OUT_TXT)
print("Saved:", EIDS_OUT_PKL)

# =========================
# Stats helpers
# =========================
def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def p_to_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def add_sig_bracket(ax, x1, x2, y, text, h):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=9)

def add_pairwise_sig(ax, vals_by_group):
    if DO_KW_GATE:
        groups_for_kw = [v for v in vals_by_group.values() if len(v) >= MIN_N_PER_GROUP]
        if len(groups_for_kw) < 2:
            return
        try:
            _, p_kw = kruskal(*groups_for_kw)
        except Exception:
            return
        if not np.isfinite(p_kw) or p_kw >= ALPHA:
            return
        ax.text(
            0.02, 0.98, f"p={p_kw:.3g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9
        )

    pairs = [("fast", "normal"), ("fast", "slow"), ("normal", "slow")]
    raw_ps, valid_pairs = [], []

    for a, b in pairs:
        va, vb = vals_by_group[a], vals_by_group[b]
        if len(va) >= MIN_N_PER_GROUP and len(vb) >= MIN_N_PER_GROUP:
            try:
                _, p = mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                continue
            raw_ps.append(float(p))
            valid_pairs.append((a, b))

    if not raw_ps:
        return

    qvals = bh_fdr(raw_ps)

    all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
        if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
    y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
    y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
    yr = max(1e-6, y_max - y_min)

    base_y = y_max + BRACKET_BASE_FRAC * yr
    step = BRACKET_STEP_FRAC * yr
    h = BRACKET_H_FRAC * yr

    x = {"fast": 1, "normal": 2, "slow": 3}
    layer = 0

    for (a, b), q in sorted(zip(valid_pairs, qvals), key=lambda t: t[1]):
        if q < ALPHA:
            label = p_to_stars(q) or f"q={q:.3f}"
            y = base_y + layer * step
            add_sig_bracket(ax, x[a], x[b], y, label, h)
            layer += 1

def add_n_labels(ax, vals_by_group):
    all_vals = np.concatenate([v for v in vals_by_group.values() if len(v)], axis=0) \
        if any(len(v) for v in vals_by_group.values()) else np.array([0.0])
    y_max = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 0.0
    y_min = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
    yr = max(1e-6, y_max - y_min)

    y_n = y_max + N_LABEL_PAD_FRAC * yr
    for j, g in enumerate(GROUP_ORDER, start=1):
        n = len(vals_by_group[g])
        ax.text(j, y_n, f"n={n}", ha="center", va="bottom", fontsize=9)

    top = y_max + max(N_LABEL_PAD_FRAC + 0.22, 0.35) * yr
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, max(cur_hi, top))

# =========================
# One-sample test: r2_corr > 0, ROI × RT subgroup (Wilcoxon, one-sided)
# =========================
def wilcoxon_gt0(values, min_n=MIN_N_FOR_GT0_TEST):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < min_n:
        return np.nan
    if np.allclose(v, 0):
        return np.nan
    try:
        _, p = wilcoxon(v, zero_method="wilcox", alternative="greater")
        return float(p)
    except Exception:
        return np.nan

sig_rows = []
for roi in rois:
    for grp in GROUP_ORDER:
        sub = df[(df["roi"] == roi) & (df["group"] == grp)]
        vals = pd.to_numeric(sub["r2_corr"], errors="coerce").to_numpy()

        n = int(np.sum(np.isfinite(vals)))
        med = float(np.nanmedian(vals)) if n > 0 else np.nan
        p = wilcoxon_gt0(vals)

        sig_rows.append(
            {
                "roi": roi,
                "group": grp,
                "n_sessions": n,
                "median_corrected_R2": med,
                "p_corrected_R2_gt0": p,
            }
        )

sig_df = pd.DataFrame(sig_rows)
SIG_CSV = OUT_DIR / f"r2_corr_gt0_tests_{TAG}.csv"
sig_df.to_csv(SIG_CSV, index=False)

print("\nCorrected R2 > 0 (Wilcoxon, one-sided)")
print(sig_df.sort_values(["roi", "group"]).to_string(index=False))
print("Saved:", SIG_CSV)

# =========================
# Plotting
#   Changes requested:
#     1) REMOVE y=0 dashed line
#     2) Color ONLY the dots by group: fast='y', normal='k', slow='m'
#     3) Do NOT fill the boxes (keep default boxplot appearance)
# =========================
def plot_metric(metric: str, ylabel: str, out_pdf: Path):
    if metric not in df.columns:
        raise RuntimeError(f"Metric '{metric}' not found in df columns")

    ncols = 3
    nrows = int(np.ceil(len(rois) / ncols))

    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 3.9 * nrows),
            sharey=False,
        )
        axes = np.array(axes).reshape(-1)

        rng = np.random.default_rng(0)

        for i, roi in enumerate(rois):
            ax = axes[i]
            sub = df[df["roi"] == roi]

            vals_by_group = {}
            data = []
            for g in GROUP_ORDER:
                v = sub.loc[sub["group"] == g, metric].to_numpy()
                v = pd.to_numeric(v, errors="coerce")
                v = v[np.isfinite(v)]
                vals_by_group[g] = v
                data.append(v)

            # If all groups are empty, blank panel
            if sum(len(v) for v in data) == 0:
                ax.axis("off")
                continue

            # # Boxplot: unchanged (no colored fills)
            # ax.boxplot(
            #     data,
            #     positions=[1, 2, 3],
            #     widths=0.6,
            #     showfliers=False,
            # )
            bp = ax.boxplot(
                data,
                positions=[1, 2, 3],
                widths=0.6,
                showfliers=False,
                patch_artist=False,  # IMPORTANT: no filled boxes
            )

            # Color box outlines, whiskers, caps, medians by group
            for i, g in enumerate(GROUP_ORDER):
                color = GROUP_COLORS[g]

                # box outline
                bp["boxes"][i].set(color=color, linewidth=1.5)

                # whiskers (2 per box)
                bp["whiskers"][2 * i].set(color=color, linewidth=1.2)
                bp["whiskers"][2 * i + 1].set(color=color, linewidth=1.2)

                # caps (2 per box)
                bp["caps"][2 * i].set(color=color, linewidth=1.2)
                bp["caps"][2 * i + 1].set(color=color, linewidth=1.2)

                # median line
                bp["medians"][i].set(color=color, linewidth=1.5)

            # Scatter: color by group (advisor convention)
            for j, g in enumerate(GROUP_ORDER, start=1):
                v = vals_by_group[g]
                if len(v):
                    ax.scatter(
                        j + rng.uniform(-0.15, 0.15, size=len(v)),
                        v,
                        s=18,
                        alpha=0.6,
                        color=GROUP_COLORS[g],
                    )

            # REMOVED per request:
            # ax.axhline(0, linestyle="--", linewidth=1)

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(GROUP_ORDER)
            ax.set_title(roi, fontsize=TITLE_FONTSIZE, pad=TITLE_PAD)

            add_n_labels(ax, vals_by_group)
            add_pairwise_sig(ax, vals_by_group)

        for k in range(len(rois), len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"{ylabel} by RT group", fontsize=14, y=0.995)
        fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=12)

        fig.tight_layout(rect=[0.03, 0.02, 1, 0.965])

        pdf.savefig(fig)
        plt.close(fig)

    print("Saved:", out_pdf)

for col, ylabel, fname in METRICS:
    plot_metric(col, ylabel, OUT_DIR / fname)

print("\nOutputs in:", OUT_DIR)
print("Summary CSV:", SUMMARY_CSV)
print("Union EIDs:", EIDS_OUT_TXT, "and", EIDS_OUT_PKL)

from scipy.stats import ttest_1samp

# =========================
# EXTRA CHECK (requested): mean r2_corr > 0 for normal-RT in MOs
# =========================
def mean_gt0_ttest(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"n": int(v.size), "mean": np.nan, "t": np.nan, "p_one_sided": np.nan}
    m = float(np.mean(v))
    t, p_two = ttest_1samp(v, popmean=0.0, nan_policy="omit")
    # one-sided p for H1: mean > 0
    if np.isnan(t) or np.isnan(p_two):
        p_one = np.nan
    else:
        p_one = (p_two / 2.0) if m > 0 else (1.0 - p_two / 2.0)
    return {"n": int(v.size), "mean": m, "t": float(t), "p_one_sided": float(p_one)}

# Filter to MOs, normal
sub_mos_norm = df[(df["roi"] == "MOp") & (df["group"] == "normal")]
vals = pd.to_numeric(sub_mos_norm["r2_corr"], errors="coerce").to_numpy()

res = mean_gt0_ttest(vals)

print("\n[CHECK] Mean corrected R2 > 0 for normal-RT in MOs (one-sample t-test, one-sided)")
print(f"n={res['n']}  mean={res['mean']:.6f}  t={res['t']:.4f}  p(one-sided)={res['p_one_sided']:.6g}")

# Save result
CHECK_CSV = OUT_DIR / f"mean_r2_corr_gt0_check_MOs_normal_{TAG}.csv"
pd.DataFrame([{
    "roi": "MOp",
    "group": "normal",
    "n_sessions": res["n"],
    "mean_corrected_R2": res["mean"],
    "t_stat": res["t"],
    "p_mean_corrected_R2_gt0_one_sided": res["p_one_sided"],
}]).to_csv(CHECK_CSV, index=False)
print("Saved:", CHECK_CSV)