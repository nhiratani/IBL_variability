
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import os


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


PEARSON_SUMMARY_PKL = Path(
    "prior_localization_sessionfit_output_whole_session_behav_sessions/"
    "whole_session_summary_WHOLE_SESSION_BEHAV_SESSIONS_UPDATED_ROIS_ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_"
    "SSp-tr_SSp-bfd_SSp-ll_SSp-ul_SSp-n_SSs_VISp_VISpl_AUDp.pkl"
)

ATI_CSV = Path(
    "prior_localization_sessionfit_output_behav_sessions/"
    "ati_outputs_all_sessions/"
    "ATI_session_level.csv"
)


OUT_DIR = Path(
    "prior_localization_sessionfit_output_behav_sessions/"
    "plots_metric_vs_ATI_ggplot_pdf_residual"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI_LIST = [
    "ORBvl", "PL", "ACAd", "TEa", "VISC", "AId", "AIv", "MOs", "MOp",
    "SSp-tr", "SSp-bfd", "SSp-ll", "SSp-ul", "SSp-n",
    "SSs", "VISp", "VISpl", "AUDp",
]
ROI_SET = set(ROI_LIST)

# Try to use animal-level ATI if present; otherwise fall back to session-level ATI
ATI_COL_CANDIDATES = ["ATI_animal", "ATI_mouse", "ATI_subject", "ATI_session"]
X_COL_FALLBACK = "ATI_session"

N_COLS = 3
MIN_N_FOR_SUBPLOT = 3  # show ROI only if it has >=3 finite points for this metric in that plot
MIN_N_FOR_CORR = 3     # p-value requires >=3

# ONLY draw overall regression line if p < P_ALPHA
P_ALPHA = float(os.getenv("P_ALPHA", "0.05"))


def fmt3(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.3f}"

def safe_pearsonr(x, y):
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < MIN_N_FOR_CORR:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan

def safe_linregress(x, y):
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return None
    try:
        return linregress(x, y)
    except Exception:
        return None

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_all_rois_ggplot(
    df: pd.DataFrame,
    rois: list[str],
    roi_to_color: dict[str, tuple],
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    outpath_pdf: Path,
    add_overall_fit: bool = True,
    add_r_text: bool = True,
    p_alpha: float = P_ALPHA,
):
    """
    All-ROIs plot:
      - scatter by ROI
      - draw ONE overall regression line ONLY if p < p_alpha
      - optional ONE r/p text
      - NO formula explanations in title
    """
    fig = plt.figure(figsize=(7.8, 6.2))
    ax = plt.gca()

    for roi in rois:
        sub = df[df["roi"] == roi]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=35,
            alpha=0.8,
            label=roi,
            color=roi_to_color[roi],
        )

    # compute overall r,p once
    r, p = safe_pearsonr(df[x_col], df[y_col])

    # ONLY draw overall fit if significant
    if add_overall_fit and np.isfinite(p) and (p < float(p_alpha)):
        lr_all = safe_linregress(df[x_col], df[y_col])
        if lr_all is not None and np.isfinite(lr_all.slope) and np.isfinite(lr_all.intercept):
            xs = np.linspace(np.nanmin(df[x_col]), np.nanmax(df[x_col]), 250)
            ys = lr_all.slope * xs + lr_all.intercept
            ax.plot(xs, ys, color="k", linewidth=1.4, alpha=0.9)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    ax.legend(frameon=False, ncol=1)

    if add_r_text:
        ax.text(
            0.98, 0.02,
            f"r = {fmt3(r)}\np = {fmt3(p)}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=12,
            color="black",
        )

    fig.tight_layout()
    fig.savefig(outpath_pdf, dpi=300)
    plt.close(fig)

def plot_by_roi_subplots_ggplot(
    df: pd.DataFrame,
    rois: list[str],
    roi_to_color: dict[str, tuple],
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    outpath_pdf: Path,
):

    nrows = int(np.ceil(len(rois) / N_COLS))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=N_COLS,
        figsize=(4.6 * N_COLS, 4.0 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    for i, roi in enumerate(rois):
        ax = axes[i]
        sub = df[df["roi"] == roi]

        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=35,
            alpha=0.85,
            color=roi_to_color[roi],
        )

        # p-value only (no r)
        _, p = safe_pearsonr(sub[x_col], sub[y_col])
        ax.set_title(f"{roi} (n={len(sub)})\np={fmt3(p)}", fontsize=11)

        if i % N_COLS == 0:
            ax.set_ylabel(y_label)
        if i >= (nrows - 1) * N_COLS:
            ax.set_xlabel(x_label)

    for j in range(len(rois), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{y_label} vs {x_label} (by ROI)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath_pdf, dpi=300)
    plt.close(fig)

# ============================================================
# LOAD + MERGE
# ============================================================
if not PEARSON_SUMMARY_PKL.exists():
    raise FileNotFoundError(f"Pearson summary PKL not found: {PEARSON_SUMMARY_PKL.resolve()}")
if not ATI_CSV.exists():
    raise FileNotFoundError(f"ATI CSV not found: {ATI_CSV.resolve()}")

with open(PEARSON_SUMMARY_PKL, "rb") as f:
    rows = pickle.load(f)

df_dec = pd.DataFrame(rows)
df_ati = pd.read_csv(ATI_CSV)

# Normalize ids
if "eid" in df_dec.columns:
    df_dec["eid"] = df_dec["eid"].astype(str)
elif "session_id" in df_dec.columns:
    df_dec["eid"] = df_dec["session_id"].astype(str)
else:
    raise RuntimeError("Summary PKL missing 'eid' or 'session_id' column.")

df_ati["session_id"] = df_ati["session_id"].astype(str)

# Pick ATI X column
X_COL = pick_first_existing(df_ati, ATI_COL_CANDIDATES) or X_COL_FALLBACK
if X_COL not in df_ati.columns:
    raise RuntimeError(f"ATI CSV missing any of {ATI_COL_CANDIDATES} (and also missing '{X_COL_FALLBACK}').")

X_LABEL = "Animal-level ATI" if X_COL in ["ATI_animal", "ATI_mouse", "ATI_subject"] else "Session-level ATI"

# ROI column normalization
roi_col = pick_first_existing(df_dec, ["roi", "region", "acronym"])
if roi_col is None:
    raise RuntimeError("Summary PKL missing ROI column (expected one of: roi / region / acronym).")
if roi_col != "roi":
    df_dec = df_dec.rename(columns={roi_col: "roi"})
df_dec["roi"] = df_dec["roi"].astype(str)

# Keep only requested ROIs
df_dec = df_dec[df_dec["roi"].isin(ROI_SET)].copy()

# Merge with ATI
df = df_dec.merge(
    df_ati[["session_id", X_COL]],
    left_on="eid",
    right_on="session_id",
    how="inner",
)

df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")


if "z_corr" in df.columns:
    df["z_corr"] = pd.to_numeric(df["z_corr"], errors="coerce")
else:
    alt = pick_first_existing(df, ["z", "z_score", "zscore", "z_corr_mean"])
    df["z_corr"] = pd.to_numeric(df[alt], errors="coerce") if alt else np.nan

# corrected Pearson r: r_corr = r_real - r_fake_mean
if "r_real" in df.columns and "r_fake_mean" in df.columns:
    df["r_real"] = pd.to_numeric(df["r_real"], errors="coerce")
    df["r_fake_mean"] = pd.to_numeric(df["r_fake_mean"], errors="coerce")
    df["r_corr"] = df["r_real"] - df["r_fake_mean"]
else:
    alt = pick_first_existing(df, ["r_corr", "r_corr_mean", "pearson_r", "rho", "r"])
    df["r_corr"] = pd.to_numeric(df[alt], errors="coerce") if alt else np.nan

# corrected R2
if "r2_corr" in df.columns:
    df["r2_corr"] = pd.to_numeric(df["r2_corr"], errors="coerce")
else:
    if "r2_real" in df.columns and "r2_fake_mean" in df.columns:
        r2_real = pd.to_numeric(df["r2_real"], errors="coerce")
        r2_fake_mean = pd.to_numeric(df["r2_fake_mean"], errors="coerce")
        denom = (1.0 - r2_fake_mean)
        df["r2_corr"] = (r2_real - r2_fake_mean) / denom
    else:
        alt = pick_first_existing(df, ["r2", "r2_session", "r2_corr_mean"])
        df["r2_corr"] = pd.to_numeric(df[alt], errors="coerce") if alt else np.nan


for base_col in ["r2_corr", "r_corr", "z_corr"]:
    mean_col = f"{base_col}_roi_mean"
    resid_col = f"{base_col}_resid"
    df[mean_col] = df.groupby("roi")[base_col].transform(lambda s: np.nanmean(pd.to_numeric(s, errors="coerce")))
    df[resid_col] = pd.to_numeric(df[base_col], errors="coerce") - pd.to_numeric(df[mean_col], errors="coerce")


present_rois = [r for r in ROI_LIST if r in set(df["roi"].dropna().unique())]
if len(present_rois) == 0:
    raise RuntimeError("After merge/filtering, no ROI rows remain. Check PKL + ATI CSV IDs/ROIs.")

colors = plt.cm.tab20.colors
roi_to_color = {roi: colors[i % len(colors)] for i, roi in enumerate(present_rois)}

print("ROIs present:", present_rois)
print("N merged rows (session x roi):", len(df))
print("Using X_COL:", X_COL)
print("Fit-line threshold P_ALPHA:", P_ALPHA)


METRICS = [
    ("r2_corr", "Corrected R2", "r2corr"),
    ("r_corr", "Corrected Pearson r", "rcorr"),
    ("z_corr", "Z-score", "zcorr"),
]

RESID_LABELS = {
    "r2_corr": "Residual corrected R2",
    "r_corr": "Residual corrected Pearson r",
    "z_corr": "Residual Z-score",
}


for base_col, base_label, tag in METRICS:
    if base_col not in df.columns:
        print(f"[WARN] metric '{base_col}' not in df; skipping")
        continue


    y_num = pd.to_numeric(df[base_col], errors="coerce")
    dsub = df[np.isfinite(df[X_COL]) & np.isfinite(y_num)].copy()
    dsub[base_col] = pd.to_numeric(dsub[base_col], errors="coerce")

    if len(dsub) == 0:
        print(f"[WARN] no finite rows for {base_col}; skipping")
        continue

    counts_sub = dsub.groupby("roi")[base_col].apply(lambda s: np.isfinite(pd.to_numeric(s, errors="coerce")).sum())
    rois_ge3_sub = [roi for roi in present_rois if int(counts_sub.get(roi, 0)) >= MIN_N_FOR_SUBPLOT]

    if len(rois_ge3_sub) == 0:
        print(f"[WARN] no ROIs with >= {MIN_N_FOR_SUBPLOT} points for subgroup {base_col}; skipping subgroup plot")
    else:
        dsub_plot = dsub[dsub["roi"].isin(rois_ge3_sub)].copy()
        out_sub = OUT_DIR / f"{tag}_vs_ATI_by_roi.pdf"
        plot_by_roi_subplots_ggplot(
            dsub_plot,
            rois_ge3_sub,
            roi_to_color,
            x_col=X_COL,
            x_label=X_LABEL,
            y_col=base_col,
            y_label=base_label,
            outpath_pdf=out_sub,
        )
        print("[SAVED]", out_sub)


    resid_col = f"{base_col}_resid"
    if resid_col not in df.columns:
        print(f"[WARN] residual column '{resid_col}' missing; skipping all-ROIs residual plot for {base_col}")
        continue

    y_res = pd.to_numeric(df[resid_col], errors="coerce")
    dall = df[np.isfinite(df[X_COL]) & np.isfinite(y_res)].copy()
    dall[resid_col] = pd.to_numeric(dall[resid_col], errors="coerce")

    if len(dall) == 0:
        print(f"[WARN] no finite rows for residual {resid_col}; skipping all-ROIs plot")
        continue

    counts_all = dall.groupby("roi")[resid_col].apply(lambda s: np.isfinite(pd.to_numeric(s, errors="coerce")).sum())
    rois_ge3_all = [roi for roi in present_rois if int(counts_all.get(roi, 0)) >= MIN_N_FOR_SUBPLOT]
    if len(rois_ge3_all) == 0:
        print(f"[WARN] no ROIs with >= {MIN_N_FOR_SUBPLOT} points for all-ROIs residual {resid_col}; skipping all-ROIs plot")
        continue

    dall_plot = dall[dall["roi"].isin(rois_ge3_all)].copy()
    out_all = OUT_DIR / f"{tag}_vs_ATI_all_rois_RESIDUAL.pdf"
    plot_all_rois_ggplot(
        dall_plot,
        rois_ge3_all,
        roi_to_color,
        x_col=X_COL,
        x_label=X_LABEL,
        y_col=resid_col,
        y_label=RESID_LABELS.get(base_col, f"Residual {base_label}"),
        outpath_pdf=out_all,
        add_overall_fit=True,
        add_r_text=True,
        p_alpha=P_ALPHA,
    )
    print("[SAVED]", out_all)

print("\nAll outputs in:", OUT_DIR.resolve())