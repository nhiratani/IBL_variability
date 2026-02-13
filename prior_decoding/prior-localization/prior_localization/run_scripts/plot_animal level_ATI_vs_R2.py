
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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
    "./prior_localization_sessionfit_output_behav_sessions/"
    "plots_ANIMAL_level_ATI_vs_metrics_residualized_by_roi_MATCHED_SESSIONS_TRIALSUM"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ATI_SESSION_COL = "ATI_session"
X_COL = "ATI_animal"
X_LABEL = "Animal-level ATI"


ATI_COUNT_COLS = ["n_total", "n_impulsive", "n_slow"]

METRICS = [
    ("r2_corr", "Corrected R2", "r2corr"),
    ("r_corr", "Corrected Pearson r", "rcorr"),
    ("z_corr", "Z-score", "zcorr"),
]

MIN_N_FOR_CORR = 3


N_PERM = 10000
PERM_SEED = 0
PERM_TWO_SIDED = True


SEX_COLOR_MAP = {"M": "C1", "F": "C0", "UNKNOWN": "0.35"}
SEX_LABEL_MAP = {"M": "Male", "F": "Female", "UNKNOWN": "Unknown"}


def fmt4(x) -> str:
    try:
        x = float(x)
    except Exception:
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.4f}"

def _safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def normalize_sex(val) -> str:
    if val is None:
        return "UNKNOWN"
    s = str(val).strip()
    if not s:
        return "UNKNOWN"
    s_up = s.upper()
    if s_up in ("M", "MALE") or s_up.startswith("M"):
        return "M"
    if s_up in ("F", "FEMALE") or s_up.startswith("F"):
        return "F"
    return "UNKNOWN"

def permutation_p_for_corr(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = PERM_SEED,
    two_sided: bool = True,
):

    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    n = len(x)
    if n < MIN_N_FOR_CORR:
        return np.nan, np.nan, None

    # observed r
    try:
        r_obs, _ = pearsonr(x, y)
        r_obs = float(r_obs)
    except Exception:
        return np.nan, np.nan, None

    rng = np.random.default_rng(seed)

    r_perm = np.empty(n_perm, dtype=float)
    y_work = y.copy()

    for i in range(n_perm):
        rng.shuffle(y_work)  # in-place
        try:
            r_i, _ = pearsonr(x, y_work)
        except Exception:
            r_i = np.nan
        r_perm[i] = r_i

    r_perm = r_perm[np.isfinite(r_perm)]
    if r_perm.size == 0:
        return r_obs, np.nan, None

    if two_sided:
        extreme = np.sum(np.abs(r_perm) >= abs(r_obs))
    else:

        extreme = np.sum(r_perm >= r_obs)

    p_perm = (1.0 + extreme) / (1.0 + r_perm.size)
    return r_obs, float(p_perm), r_perm

def safe_pearsonr_with_perm(x, y):

    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < MIN_N_FOR_CORR:
        return np.nan, np.nan, np.nan, np.nan

    try:
        r_param, p_param = pearsonr(x, y)   # two-sided parametric p
        r_param = float(r_param)
        p_param = float(p_param)
    except Exception:
        r_param, p_param = np.nan, np.nan

    r_perm, p_perm, _ = permutation_p_for_corr(
        x, y, n_perm=N_PERM, seed=PERM_SEED, two_sided=PERM_TWO_SIDED
    )

    return r_param, p_param, r_perm, p_perm

def plot_animal_level_pdf(
    df_animal: pd.DataFrame,
    y_col: str,
    y_label: str,
    outpath: Path,
    has_sex: bool = True,
):
    d = df_animal[np.isfinite(df_animal[X_COL]) & np.isfinite(df_animal[y_col])].copy()

    r_param, p_param, r_perm, p_perm = safe_pearsonr_with_perm(d[X_COL], d[y_col])

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = plt.gca()

    if has_sex and "sex" in d.columns:
        d["sex_norm"] = d["sex"].map(normalize_sex)
        for sex_key in ("M", "F", "UNKNOWN"):
            dd = d[d["sex_norm"] == sex_key]
            if len(dd) == 0:
                continue
            ax.scatter(
                dd[X_COL],
                dd[y_col],
                s=46,
                alpha=0.85,
                color=SEX_COLOR_MAP.get(sex_key, "0.35"),
                label=SEX_LABEL_MAP.get(sex_key, sex_key),
            )
        ax.legend(frameon=True, loc="best")
    else:
        ax.scatter(d[X_COL], d[y_col], s=46, alpha=0.85, color="C0")

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(y_label)

    ax.text(
        0.02,
        0.98,
        "Pearson:\n"
        f"r = {fmt4(r_param)}\n"
        f"p = {fmt4(p_param)}\n\n"
        f"Perm (N={N_PERM}):\n"
        f"r = {fmt4(r_perm)}\n"
        f"p = {fmt4(p_perm)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="black",
    )

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


if not PEARSON_SUMMARY_PKL.exists():
    raise FileNotFoundError(f"Decoding summary PKL not found: {PEARSON_SUMMARY_PKL.resolve()}")

with open(PEARSON_SUMMARY_PKL, "rb") as f:
    rows = pickle.load(f)

df_dec = pd.DataFrame(rows)

if "eid" not in df_dec.columns:
    raise RuntimeError("Decoding summary PKL must contain column 'eid'")
df_dec["eid"] = df_dec["eid"].astype(str)

# ensure roi exists
if "roi" not in df_dec.columns:
    if "region" in df_dec.columns:
        df_dec = df_dec.rename(columns={"region": "roi"})
    elif "acronym" in df_dec.columns:
        df_dec = df_dec.rename(columns={"acronym": "roi"})
    else:
        df_dec["roi"] = "UNKNOWN_ROI"
df_dec["roi"] = df_dec["roi"].astype(str)


if "z_corr" in df_dec.columns:
    df_dec["z_corr"] = _safe_numeric(df_dec["z_corr"])
else:
    alt = next((c for c in ["z", "z_score", "zscore"] if c in df_dec.columns), None)
    df_dec["z_corr"] = _safe_numeric(df_dec[alt]) if alt else np.nan

if "r_corr" in df_dec.columns:
    df_dec["r_corr"] = _safe_numeric(df_dec["r_corr"])
else:
    if "r_real" in df_dec.columns and "r_fake_mean" in df_dec.columns:
        df_dec["r_real"] = _safe_numeric(df_dec["r_real"])
        df_dec["r_fake_mean"] = _safe_numeric(df_dec["r_fake_mean"])
        df_dec["r_corr"] = df_dec["r_real"] - df_dec["r_fake_mean"]
    else:
        df_dec["r_corr"] = _safe_numeric(df_dec["r_real"]) if "r_real" in df_dec.columns else np.nan

if "r2_corr" in df_dec.columns:
    df_dec["r2_corr"] = _safe_numeric(df_dec["r2_corr"])
else:
    if "r2_real" in df_dec.columns and "r2_fake_mean" in df_dec.columns:
        r2_real = _safe_numeric(df_dec["r2_real"])
        r2_fake_mean = _safe_numeric(df_dec["r2_fake_mean"])
        df_dec["r2_corr"] = (r2_real - r2_fake_mean) / (1.0 - r2_fake_mean)
    else:
        df_dec["r2_corr"] = _safe_numeric(df_dec["r2_real"]) if "r2_real" in df_dec.columns else np.nan

included_eids = set(df_dec["eid"].dropna().astype(str).unique())
print(f"[MATCH] decoding summary defines {len(included_eids)} unique eids (sessions)")


if not ATI_CSV.exists():
    raise FileNotFoundError(f"ATI session-level CSV not found: {ATI_CSV.resolve()}")

df_ati = pd.read_csv(ATI_CSV)

required_cols = {"session_id", "subject"} | set(ATI_COUNT_COLS)
missing = [c for c in required_cols if c not in df_ati.columns]
if missing:
    raise RuntimeError(
        f"ATI_session_level.csv must contain columns: {sorted(required_cols)}\n"
        f"Missing: {missing}\n"
        f"File: {ATI_CSV}"
    )

df_ati["session_id"] = df_ati["session_id"].astype(str)
df_ati["subject"] = df_ati["subject"].astype(str)
for c in ATI_COUNT_COLS:
    df_ati[c] = _safe_numeric(df_ati[c])

has_sex = "sex" in df_ati.columns
if has_sex:
    df_ati["sex"] = df_ati["sex"].astype(str)


df_ati_matched = df_ati[df_ati["session_id"].isin(included_eids)].copy()

if "status" in df_ati_matched.columns:
    df_ati_matched = df_ati_matched[df_ati_matched["status"].astype(str).str.lower() == "ok"].copy()

# Audit
audit = {
    "ati_rows_total": len(df_ati),
    "ati_rows_matched": len(df_ati_matched),
    "ati_unique_sessions_total": int(df_ati["session_id"].nunique()),
    "ati_unique_sessions_matched": int(df_ati_matched["session_id"].nunique()),
    "ati_unique_animals_total": int(df_ati["subject"].nunique()),
    "ati_unique_animals_matched": int(df_ati_matched["subject"].nunique()),
}
audit_path = OUT_DIR / "ati_matching_audit.csv"
pd.DataFrame([audit]).to_csv(audit_path, index=False)
print("[SAVED]", audit_path)


agg_dict = {
    "n_total_trials": ("n_total", "sum"),
    "n_impulsive": ("n_impulsive", "sum"),
    "n_slow": ("n_slow", "sum"),
    "n_sessions_ati": ("session_id", lambda x: len(pd.unique(x))),
}
if has_sex:
    agg_dict["sex"] = ("sex", "first")

df_ati_animal = (
    df_ati_matched.groupby("subject", as_index=False)
                  .agg(**agg_dict)
)

df_ati_animal["ATI_animal"] = (
    (df_ati_animal["n_impulsive"] - df_ati_animal["n_slow"]) /
    df_ati_animal["n_total_trials"].replace(0, np.nan)
)


if "subject" not in df_dec.columns or df_dec["subject"].isna().all():
    df_dec = df_dec.merge(
        df_ati_matched[["session_id", "subject"]].drop_duplicates(),
        left_on="eid",
        right_on="session_id",
        how="left",
    )
    df_dec.drop(columns=["session_id"], inplace=True, errors="ignore")

if "subject" not in df_dec.columns:
    raise RuntimeError("Could not obtain 'subject' for decoding summary rows.")
df_dec["subject"] = df_dec["subject"].astype(str)


metric_cols = [m[0] for m in METRICS]
use_cols = ["subject", "eid", "roi"] + metric_cols
for c in use_cols:
    if c not in df_dec.columns:
        df_dec[c] = np.nan

df_use = df_dec[use_cols].copy()
for y_col in metric_cols:
    df_use[y_col] = _safe_numeric(df_use[y_col])

roi_means = (
    df_use.groupby("roi", as_index=False)
          .agg(**{f"{y}_roi_mean": (y, "mean") for y in metric_cols})
)

df_use = df_use.merge(roi_means, on="roi", how="left")

for y in metric_cols:
    df_use[f"{y}_resid"] = df_use[y] - df_use[f"{y}_roi_mean"]

df_an_roi = (
    df_use.groupby(["subject", "roi"], as_index=False)
          .agg(
              **{f"{y}_resid": (f"{y}_resid", "mean") for y in metric_cols},
              n_rows=("eid", "count"),
              n_sessions=("eid", lambda x: len(pd.unique(x))),
          )
)

df_an_metrics = (
    df_an_roi.groupby("subject", as_index=False)
             .agg(
                 **{y: (f"{y}_resid", "mean") for y in metric_cols},
                 n_rois=("roi", "nunique"),
             )
).rename(columns={y: f"{y}_resid_animal" for y in metric_cols})


merge_cols = ["subject", "ATI_animal", "n_sessions_ati", "n_total_trials", "n_impulsive", "n_slow"] + (
    ["sex"] if has_sex else []
)
df_animal = df_an_metrics.merge(df_ati_animal[merge_cols], on="subject", how="inner")

out_table = OUT_DIR / "animal_level_metrics_ROI_residualized_MATCHED_TRIALSUM_ATI.csv"
df_animal.to_csv(out_table, index=False)

print("N animals (merged):", df_animal["subject"].nunique())
print("Saved merged animal table:", out_table)


plot_metrics = [
    ("r2_corr_resid_animal", "ROI-residualized Corrected R2", "r2corr_resid"),
    ("r_corr_resid_animal",  "ROI-residualized Corrected Pearson r", "rcorr_resid"),
    ("z_corr_resid_animal",  "ROI-residualized Z-score", "zcorr_resid"),
]

for y_col, y_label, tag in plot_metrics:
    outpath = OUT_DIR / f"{tag}_vs_animal_ATI_MATCHED_TRIALSUM_perm.pdf"
    plot_animal_level_pdf(
        df_animal,
        y_col=y_col,
        y_label=y_label,
        outpath=outpath,
        has_sex=has_sex,
    )
    print("[SAVED]", outpath)

print("All outputs in:", OUT_DIR.resolve())