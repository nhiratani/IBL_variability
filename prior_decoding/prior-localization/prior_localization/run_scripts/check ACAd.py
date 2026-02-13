

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

WHOLE_SESSION_PKL = Path(
    "prior_localization_sessionfit_output_whole_session/"
    "whole_session_summary_VISa_VISp_ORBvl_PL_MOs_MOp_ACAd.pkl"
)

PEARSON_SUMMARY_PKL = Path(
    "prior_localization_sessionfit_output_behav_sessions/"
    "pearson_summary_BEHAV_SESSIONS_UPDATED_ROIS_ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_"
    "SSp-tr_SSp-bfd_SSp-ll_SSp-ul_SSp-n_SSs_VISp_VISpl_AUDp.pkl"
)

def load_acad_r2(pkl_path: Path) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        rows = pickle.load(f)
    df = pd.DataFrame(rows)

    # normalize eid
    if "eid" not in df.columns and "session_id" in df.columns:
        df = df.rename(columns={"session_id": "eid"})
    df["eid"] = df["eid"].astype(str)

    # ACAd + finite corrected R2
    sub = df[df["roi"] == "ACAd"].copy()
    sub["r2_corr"] = pd.to_numeric(sub["r2_corr"], errors="coerce")
    sub = sub[np.isfinite(sub["r2_corr"])]

    # ONE value per session (important)
    sub = (
        sub.groupby("eid", as_index=False)
           .agg(r2_corr=("r2_corr", "mean"))
    )

    return sub


acad_whole = load_acad_r2(WHOLE_SESSION_PKL)
acad_behav = load_acad_r2(PEARSON_SUMMARY_PKL)

eids_whole = set(acad_whole["eid"])
eids_behav = set(acad_behav["eid"])


if len(eids_whole) <= len(eids_behav):
    smaller_name = "WHOLE"
    missing_from_smaller = sorted(eids_behav - eids_whole)
else:
    smaller_name = "BEHAV"
    missing_from_smaller = sorted(eids_whole - eids_behav)

print(f"\n==== Session IDs present in the larger set but MISSING from {smaller_name} ====")
for eid in missing_from_smaller:
    print(eid)


merged = acad_whole.merge(
    acad_behav,
    on="eid",
    how="inner",
    suffixes=("_whole", "_behav")
)

merged["diff_behav_minus_whole"] = (
    merged["r2_corr_behav"] - merged["r2_corr_whole"]
)

print("\n==== Shared ACAd sessions: corrected R2 comparison ====")
print(
    merged[
        ["eid", "r2_corr_whole", "r2_corr_behav", "diff_behav_minus_whole"]
    ].sort_values(
        "diff_behav_minus_whole",
        key=lambda x: np.abs(x),
        ascending=False
    ).to_string(index=False)
)

# Optional save
merged.to_csv("ACAd_corrected_R2_shared_sessions.csv", index=False)

print("\nSaved shared-session comparison to ACAd_corrected_R2_shared_sessions.csv")

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

PKL = Path("prior_localization_sessionfit_output/pearson_summary_ACAd_2.pkl")

with open(PKL, "rb") as f:
    rows = pickle.load(f)

df = pd.DataFrame(rows)

print("==== RAW CONTENT ====")
print("Columns:", list(df.columns))
print("Total rows:", len(df))


if "eid" not in df.columns and "session_id" in df.columns:
    df = df.rename(columns={"session_id": "eid"})
df["eid"] = df["eid"].astype(str)
df["roi"] = df["roi"].astype(str)


for c in ["r2_real", "r2_fake_mean", "r2_corr"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


print("\n==== Corrected R² (per row) ====")
print(
    df[
        ["eid", "roi", "group", "r2_real", "r2_fake_mean", "r2_corr"]
    ].to_string(index=False)
)


df_session = (
    df.groupby(["eid", "roi"], as_index=False)
      .agg(
          r2_real=("r2_real", "mean"),
          r2_fake_mean=("r2_fake_mean", "mean"),
          r2_corr=("r2_corr", "mean"),
      )
)

print("\n==== Per-session corrected R² (collapsed) ====")
print(df_session.to_string(index=False))