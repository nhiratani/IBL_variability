
from __future__ import annotations

import pickle
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE

from prior_localization.my_rt import (
    load_wheel_data, calc_wheel_velocity, calc_trialwise_wheel, calc_movement_onset_times
)

warnings.filterwarnings("ignore")

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


SUBGROUP_SUMMARY_DIR = Path(
    "/Users/changyin/Documents/prior-localization/prior_localization/run_scripts/prior_localization_sessionfit_output"
)
WHOLE_SESSION_SUMMARY_DIR = Path(
    "/Users/changyin/Documents/prior-localization/prior_localization/run_scripts/prior_localization_sessionfit_output_whole_session"
)

SUBGROUP_SUMMARY_FILES = [
    SUBGROUP_SUMMARY_DIR / "pearson_summary_MOs_ACAd_MOp_ORBvl.pkl",
    SUBGROUP_SUMMARY_DIR / "pearson_summary_VISa_VISp_PL.pkl",
]

WHOLE_SESSION_SUMMARY_FILE = WHOLE_SESSION_SUMMARY_DIR / "whole_session_summary_VISa_VISp_ORBvl_PL_MOs_MOp_ACAd.pkl"


ROIS = ["MOs", "MOp", "VISa"]


OUT_BIG = Path(
    "/Users/changyin/Documents/prior-localization/prior_localization/run_scripts/decoded_vs_target_bigfolder"
)

FAST_THR = 0.08
SLOW_THR = 1.25

RUN_INDEX = 0
TOP_N = 10

RANK_METRIC = "r2_corr"



def _trial_scalar_list(x_list) -> np.ndarray:
    """list-of-arrays per trial -> scalar per trial (mean), like your metric code."""
    out = np.full(len(x_list), np.nan, float)
    for i, xi in enumerate(x_list):
        if xi is None:
            continue
        arr = np.asarray(xi).reshape(-1)
        if arr.size == 0 or (not np.all(np.isfinite(arr))):
            continue
        out[i] = float(np.mean(arr))
    return out


def load_roi_fit_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_pred_target_for_real_session(d: dict, run_index: int = 0):

    keep_idx_full = np.asarray(d.get("keep_idx_full", []), dtype=int)

    fits = [fr for fr in d.get("fit", []) if int(fr.get("pseudo_id", -999)) == -1]
    if len(fits) == 0:
        raise RuntimeError("No pseudo_id == -1 found in fit list.")

    if run_index >= len(fits):
        raise RuntimeError(
            f"run_index={run_index} but only {len(fits)} run(s) exist for pseudo_id=-1."
        )

    fr = fits[run_index]
    preds = fr.get("predictions_test", None)
    targ = fr.get("target", None)
    if preds is None or targ is None:
        raise RuntimeError("Missing predictions_test or target in fit record.")

    yhat = _trial_scalar_list(preds)
    y = _trial_scalar_list(targ)

    return keep_idx_full, yhat, y


def compute_rt_masks(one: ONE, eid: str, fast_thr: float, slow_thr: float) -> dict[str, np.ndarray]:

    trials = one.load_object(eid, "trials", collection="alf")
    stim = np.asarray(trials["stimOn_times"])
    fb = np.asarray(trials["feedback_times"])
    n_trials = len(stim)

    pos, ts = load_wheel_data(one, eid)
    if pos is None or ts is None:
        raise RuntimeError(f"No wheel data for eid={eid}")

    vel = calc_wheel_velocity(pos, ts)
    _, trial_ts, trial_vel = calc_trialwise_wheel(pos, ts, vel, stim, fb)
    (_, first_mo, _, _, _) = calc_movement_onset_times(trial_ts, trial_vel, stim)

    rt = first_mo - stim
    valid = np.isfinite(rt)

    fast = valid & (rt < fast_thr)
    slow = valid & (rt > slow_thr)
    normal = valid & (~fast) & (~slow)

    assert fast.shape == (n_trials,)
    assert normal.shape == (n_trials,)
    assert slow.shape == (n_trials,)

    return {"fast": fast, "normal": normal, "slow": slow}


def save_plot_pdf(out_pdf: Path, yhat: np.ndarray, y: np.ndarray, title: str):

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4.2))
    ax = plt.gca()

    ax.plot(yhat, linewidth=1.8, alpha=0.95, label="decoded prior")
    ax.plot(y, linewidth=1.8, alpha=0.95, label="target")

    ax.set_xlabel("Decoded trial index")
    ax.set_ylabel("pLeft")
    ax.set_title(title)

    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)


def load_summary_rows(pkl_path: Path) -> list[dict]:
    with open(pkl_path, "rb") as f:
        rows = pickle.load(f)
    return rows


def pick_top_rows(
    rows: list[dict],
    roi: str,
    group: str | None,
    metric: str,
    top_n: int | None
):

    rr = []
    for r in rows:
        if str(r.get("roi")) != roi:
            continue
        if group is not None and r.get("group") != group:
            continue
        if r.get("status") != "ok":
            continue
        p = r.get("roi_pkl_path")
        if not p:
            continue
        rr.append(r)

    def keyfun(r):
        v = r.get(metric, np.nan)
        return -v if np.isfinite(v) else np.inf  # NaNs go last

    rr_sorted = sorted(rr, key=keyfun)
    return rr_sorted if top_n is None else rr_sorted[:top_n]




def main():
    OUT_SUB = OUT_BIG / "subgroup"
    OUT_WHOLE = OUT_BIG / "whole_session"
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    OUT_WHOLE.mkdir(parents=True, exist_ok=True)


    subgroup_rows = []
    for p in SUBGROUP_SUMMARY_FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing subgroup summary PKL: {p}")
        subgroup_rows.extend(load_summary_rows(p))

    if not WHOLE_SESSION_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing whole-session summary PKL: {WHOLE_SESSION_SUMMARY_FILE}")
    whole_rows = load_summary_rows(WHOLE_SESSION_SUMMARY_FILE)

    for roi in ROIS:
        roi_out = OUT_WHOLE / roi
        picked = pick_top_rows(whole_rows, roi=roi, group="session", metric=RANK_METRIC, top_n=TOP_N)

        for r in picked:
            eid = r["eid"]
            roi_pkl_path = Path(r["roi_pkl_path"])

            try:
                d = load_roi_fit_pkl(roi_pkl_path)
                keep_idx_full, yhat, y = extract_pred_target_for_real_session(d, run_index=RUN_INDEX)

                title = f"Whole-session trace | ROI={roi} | eid={eid} "
                out_pdf = roi_out / f"{eid}_run{RUN_INDEX}_decoded_vs_target.pdf"
                save_plot_pdf(out_pdf, yhat, y, title)

            except Exception as e:
                print("[WHOLE][SKIP]", roi, eid, "->", type(e).__name__, e)

    one = ONE()  # uses your local ONE config / cache

    for roi in ROIS:
        for group in ["fast", "normal", "slow"]:
            group_out = OUT_SUB / roi / group
            picked = pick_top_rows(subgroup_rows, roi=roi, group=group, metric=RANK_METRIC, top_n=TOP_N)

            for r in picked:
                eid = r["eid"]
                roi_pkl_path = Path(r["roi_pkl_path"])

                try:
                    d = load_roi_fit_pkl(roi_pkl_path)
                    keep_idx_full, yhat, y = extract_pred_target_for_real_session(d, run_index=RUN_INDEX)

                    masks_full = compute_rt_masks(one, eid, fast_thr=FAST_THR, slow_thr=SLOW_THR)

                    mask_sub = masks_full[group][keep_idx_full]
                    sub_idx = np.flatnonzero(mask_sub)

                    yhat_g = yhat[sub_idx]
                    y_g = y[sub_idx]

                    title = f"Subgroup={group} | ROI={roi} | eid={eid} | n={len(sub_idx)} "
                    out_pdf = group_out / f"{eid}_run{RUN_INDEX}_decoded_vs_target.pdf"
                    save_plot_pdf(out_pdf, yhat_g, y_g, title)

                except Exception as e:
                    print("[SUBGROUP][SKIP]", roi, group, eid, "->", type(e).__name__, e)

    print("\nDone.")
    print("Saved to:", OUT_BIG)


if __name__ == "__main__":
    main()