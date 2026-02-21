from __future__ import annotations

from pathlib import Path
import json
import pickle
import warnings
import os
import numpy as np
import pandas as pd

from one.api import ONE
import one.alf.exceptions as alferr

from prior_localization.fit_data import fit_session_ephys


ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_CACHE_DIR = Path("/Volumes/T7 Shield/SSD_ONE/ONE")
USE_PER_PROCESS_TABLES = False

# Pick ONE session + optional ROI filter
SESSION_ID = "dfbe628d-365b-461c-a07f-8b9911ba83aa"
ROI_LIST = ["ACAd"]
ROI_SET = set(ROI_LIST)

# Output root for this single-session run
OUT_ROOT = Path("./prior_localization_single_session_output")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TAG = "SINGLE_SESSION"

# Decode controls
N_PSEUDO = int(os.getenv("N_PSEUDO", "200"))
N_RUNS = int(os.getenv("N_RUNS", "2"))
DEBUG = bool(int(os.getenv("DEBUG", "0")))

ALIGN_EVENT = "stimOn_times"
TIME_WINDOW = (-0.6, -0.1)

DROP_LAST_N = int(os.getenv("DROP_LAST_N", "40"))
MIN_RAW_TRIALS = int(os.getenv("MIN_RAW_TRIALS", "401"))

MIN_TRIALS_FOR_METRICS = int(os.getenv("MIN_TRIALS_FOR_METRICS", "10"))

def load_trials_df(one: ONE, eid: str) -> tuple[pd.DataFrame, int]:
    """Load ALF trials into a DataFrame (NO wheel/RT computation)."""
    trials_obj = one.load_object(eid, "trials", collection="alf")
    n_raw_trials = len(trials_obj["stimOn_times"])

    data = {}
    for k, v in trials_obj.items():
        arr = np.asarray(v)
        if k == "intervals" and arr.ndim == 2 and arr.shape[1] == 2:
            data["intervals_start"] = arr[:, 0]
            data["intervals_end"] = arr[:, 1]
        elif arr.ndim == 1:
            data[k] = arr
        else:
            data[k] = list(arr)

    df = pd.DataFrame(data)
    return df, int(n_raw_trials)


def make_drop_last_mask(n_trials: int, drop_last_n: int) -> np.ndarray:
    m = np.ones(int(n_trials), dtype=bool)
    if drop_last_n and drop_last_n > 0:
        m[max(0, n_trials - drop_last_n) :] = False
    return m

def _trial_scalar_list(x_list):

    out = np.full(len(x_list), np.nan, float)
    for i, xi in enumerate(x_list):
        if xi is None:
            continue
        arr = np.asarray(xi).reshape(-1)
        if arr.size == 0 or (not np.all(np.isfinite(arr))):
            continue
        out[i] = float(np.mean(arr))
    return out


def _pearsonr_safe(a, b):
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]
    if a.size < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _r2_safe(y_true, y_pred):
    y_true = np.asarray(y_true, float).reshape(-1)
    y_pred = np.asarray(y_pred, float).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if y_true.size < 3:
        return np.nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return np.nan
    sse = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - sse / denom)


def _z_and_p_from_null(real_val, null_vals, one_sided="greater_equal"):
    null_vals = np.asarray(null_vals, float)
    null_vals = null_vals[np.isfinite(null_vals)]
    if (not np.isfinite(real_val)) or null_vals.size < 5:
        return np.nan, np.nan, np.nan, np.nan

    mu = float(np.mean(null_vals))
    sd = float(np.std(null_vals, ddof=1)) if null_vals.size > 1 else float(np.std(null_vals))
    z = float((real_val - mu) / sd) if sd > 0 else np.nan

    if one_sided == "greater_equal":
        p = float((1 + np.sum(null_vals >= real_val)) / (1 + null_vals.size))
    elif one_sided == "less_equal":
        p = float((1 + np.sum(null_vals <= real_val)) / (1 + null_vals.size))
    else:
        raise ValueError("one_sided must be 'greater_equal' or 'less_equal'")

    return mu, sd, z, p


def _corrected_r2_findling(r2_real, r2_fake_mean):
    if (not np.isfinite(r2_real)) or (not np.isfinite(r2_fake_mean)):
        return np.nan
    denom = 1.0 - float(r2_fake_mean)
    if denom == 0:
        return np.nan
    return float((r2_real - float(r2_fake_mean)) / denom)


def find_roi_pkl(session_dir: Path, roi: str):

    for p in sorted(session_dir.rglob("*.pkl")):
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
            reg = d.get("region", "")
            reg = reg[0] if isinstance(reg, (list, tuple)) else str(reg)
            if str(reg).strip() == str(roi).strip():
                return p
        except Exception:
            continue
    return None


def compute_session_stats_from_pkl(region_pkl: Path):

    with open(region_pkl, "rb") as f:
        d = pickle.load(f)

    keep_idx_full = np.asarray(d.get("keep_idx_full", []), dtype=int)
    n_used = int(keep_idx_full.size)

    if n_used < int(MIN_TRIALS_FOR_METRICS):
        return {
            "n_used": n_used,
            "r_real": np.nan,
            "r_fake_mean": np.nan,
            "r_fake_std": np.nan,
            "z_corr": np.nan,
            "p_emp": np.nan,
            "r2_real": np.nan,
            "r2_fake_mean": np.nan,
            "r2_fake_std": np.nan,
            "z_r2": np.nan,
            "p_emp_r2": np.nan,
            "r2_corr": np.nan,
        }

    pid_to_r = {}
    pid_to_r2 = {}

    for fr in d.get("fit", []):
        pid = int(fr.get("pseudo_id", -999))
        preds = fr.get("predictions_test", None)
        targ = fr.get("target", None)
        if preds is None or targ is None:
            continue

        yhat = _trial_scalar_list(preds)
        y = _trial_scalar_list(targ)

        r = _pearsonr_safe(yhat, y)
        r2 = _r2_safe(y, yhat)

        if np.isfinite(r):
            pid_to_r.setdefault(pid, []).append(r)
        if np.isfinite(r2):
            pid_to_r2.setdefault(pid, []).append(r2)

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    r_real = mean_or_nan(pid_to_r.get(-1, []))
    r2_real = mean_or_nan(pid_to_r2.get(-1, []))

    fake_ids_r = sorted([pid for pid in pid_to_r.keys() if pid != -1])
    r_fake = np.asarray([mean_or_nan(pid_to_r[pid]) for pid in fake_ids_r], float)
    r_fake = r_fake[np.isfinite(r_fake)]

    fake_ids_r2 = sorted([pid for pid in pid_to_r2.keys() if pid != -1])
    r2_fake = np.asarray([mean_or_nan(pid_to_r2[pid]) for pid in fake_ids_r2], float)
    r2_fake = r2_fake[np.isfinite(r2_fake)]

    r_fake_mean, r_fake_sd, z_r, p_emp_r = _z_and_p_from_null(r_real, r_fake, one_sided="greater_equal")
    r2_fake_mean, r2_fake_sd, z_r2, p_emp_r2 = _z_and_p_from_null(r2_real, r2_fake, one_sided="greater_equal")

    r2_corr = _corrected_r2_findling(r2_real, r2_fake_mean)

    return {
        "n_used": n_used,
        "r_real": r_real,
        "r_fake_mean": r_fake_mean,
        "r_fake_std": r_fake_sd,
        "z_corr": z_r,
        "p_emp": p_emp_r,
        "r2_real": r2_real,
        "r2_fake_mean": r2_fake_mean,
        "r2_fake_std": r2_fake_sd,
        "z_r2": z_r2,
        "p_emp_r2": p_emp_r2,
        "r2_corr": r2_corr,
    }



def make_one() -> ONE:
    """Create ONE instance that reads/writes cache on the external SSD."""
    cache_dir = ONE_CACHE_DIR

    if USE_PER_PROCESS_TABLES:
        tables_dir = cache_dir.parent / "ONE_tables_per_process" / f"pid_{os.getpid()}"
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        tables_dir = None

    one = ONE(
        base_url=ONE_BASE_URL,
        password="international",
        cache_dir=cache_dir,
        tables_dir=tables_dir,
    )

    print(f"[ONE] base_url={ONE_BASE_URL}")
    print(f"[ONE] cache_dir={one.cache_dir}")
    if getattr(one, "tables_dir", None) is not None:
        print(f"[ONE] tables_dir={one.tables_dir}")

    host_dir = Path(one.cache_dir) / "openalyx.internationalbrainlab.org"
    print(f"[ONE] host_cache_exists={host_dir.exists()}  ({host_dir})")
    return one



def run_one_session(eid: str):
    warnings.filterwarnings("ignore", category=alferr.ALFWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not ONE_CACHE_DIR.exists():
        raise RuntimeError(
            f"[ERROR] ONE_CACHE_DIR not found:\n  {ONE_CACHE_DIR}\n"
            "Is the SSD mounted? Is the folder name exactly correct (spaces matter)?\n"
            "Expected structure like:\n"
            "  /Volumes/T7 Shield/SSD_ONE/ONE/openalyx.internationalbrainlab.org/\n"
        )

    one = make_one()


    try:
        probe_name = one.eid2pid(eid)[1]
    except Exception:
        probe_name = "UNKNOWN_PROBE"


    try:
        ses = one.alyx.rest("sessions", "read", id=eid)
        subject = ses.get("subject", "UNKNOWN_SUBJECT")
    except Exception as e:
        raise RuntimeError(f"Failed to read session metadata for {eid}: {type(e).__name__}: {e}")


    trials_df, n_raw_trials = load_trials_df(one, eid)
    print(f"[LOAD] eid={eid} trials={n_raw_trials}")

    if n_raw_trials < MIN_RAW_TRIALS:
        print(f"[SKIP] eid={eid} n_raw_trials={n_raw_trials} < MIN_RAW_TRIALS={MIN_RAW_TRIALS}")
        return

    drop_last_mask = make_drop_last_mask(len(trials_df), DROP_LAST_N)


    pseudo_ids = [-1] + list(range(1, N_PSEUDO + 1))
    session_dir = OUT_ROOT / eid / "session_fit"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] eid={eid} probe={probe_name} subject={subject}")
    print(f"[RUN] ROI_LIST={ROI_LIST}")
    print(f"[RUN] N_PSEUDO={N_PSEUDO} N_RUNS={N_RUNS} DROP_LAST_N={DROP_LAST_N}")
    print(f"[RUN] ALIGN_EVENT={ALIGN_EVENT} TIME_WINDOW={TIME_WINDOW}")

    _ = fit_session_ephys(
        one=one,
        session_id=eid,
        subject=subject,
        probe_name=probe_name,
        output_dir=session_dir,
        pseudo_ids=pseudo_ids,
        target="pLeft",
        align_event=ALIGN_EVENT,
        time_window=TIME_WINDOW,
        model="optBay",
        n_runs=N_RUNS,
        trials_df=trials_df,
        trial_mask=drop_last_mask,
        group_label="session",
        debug=bool(DEBUG),
        roi_set=ROI_SET,
    )
    rows = []
    for roi in ROI_LIST:
        roi_pkl = find_roi_pkl(session_dir, roi)
        if roi_pkl is None:
            print(f"[ROI] {roi}: no ROI pkl found -> skip")
            rows.append({
                "eid": eid, "subject": subject, "probe": str(probe_name),
                "roi": roi, "group": "session", "status": "skip_no_roi",
                "n_trials_group_used": 0, "drop_last_n": int(DROP_LAST_N),
                "roi_pkl_path": None,
            })
            continue

        stats = compute_session_stats_from_pkl(roi_pkl)
        print(f"[ROI] {roi}: n_used={stats['n_used']} r2_corr={stats['r2_corr']} r2_real={stats['r2_real']} r2_fake_mean={stats['r2_fake_mean']}")

        rows.append({
            "eid": eid,
            "subject": subject,
            "probe": str(probe_name),
            "roi": roi,
            "group": "session",
            "n_trials_group_used": int(stats["n_used"]),
            "r_real": stats["r_real"],
            "r_fake_mean": stats["r_fake_mean"],
            "r_fake_std": stats["r_fake_std"],
            "z_corr": stats["z_corr"],
            "p_emp": stats["p_emp"],
            "r2_real": stats["r2_real"],
            "r2_fake_mean": stats["r2_fake_mean"],
            "r2_fake_std": stats["r2_fake_std"],
            "r2_corr": stats["r2_corr"],
            "z_r2": stats["z_r2"],
            "p_emp_r2": stats["p_emp_r2"],
            "drop_last_n": int(DROP_LAST_N),
            "roi_pkl_path": str(roi_pkl),
            "status": "ok" if int(stats["n_used"]) >= int(MIN_TRIALS_FOR_METRICS) else "skip_lt_min_trials",
        })


    roi_tag = "_".join(ROI_LIST) if ROI_LIST else "ALL"
    out_json = OUT_ROOT / eid / f"whole_session_summary_{TAG}_{roi_tag}.json"
    out_pkl = OUT_ROOT / eid / f"whole_session_summary_{TAG}_{roi_tag}.pkl"

    out_json.write_text(
        json.dumps(rows, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    )
    with open(out_pkl, "wb") as f:
        pickle.dump(rows, f)

    print("[SAVE]", out_json.resolve())
    print("[SAVE]", out_pkl.resolve())


def main():
    run_one_session(SESSION_ID)


if __name__ == "__main__":
    main()