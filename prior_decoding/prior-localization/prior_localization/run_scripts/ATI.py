
from __future__ import annotations

from pathlib import Path
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from one.api import ONE
import one.alf.exceptions as alferr

from prior_localization.my_rt import (
    load_wheel_data,
    calc_wheel_velocity,
    calc_trialwise_wheel,
    calc_movement_onset_times,
)



BEHAV_RUN_LOG_JSON = Path(
    "prior_localization_sessionfit_output_behav_sessions/"
    "run_log_BEHAV_SESSIONS_UPDATED_ROIS_ORBvl_PL_ACAd_TEa_VISC_AId_AIv_MOs_MOp_"
    "SSp-tr_SSp-bfd_SSp-ll_SSp-ul_SSp-n_SSs_VISp_VISpl_AUDp.json"
)

OUT_DIR = Path("prior_localization_sessionfit_output_behav_sessions/ati_outputs_all_sessions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAST_THR = 0.08
SLOW_THR = 1.25
DROP_LAST_N = 40


ONE_BASE_URL = "https://openalyx.internationalbrainlab.org"
ONE_CACHE_DIR = Path("/Volumes/T7 Shield/SSD_ONE/ONE")
USE_PER_PROCESS_TABLES = False

EID_LIST_TXT_FALLBACK = Path(
    "/Users/changyin/Documents/prior-localization/"
    "prior_localization/run_scripts/"
    "prior_localization_sessionfit_output/roi_all/"
    "eids_union_FROM_PKLS_ALL_ROIS_from_two_pkls.txt"
)


def load_eids_from_run_log(run_log_path: Path) -> list[str]:
    if not run_log_path.exists():
        raise FileNotFoundError(run_log_path)
    log = json.loads(run_log_path.read_text())
    eids = []
    for row in log:
        eid = row.get("eid") or row.get("session_id")
        if eid:
            eids.append(str(eid))
    return sorted(set(eids))

def load_eids_txt(path: Path) -> list[str]:
    p = Path(os.path.expanduser(str(path)))
    if not p.exists():
        raise FileNotFoundError(p)
    eids = [
        ln.strip() for ln in p.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return sorted(set(eids))

def make_one() -> ONE:
    if not ONE_CACHE_DIR.exists():
        raise RuntimeError(
            f"[ERROR] ONE_CACHE_DIR not found:\n  {ONE_CACHE_DIR}\n"
            "Is the T7 mounted? Is the folder name exactly correct (spaces matter)?\n"
            "Expected structure like:\n"
            "  /Volumes/T7 Shield/SSD_ONE/ONE/openalyx.internationalbrainlab.org/\n"
        )

    if USE_PER_PROCESS_TABLES:
        tables_dir = ONE_CACHE_DIR.parent / "ONE_tables_per_process" / f"pid_{os.getpid()}"
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        tables_dir = None

    one = ONE(
        base_url=ONE_BASE_URL,
        password="international",
        cache_dir=ONE_CACHE_DIR,
        tables_dir=tables_dir,
    )

    print(f"[ONE] base_url={ONE_BASE_URL}")
    print(f"[ONE] cache_dir={one.cache_dir}")
    if getattr(one, "tables_dir", None) is not None:
        print(f"[ONE] tables_dir={one.tables_dir}")
    host_dir = Path(one.cache_dir) / "openalyx.internationalbrainlab.org"
    print(f"[ONE] host_cache_exists={host_dir.exists()}  ({host_dir})")
    return one

def load_trials_as_df(one: ONE, eid: str) -> pd.DataFrame:
    trials = one.load_object(eid, "trials", collection="alf")
    data = {}
    for k, v in trials.items():
        arr = np.asarray(v)
        if k == "intervals" and arr.ndim == 2:
            data["intervals_start"] = arr[:, 0]
            data["intervals_end"] = arr[:, 1]
        elif arr.ndim == 1:
            data[k] = arr
        else:
            data[k] = list(arr)
    return pd.DataFrame(data)

def compute_first_movement_and_rt(one: ONE, eid: str, df: pd.DataFrame) -> pd.DataFrame:
    pos, ts = load_wheel_data(one, eid)
    if pos is None or ts is None:
        raise RuntimeError("no wheel data")

    vel = calc_wheel_velocity(pos, ts)

    _, trial_ts, trial_vel = calc_trialwise_wheel(
        pos, ts, vel,
        df["stimOn_times"],
        df["feedback_times"],
    )

    (_, first_mo, _, _, _) = calc_movement_onset_times(
        trial_ts, trial_vel,
        df["stimOn_times"],
    )

    out = df.copy()
    out["first_movement_onset_times"] = first_mo
    out["reaction_time"] = out["first_movement_onset_times"] - out["stimOn_times"]
    return out

def _coerce_subject_nickname(subject_field) -> str:

    if isinstance(subject_field, str):
        return subject_field
    if isinstance(subject_field, dict):
        # common keys seen in Alyx payloads
        for k in ("nickname", "name", "subject", "id"):
            v = subject_field.get(k)
            if isinstance(v, str) and v:
                return v
    return "UNKNOWN"

def make_subject_sex_getter(one: ONE):

    cache: dict[str, str] = {}

    def get_sex(subject_nickname: str) -> str:
        subj = str(subject_nickname)
        if subj in cache:
            return cache[subj]
        if subj in ("UNKNOWN", "", "None"):
            cache[subj] = "UNKNOWN"
            return cache[subj]

        sex = "UNKNOWN"
        try:

            sub = one.alyx.rest("subjects", "read", id=subj)
            # Alyx typically stores sex as 'sex' with values like 'M'/'F' (sometimes full words)
            sex_val = sub.get("sex", None)
            if isinstance(sex_val, str) and sex_val.strip():
                sex = sex_val.strip()
        except Exception:
            sex = "UNKNOWN"

        cache[subj] = sex
        return sex

    return get_sex


def compute_session_ati(df: pd.DataFrame, eid: str, subject: str, sex: str) -> dict:
    # drop last N FIRST
    df = df.iloc[: max(0, len(df) - DROP_LAST_N)].copy()

    ok = (
        np.isfinite(pd.to_numeric(df["stimOn_times"], errors="coerce").to_numpy())
        & np.isfinite(pd.to_numeric(df["first_movement_onset_times"], errors="coerce").to_numpy())
    )
    df = df.loc[ok].copy()

    n = int(len(df))
    if n == 0:
        return {
            "session_id": eid,
            "subject": subject,
            "sex": sex,  # <<< added
            "n_total": 0,
            "n_impulsive": 0,
            "n_slow": 0,
            "ATI_session": np.nan,
            "status": "skip_no_valid_trials",
        }

    rt = pd.to_numeric(df["reaction_time"], errors="coerce").to_numpy()
    rt = rt[np.isfinite(rt)]
    n = int(len(rt))
    if n == 0:
        return {
            "session_id": eid,
            "subject": subject,
            "sex": sex,  # <<< added
            "n_total": 0,
            "n_impulsive": 0,
            "n_slow": 0,
            "ATI_session": np.nan,
            "status": "skip_no_valid_rt",
        }

    n_imp = int(np.sum(rt < FAST_THR))
    n_slow = int(np.sum(rt > SLOW_THR))
    ati = float((n_imp - n_slow) / n)

    return {
        "session_id": eid,
        "subject": subject,
        "sex": sex,  # <<< added
        "n_total": n,
        "n_impulsive": n_imp,
        "n_slow": n_slow,
        "ATI_session": ati,
        "status": "ok",
    }



def main():
    warnings.filterwarnings("ignore")

    if BEHAV_RUN_LOG_JSON.exists():
        eids = load_eids_from_run_log(BEHAV_RUN_LOG_JSON)
        print(f"[EIDS] loaded {len(eids)} sessions from run log:\n  {BEHAV_RUN_LOG_JSON}")
    else:
        eids = load_eids_txt(EID_LIST_TXT_FALLBACK)
        print(f"[EIDS] loaded {len(eids)} sessions from txt fallback:\n  {EID_LIST_TXT_FALLBACK}")

    print("First few:", eids[:3])

    one = make_one()
    get_sex = make_subject_sex_getter(one)  # <<< added

    session_rows = []
    run_log = []

    # animal-level accumulator
    animal_acc: dict[str, dict] = {}

    for eid in eids:
        try:
            ses = one.alyx.rest("sessions", "read", id=eid)
            subject = _coerce_subject_nickname(ses.get("subject", "UNKNOWN"))  # <<< robust
            sex = get_sex(subject)  # <<< added

            df_trials = load_trials_as_df(one, eid)

            if "stimOn_times" not in df_trials.columns or "feedback_times" not in df_trials.columns:
                raise RuntimeError("missing stimOn_times or feedback_times in trials")

            df_rt = compute_first_movement_and_rt(one, eid, df_trials)

            row = compute_session_ati(df_rt, eid, subject, sex)  # <<< pass sex
            session_rows.append(row)

            if row["status"] == "ok":
                acc = animal_acc.setdefault(
                    subject,
                    {"sex": sex, "n_total": 0, "n_imp": 0, "n_slow": 0, "sessions": []},  # <<< store sex
                )

                # if sex ever disagrees across sessions, keep first but note it
                if acc.get("sex", "UNKNOWN") == "UNKNOWN" and sex != "UNKNOWN":
                    acc["sex"] = sex
                elif sex != "UNKNOWN" and acc.get("sex", "UNKNOWN") not in ("UNKNOWN", sex):
                    print(f"[WARN] Sex mismatch for subject {subject}: acc={acc.get('sex')} vs session={sex}")

                acc["n_total"] += row["n_total"]
                acc["n_imp"] += row["n_impulsive"]
                acc["n_slow"] += row["n_slow"]
                acc["sessions"].append(eid)

                animal_ati = (
                    (acc["n_imp"] - acc["n_slow"]) / acc["n_total"]
                    if acc["n_total"] > 0 else np.nan
                )

                print(
                    f"[DONE] {eid} | subj={subject} | sex={sex} | "
                    f"session ATI={row['ATI_session']:.4f} "
                    f"(n={row['n_total']}, imp={row['n_impulsive']}, slow={row['n_slow']}) | "
                    f"animal ATI={animal_ati:.4f} "
                    f"(sessions={len(acc['sessions'])}, n={acc['n_total']})"
                )
            else:
                print(f"[SKIP] {eid} | subj={subject} | sex={sex} | {row['status']}")

            run_log.append({"eid": eid, "subject": subject, "sex": sex, "status": row["status"]})  # <<< added sex

        except (alferr.ALFObjectNotFound, FileNotFoundError) as e:
            msg = f"missing_data:{type(e).__name__}"
            print(f"[SKIP] {eid} {msg}: {e}")
            run_log.append({"eid": eid, "subject": None, "sex": None, "status": msg})

        except Exception as e:
            msg = f"exception:{type(e).__name__}"
            print(f"[SKIP] {eid} {msg}: {e}")
            run_log.append({"eid": eid, "subject": None, "sex": None, "status": msg})


    session_df = pd.DataFrame(session_rows)
    session_df.to_csv(OUT_DIR / "ATI_session_level.csv", index=False)
    with open(OUT_DIR / "ATI_session_level.pkl", "wb") as f:
        pickle.dump(session_rows, f)


    animal_rows = []
    for subj, acc in animal_acc.items():
        ati = (
            (acc["n_imp"] - acc["n_slow"]) / acc["n_total"]
            if acc["n_total"] > 0 else np.nan
        )
        animal_rows.append(
            {
                "subject": subj,
                "sex": acc.get("sex", "UNKNOWN"),  # <<< added
                "session_ids": ";".join(acc["sessions"]),
                "n_sessions": len(acc["sessions"]),
                "n_total_trials": acc["n_total"],
                "n_impulsive": acc["n_imp"],
                "n_slow": acc["n_slow"],
                "ATI_animal": ati,
            }
        )

    animal_df = pd.DataFrame(animal_rows)
    animal_df.to_csv(OUT_DIR / "ATI_animal_level.csv", index=False)
    with open(OUT_DIR / "ATI_animal_level.pkl", "wb") as f:
        pickle.dump(animal_rows, f)

    (OUT_DIR / "run_log_ATI.json").write_text(json.dumps(run_log, indent=2))

    print("\nSaved outputs to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()