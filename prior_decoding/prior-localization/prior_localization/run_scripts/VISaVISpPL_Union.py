from __future__ import annotations

from pathlib import Path
import pickle

DOWNLOADS = Path("/Users/changyin/Downloads/spids")
IN_FILES = [
    DOWNLOADS / "PL_eid_pid.txt",
    DOWNLOADS / "VISa_eid_pid.txt",
    DOWNLOADS / "VISp_eid_pid.txt",
]


TAG = "VISaVISpPL"


def read_eids_from_region_eid_pid_txt(path: Path) -> set[str]:
    """
    Each line looks like:
      <region> <eid> <pid>
    separated by spaces/tabs.
    We take the 2nd column as eid.
    """
    eids: set[str] = set()
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()  # splits on any whitespace
        if len(parts) < 2:
            continue
        eid = parts[1].strip()
        if eid:
            eids.add(eid)
    return eids


def main():

    SCRIPT_DIR = Path(__file__).resolve().parent  #
    DEFAULT_EID_DIR = SCRIPT_DIR / "prior_localization_sessionfit_output" / "roi_eids_all"
    DEFAULT_EID_DIR.mkdir(parents=True, exist_ok=True)

    # === UNION EIDS ===
    all_eids: set[str] = set()
    for f in IN_FILES:
        if not f.exists():
            raise FileNotFoundError(f"Missing input file: {f}")
        all_eids |= read_eids_from_region_eid_pid_txt(f)

    eids_sorted = sorted(all_eids)
    if not eids_sorted:
        raise RuntimeError("No EIDs found. Check your input file formatting.")
    out_txt = DEFAULT_EID_DIR / f"eids_union_{TAG}.txt"
    out_pkl = DEFAULT_EID_DIR / f"eids_union_{TAG}.pkl"

    out_txt.write_text("\n".join(eids_sorted) + "\n")
    with open(out_pkl, "wb") as f:
        pickle.dump(eids_sorted, f)

    print(f"[OK] Unioned {len(eids_sorted)} unique EIDs")
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_pkl}")
    print("NOTE: Did NOT write eids_union.txt / eids_union.pkl (to avoid overwriting).")


if __name__ == "__main__":
    main()
