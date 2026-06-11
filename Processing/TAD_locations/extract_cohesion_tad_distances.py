"""
extract_cohesion_tad_distances.py  —  Stage 1 (cluster)
========================================================
For each Instance folder, finds the first timepoint at which cohesive cohesin
binds on the sister chromatid (monomer index >= N_PER_STRAND = 2000), then
records the monomer indices of those endpoints and their 3D coordinates, along
with the TAD boundary positions (offset +2000 from damaged-strand storage).

TAD_Boundary_Positions.npy shape: (N_DSB=5, 4), values in [0, 2000)
  → add N_PER_STRAND to convert to sister-chromatid index space.

Output per instance: a lightweight .npy dict
"""

import argparse
import os
import numpy as np

# Parameters
N_PER_STRAND   = 2000
N_DSB          = 5
N              = N_PER_STRAND * 2        
SUBFOLDER      = "path/to/trajectory" # change this!
DSB_SITE_LEFT  = 999
DSB_SITE_RIGHT = 1000

def find_first_two_cohesion_events(cohesion_positions_all, dsb_instance):
    """
    Finds first binding for left and right DSB flanks.

    Returns
    -------
    dict with keys:
        "left":  (t_first, [indices]) or None
        "right": (t_first, [indices]) or None
    """
    offset = dsb_instance * N

    sister_lo = offset + N_PER_STRAND
    sister_hi = offset + N

    left_found  = None
    right_found = None

    for t, cohesins in enumerate(cohesion_positions_all):
        if cohesins is None or len(cohesins) == 0:
            continue

        for (a, b) in cohesins:

            # classify endpoints
            if a < b:
                dam, sis = (a, b) if a < sister_lo else (b, a)
            else:
                dam, sis = (b, a) if b < sister_lo else (a, b)

            # skip if not cross-strand
            if not (sister_lo <= sis < sister_hi):
                continue

            # determine which DSB side
            dam_local = dam - offset

            # determine which DSB side by nearest endpoint
            d_left  = abs(dam_local - DSB_SITE_LEFT)
            d_right = abs(dam_local - DSB_SITE_RIGHT)

            if d_left <= d_right:
                side = "left"
            else:
                side = "right"

            # assign event
            if side == "left" and left_found is None:
                left_found = (t, np.array([sis], dtype=int))

            elif side == "right" and right_found is None:
                right_found = (t, np.array([sis], dtype=int))

        if left_found is not None and right_found is not None:
            break

    return {"left": left_found, "right": right_found}


def get_sister_cohesion_indices(cohesion_positions_all, t_first, dsb_instance):
    """
    Returns all global endpoint indices that fall on the sister chromatid
    of dsb_instance at timestep t_first.
    """
    sister_lo = dsb_instance * N + N_PER_STRAND
    sister_hi = (dsb_instance + 1) * N

    sister_indices = []
    for (left, right) in cohesion_positions_all[t_first]:
        if sister_lo <= left < sister_hi:
            sister_indices.append(left)
        if sister_lo <= right < sister_hi:
            sister_indices.append(right)
    return np.array(sister_indices, dtype=int)

def extract_instance(base_dir, instance_idx, out_dir):
    folder = os.path.join(base_dir, f"Instance{instance_idx}", SUBFOLDER) # Important! Part of file path for when data is split across instances. This should match your file path.
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"[Instance {instance_idx}] Loading factor data ...")

    tad_raw = np.load(os.path.join(folder, "TAD_Boundary_Positions.npy"),
                      allow_pickle=True)

    cohesion_all = np.load(
        os.path.join(folder, "RecruitedCohesiveCohesin_Positions.npy"),
        allow_pickle=True
    )

    dsb_results = []

    for dsb_i in range(N_DSB):
        tad_sister = np.asarray(tad_raw[dsb_i], dtype=int) + N_PER_STRAND
        tad_global = tad_sister + dsb_i * N

        events = find_first_two_cohesion_events(cohesion_all, dsb_i)

        tad_sister = np.asarray(tad_raw[dsb_i], dtype=int) + N_PER_STRAND
        tad_global = tad_sister + dsb_i * N

        event_list = []

        for side in ["left", "right"]:
            event = events[side]

            if event is None:
                continue

            t_first, cohesion_idx = event

            event_list.append({
                "found": True,
                "dsb_instance": dsb_i,
                "side": side,
                "t_first": t_first,
                "cohesion_indices": cohesion_idx,
                "tad_indices": tad_global,
            })

        if len(event_list) == 0:
            dsb_results.append({"found": False, "dsb_instance": dsb_i})
        else:
            dsb_results.extend(event_list)

    result = {
        "instance_idx": instance_idx,
        "dsb_results" : np.array(dsb_results, dtype=object),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cohesion_tad_{instance_idx:02d}.npy")
    np.save(out_path, result)
    print(f"[Instance {instance_idx}] Saved → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",  required=True)
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--instance",  required=True, type=int)
    args = p.parse_args()

    extract_instance(args.base_dir, args.instance, args.out_dir)
