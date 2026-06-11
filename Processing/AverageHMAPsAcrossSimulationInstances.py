#!/usr/bin/env python3
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path
from polychrom.hdf5_format import list_URIs, load_URI
from polychrom.contactmaps import monomerResolutionContactMapSubchains

# USER PARAMETERS
N = 80000
subchain_length = 4000 * 2
dsb_time = 1000
SAVE_EVERY_BLOCKS = 10


# HELPER FUNCTIONS

def collect_trajectory0_paths(base_dir):
    """
    Find all directory paths of the form:
    instance/*/simulation_name/trajectory0
    and return dict grouped by (tethering_folder, simulation_folder).
    """

    groups = {}  # key: (tethering, simulation), value: [trajectory0_paths]

    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "trajectory0":
            traj_path = Path(root)
            simulation = traj_path.parent.name
            tethering = traj_path.parent.parent.name  # Connected / Disconnected
            instance = traj_path.parent.parent.parent.name

            key = (tethering, simulation)
            groups.setdefault(key, []).append(traj_path)

    return groups


def compute_hmap_for_traj(traj_path):
    """
    Given a trajectory0 folder, compute its hmap.
    """

    folder_name = str(traj_path)
    URIs = list_URIs(folder_name)

    block_cutoff = dsb_time // SAVE_EVERY_BLOCKS
    URIs = [uri for uri in URIs if int(uri.split("::")[-1]) > block_cutoff]

    starts = list(range(0, N, subchain_length))
    map_size = subchain_length

    hmap = monomerResolutionContactMapSubchains(
        filenames=URIs,
        mapStarts=starts,
        mapN=map_size,
        cutoff=5, # cutoff of 5 is used; this can be changed if desired
        n=1,
        loadFunction=lambda x: load_URI(x)["pos"]
    )

    return hmap


def compute_group_mean(args):
    """
    Compute mean hmap for one group (tethering, simulation).
    Runs each trajectory0 folder in parallel.
    """
    (key, traj_paths, outdir) = args
    tethering, simulation = key

    print(f"Processing group: {tethering}/{simulation} (n={len(traj_paths)} instances)")

    # Parallel compute hmaps
    with Pool(cpu_count()) as pool:
        hmaps = pool.map(compute_hmap_for_traj, traj_paths)

    mean_hmap = np.mean(hmaps, axis=0)

    outfile = outdir / f"mean_hmap_{tethering}_{simulation}.npy"
    np.save(outfile, mean_hmap)
    print(f"Saved {outfile}")

    return outfile


# MAIN

def main():
    base_dir = Path(os.getcwd())
    outdir = base_dir / "mean_hmaps"
    outdir.mkdir(exist_ok=True)

    print(f"Scanning directory: {base_dir}")
    groups = collect_trajectory0_paths(base_dir)

    print(f"Found {len(groups)} simulation groups.")

    tasks = [(key, traj_paths, outdir) for key, traj_paths in groups.items()]

    for t in tasks:
        compute_group_mean(t)


if __name__ == "__main__":
    main()
