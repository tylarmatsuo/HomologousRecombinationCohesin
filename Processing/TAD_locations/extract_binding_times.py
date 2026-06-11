import argparse
import os
import numpy as np

from polychrom.hdf5_format import list_URIs, load_URI

# some points of the code below are designed for path structure InstanceX/subdirectory, so that simulations can be parallelized across instances. These have been highlighted.
# parameters
N_PER_STRAND   = 2000
N_STRANDS      = 2
N              = N_PER_STRAND * N_STRANDS
N_DSB          = 5
CONTACT_THRESHOLD = 2

DSB_SITE_LEFT  = 999
DSB_SITE_RIGHT = 1000

HOMOLOGY_LEFT  = N_PER_STRAND + DSB_SITE_LEFT
HOMOLOGY_RIGHT = N_PER_STRAND + DSB_SITE_RIGHT

RNG_SEED = 2961


def find_binding_and_positions(
    URIs,
    dsb_instance,
    N,
    dsb_left,
    dsb_right,
    hom_left,
    hom_right,
    contact_threshold,
    tad_boundaries,
    rng
):
    offset = dsb_instance * N
    frame_counter = 0

    boundary_indices = np.asarray(tad_boundaries, dtype=int)

    for uri in URIs:
        block = load_URI(uri)
        pos = block["pos"]

        if pos.ndim == 2:
            pos = pos[np.newaxis, :]

        for frame in range(pos.shape[0]):

            dsb_L = pos[frame, offset + dsb_left]
            dsb_R = pos[frame, offset + dsb_right]

            hom_L = pos[frame, offset + hom_left]
            hom_R = pos[frame, offset + hom_right]

            d_left  = np.linalg.norm(dsb_L - hom_L)
            d_right = np.linalg.norm(dsb_R - hom_R)

            if min(d_left, d_right) < contact_threshold:

                dsb_center = 0.5 * (dsb_L + dsb_R)

                if len(boundary_indices) > 0:
                    tad_positions = pos[frame, offset + boundary_indices]
                else:
                    tad_positions = np.empty((0, 3))

                n_rand = len(boundary_indices)
                if n_rand > 0:
                    rand_idx = rng.integers(0, N, size=n_rand)
                    random_positions = pos[frame, offset + rand_idx]
                else:
                    random_positions = np.empty((0, 3))

                return frame_counter, dsb_center, tad_positions, random_positions

            frame_counter += 1

    return None, None, None, None


def extract_instance(base_dir, instance_idx, subfolder, out_dir):

    rng = np.random.default_rng(RNG_SEED + instance_idx)

    folder = os.path.join(base_dir, f"Instance{instance_idx}", subfolder) # Important! File path. This should match your file path structure.

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Trajectory folder not found: {folder}")

    print(f"[Instance {instance_idx}] Listing URIs in {folder} …")
    URIs = list_URIs(folder)

    tad_path = os.path.join(folder, "TAD_Boundary_Positions.npy")
    tad_boundaries_all = np.load(tad_path, allow_pickle=True)

    binding_times = []
    dsb_positions = []
    tad_positions_all = []
    random_positions_all = []

    for dsb_instance in range(N_DSB):

        t_bind, dsb_center, tad_pos, rand_pos = find_binding_and_positions(
            URIs,
            dsb_instance,
            N,
            DSB_SITE_LEFT,
            DSB_SITE_RIGHT,
            HOMOLOGY_LEFT,
            HOMOLOGY_RIGHT,
            CONTACT_THRESHOLD,
            tad_boundaries_all[dsb_instance],
            rng
        )

        binding_times.append(t_bind)
        dsb_positions.append(dsb_center)
        tad_positions_all.append(tad_pos)
        random_positions_all.append(rand_pos)

        print(f"  DSB sub-instance {dsb_instance}: t_bind = {t_bind}")

    result = {
        "instance_idx": instance_idx,
        "binding_times": np.array(binding_times, dtype=object),
        "dsb_positions": np.array(dsb_positions, dtype=object),
        "tad_positions": np.array(tad_positions_all, dtype=object),
        "random_positions": np.array(random_positions_all, dtype=object),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"instance_{instance_idx:02d}_results.npy") # Important! File path. This should match your file path.
    np.save(out_path, result)

    print(f"[Instance {instance_idx}] Saved → {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",   required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--instance",   required=True, type=int)
    p.add_argument("--subfolder",
                   default="/trajectory0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_instance(
        base_dir=args.base_dir,
        instance_idx=args.instance,
        subfolder=args.subfolder,
        out_dir=args.out_dir,
    )