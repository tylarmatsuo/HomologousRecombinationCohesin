# The purpose of this code is to quickly concatenate first passage time output files, allowing many simulation runs
# to be combined into one file for downstream processing

from pathlib import Path
import numpy as np

# Modify this to your directory names
dir_names = ["YesLEFYesClamp", "NoLEFYesClamp", "YesLEFNoClamp", "NoLEFNoClamp", "YesLEFNoClamp_LEFsFallOffAtDSB", "YesLEFYesClamp_LEFsFallOffAtDSB", "YesLEFYesClampTranslocateOverDSB", "YesLEFNoClampTranslocateOverDSB"]

base = Path(".") # modify this as needed

for dname in dir_names:
    matches = list(base.glob(f"*/{dname}/*/*/first_pass_time_left.npy")) # modify this to match your directory structure

    arrays = []
    for f in matches:
        arr = np.load(f)
        arrays.append(arr)

    concatenated = np.concatenate(arrays, axis=0)

    out_name = f"concat_{dname}_n{len(matches)}_first_pass_time_left.npy"
    np.save(out_name, concatenated)
