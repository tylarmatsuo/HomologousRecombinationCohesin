import pickle
import os
import time
import numpy as np
import polychrom
import copy
import pandas as pd

from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.contactmaps import monomerResolutionContactMapSubchains
import simtk.openmm 
import os 
import shutil
from collections import defaultdict

import warnings
import h5py 
import glob

from parameters import N1, M, LIFETIME, SEPARATION, SEPARATION_AFTER_DSB, trajectoryLength, simulation_cycles, steady_state_relax_1D, dsb_time, ctcf_sites, dsb_site, dsb_ends_connected, B_CTCF, contacts_cutoff # Import parameters

# -------1D Simulation----------------
# several functions in this section are adapted from functions included in the polychrom package: https://github.com/open2c/polychrom
# note that the "CTCF" flag is used for all boundary elements, including DSB sites, not just CTCFs

class leg(object):
    def __init__(self, pos, attrs={"stalled":False, "CTCF":False}):
        """
        A leg has two important attribues: pos (positions) and attrs (a custom list of attributes)
        """
        self.pos = pos
        self.attrs = dict(attrs)

class cohesin(object):
    """
    A cohesin class provides fast access to attributes and positions 
    
    
    cohesin.left is a left leg of cohesin, cohesin.right is a right leg
    cohesin[-1] is also a left leg and cohesin[1] is a right leg         
    
    Also, cohesin.any("myattr") is True if myattr==True in at least one leg
    cohesin.all("myattr") is if myattr=True in both legs
    """
    def __init__(self, leg1, leg2):
        self.left = leg1
        self.right = leg2
   
    def any(self, attr):
        return self.left.attrs[attr] or self.right.attrs[attr]
    
    def all(self, attr):
        return self.left.attrs[attr] and self.right.attrs[attr]    
    
    def __getitem__(self, item):
        if item == -1:
            return self.left
        elif item == 1:
            return self.right 
        else:
            raise ValueError()
        

def unloadProb(cohesin, args):
    """
    Defines probability based on a state of cohesin 
    """
    if cohesin.any("CTCF"):
        return 1 / args["LIFETIME_CAPTURED"]
    elif cohesin.any("stalled"):
        return 1 / args["LIFETIME_STALLED"]
    return 1 / args["LIFETIME"]      
    


def loadOne(cohesins, occupied, args): 
    """
    A function to load one cohesin 
    """
    while True:
        a = np.random.randint(args["N"])
        if (occupied[a] == 0) and (occupied[a+1] == 0):
            occupied[a] = 1
            occupied[a+1] = 1 
            cohesins.append(cohesin(leg(a), leg(a+1)))
            break


def capture(cohesin, occupied, args):
    for side in [1, -1]:
        # get probability of capture or otherwise it is 0 
        if np.random.random() < args["Capture"][side].get(cohesin[side].pos, 0):  
            cohesin[side].attrs["CTCF"] = True  # captured a cohesin at CTCF     
    return cohesin 


def release(cohesin, occupied, args):
    
    """
    Releasing cohesins from boundary elements 
    """
    
    if not cohesin.any("CTCF"):
        return cohesin  # no CTCF: no release necessary 
        
    # attempting to release either side 
    for side in [-1, 1]: 
        if (np.random.random() < args["Release"][side].get(cohesin[side].pos, 0)) and (cohesin[side].attrs["CTCF"]):
            cohesin[side].attrs["CTCF"] = False 
    return cohesin 


def translocate(cohesins, occupied, args):
    """
    This function describes everything that happens with cohesins - 
    loading/unloading them and stalling against each other 
    
    It relies on the functions defined above: unload probability, capture/release. 
    """
    # first we try to unload cohesins and free the matching occupied sites 
    for i in range(len(cohesins)):
        prob = unloadProb(cohesins[i], args)
        if np.random.random() < prob:
            occupied[cohesins[i].left.pos] = 0 
            occupied[cohesins[i].right.pos] = 0 
            del cohesins[i]
            loadOne(cohesins, occupied, args)
    
    # then we try to capture and release them by CTCF sites 
    for i in range(len(cohesins)):
        cohesins[i] = capture(cohesins[i], occupied, args)
        cohesins[i] = release(cohesins[i], occupied, args)
    
    # finally we translocate, and mark stalled cohesins because 
    # the unloadProb needs this 
    for i in range(len(cohesins)):
        cohesin = cohesins[i] 
        for leg in [-1,1]: 
            if not cohesin[leg].attrs["CTCF"]: 
                # cohesins that are not at CTCFs and cannot move are labeled as stalled 
                if occupied[cohesin[leg].pos  + leg] != 0:
                    cohesin[leg].attrs["stalled"] = True
                else:
                    cohesin[leg].attrs["stalled"] = False 
                    occupied[cohesin[leg].pos] = 0
                    occupied[cohesin[leg].pos + leg] = 1
                    cohesin[leg].pos += leg        
        cohesins[i] = cohesin
        
def color(cohesins, args):
    "A helper function that converts a list of cohesins to an array colored by cohesin state"    
    def state(attrs):
        if attrs["stalled"]:
            return 2
        if attrs["CTCF"]:
            return 3
        return 1
    ar = np.zeros(args["N"])
    for i in cohesins:
        ar[i.left.pos] = state(i.left.attrs)
        ar[i.right.pos] = state(i.right.attrs)  
    return ar 

N = N1 * M
LEFNum = N // SEPARATION # Initial LEF count

if SEPARATION_AFTER_DSB:
    post_dsb_LEFnum = N // SEPARATION_AFTER_DSB
    total_LEFNum = post_dsb_LEFnum
else:
    total_LEFNum = LEFNum

CTCFs = ctcf_sites

dsb_sites = dsb_site # ex: [549,550, 1149,1150, 1849,1850, 2649,2650, 3149,3150]

ctcfLeftRelease = {}
ctcfRightRelease = {}
ctcfLeftCapture = {}
ctcfRightCapture = {}

for i in range(M):
    # CTCFs in pre-DSB regime
    for CTCF in CTCFs:
        ctcf_pos = i * N1 + CTCF 
        ctcfLeftCapture[ctcf_pos] = B_CTCF  # 50% capture probability 
        ctcfLeftRelease[ctcf_pos] = 0.0000  # hold it for ~5000 blocks on average
        ctcfRightCapture[ctcf_pos] = B_CTCF
        ctcfRightRelease[ctcf_pos] = 0.0000

args = {}
args["Release"] = {-1:ctcfLeftRelease, 1:ctcfRightRelease}
args["Capture"] = {-1:ctcfLeftCapture, 1:ctcfRightCapture}        
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_CAPTURED"] = LIFETIME # Change in lifetime when at CTCF or DSB position
args["LIFETIME_STALLED"] = LIFETIME  # change in lifetime when stalled 
    
# Positions of DSBs and CTCFs and capture probabilities for post-DSB regime.
dsbLeftCapture = copy.deepcopy(ctcfLeftCapture)
dsbLeftRelease = copy.deepcopy(ctcfLeftRelease)
dsbRightCapture = copy.deepcopy(ctcfRightCapture)
dsbRightRelease = copy.deepcopy(ctcfRightRelease)
if dsb_site:
    DSBs = [dsb_site - 1, dsb_site]
    for i in range(M):
        for dsb in DSBs:
            dsb_pos = i * N1 + dsb 
            dsbLeftCapture[dsb_pos] = 1.
            dsbLeftRelease[dsb_pos] = 0.
            dsbRightCapture[dsb_pos] = 1.
            dsbRightRelease[dsb_pos] = 0.

class bondUpdater(object):

    def __init__(self, LEFpositions):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks=100, smcStepsPerBlock=1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """


        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        #precalculating all bonds
        allBonds = []
        
        loaded_positions  = self.LEFpositions[self.curtime : self.curtime+blocks]
        for i in range(loaded_positions.shape[0]): # NEW 5/24/25 to tolerate NaNs
            frame_bonds = []
            for j in range(loaded_positions.shape[1]):
                left, right = loaded_positions[i, j]
                if not np.isnan(left) and not np.isnan(right):
                    frame_bonds.append((int(left), int(right)))
            allBonds.append(frame_bonds)
        
        if False: # old code (before 5/24/25) for bonds, probably doesn't tolerate NaNs well
            allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                            for j in range(loaded_positions.shape[1])] for i in range(blocks)]

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks 
        
        return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curBonds, pastBonds
        
for cycle in range(simulation_cycles):
    save_folder_name = f"trajectory{cycle}"
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)

    occupied = np.zeros(N)
    occupied[0] = 1
    occupied[-1] = 1 
    cohesins = []

    ### equilibriate ###
    for i in range(LEFNum):
        loadOne(cohesins,occupied, args)

    for _ in range(steady_state_relax_1D):
        translocate(cohesins, occupied, args)

    ### make 1D file containing LEF positions ###
    with h5py.File(f"{save_folder_name}/LEFPositions.h5", mode='w') as myfile:
        
        dset = myfile.create_dataset("positions", 
                                    shape=(trajectoryLength, total_LEFNum, 2), 
                                    dtype=np.float32, # NEW 5/24/25 was int32
                                    compression="gzip")
        steps = 50    # saving in 50 chunks because the whole trajectory may be large 
        assert (dsb_time % steps) == 0 # check if steps will evenly divide trajectory length
        pre_dsb_bins = np.linspace(0, dsb_time, steps, dtype=int) # chunks boundaries 
        for st,end in zip(pre_dsb_bins[:-1], pre_dsb_bins[1:]):
            cur = []
            for i in range(st, end):
                translocate(cohesins, occupied, args)  # actual step of LEF dynamics 
                positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
                if SEPARATION_AFTER_DSB: # NEW 5/24/25
                    positions = positions + [(np.nan, np.nan)] * (post_dsb_LEFnum - LEFNum)
                cur.append(positions)  # appending current positions to an array 
            cur = np.array(cur)  # when we finished a block of positions, save it to HDF5 
            dset[st:end] = cur

        post_dsb_bins = np.linspace(dsb_time, trajectoryLength, steps, dtype=int)
        # Change args values for capture and release to include DSBs (capture 1 release 0)
        args["Release"] = {-1:dsbLeftRelease, 1:dsbRightRelease}
        args["Capture"] = {-1:dsbLeftCapture, 1:dsbRightCapture}

        if SEPARATION_AFTER_DSB: # NEW 5/24/25
            post_dsb_newLEF = post_dsb_LEFnum - LEFNum
            for _ in range(post_dsb_newLEF):
                loadOne(cohesins,occupied, args) 
                
        for st,end in zip(post_dsb_bins[:-1], post_dsb_bins[1:]):
            cur = []
            for i in range(st, end):
                translocate(cohesins, occupied, args)  # actual step of LEF dynamics 
                positions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
                cur.append(positions)  # appending current positions to an array 
            cur = np.array(cur)  # when we finished a block of positions, save it to HDF5 
            dset[st:end] = cur

        myfile.attrs["N1"] = N1
        myfile.attrs["M"] = M
        myfile.attrs["N"] = N
        myfile.attrs["LEFNum"] = LEFNum

    ### parameters for 3D simulation ###

    folder = save_folder_name

    myfile = h5py.File(f"{folder}/LEFPositions.h5", mode='r')

    N1 = myfile.attrs["N1"]
    M = myfile.attrs["M"]
    N = myfile.attrs["N"]
    LEFNum = myfile.attrs["LEFNum"]
    LEFpositions = myfile["positions"]

        
    steps = 750   # MD steps per step of cohesin
    stiff = 1
    dens = 0.1
    box = (N / dens) ** 0.33  # density = 0.1.
    data = grow_cubic(N, int(box) - 2)  # creates a compact conformation 
    block = 0  # starting block 

    saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
    restartSimulationEveryBlocks = 100

    dsbTime = dsb_time # Block at which DSBs are introduced (adjust as needed)
    dsbIteration = dsbTime/restartSimulationEveryBlocks # Iteration at which DSBs are introduced

    # parameters for smc bonds
    smcBondWiggleDist = 0.2
    smcBondDist = 0.5

    # assertions for managing code below 
    assert (trajectoryLength % restartSimulationEveryBlocks) == 0 
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal  = (trajectoryLength) // restartSimulationEveryBlocks

    # Set extra bonds (linking DSB sites pre-break) and polymer chains
    dsb_extra_bonds = []
    dsb_polymer_chains = []
    breakpoints = []
    if not dsb_ends_connected and dsb_site and dsb_site > 0:
            breakpoints.append(0)
            for i in range(M):
                instance_offset = i * N1
                break_pos = instance_offset + dsb_site
                dsb_extra_bonds.append((break_pos - 1, break_pos))
                breakpoints.append(break_pos)

    breakpoints = sorted(set(breakpoints))

    if breakpoints:
        dsb_polymer_chains = [(breakpoints[i], breakpoints[i + 1], 0)
                            for i in range(len(breakpoints) - 1)]
        dsb_polymer_chains.append((breakpoints[-1], None, 0))
    else:
        dsb_polymer_chains = [(0, None, 0)]  # one continuous chain
        
    milker = bondUpdater(LEFpositions)

    reporter = HDF5Reporter(folder=folder, max_data_length=100, overwrite=True, blocks_only=False)

    ### Simulation main loop ###
    for iteration in range(simInitsTotal):
        
        # simulation parameters are defined below 
        a = Simulation(
                platform="CUDA",
                integrator="variableLangevin", 
                error_tol=0.01, 
                GPU = "0", 
                collision_rate=0.03, 
                N = len(data),
                reporters=[reporter],
                PBCbox=[box, box, box],
                precision="mixed")  # timestep not necessary for variableLangevin

        a.set_data(data)  # loads a polymer, puts a center of mass at zero
        
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains= dsb_polymer_chains,

                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength':1.0,
                    'bondWiggleDistance':0.05, # Bond distance will fluctuate +- 0.05 on average
                },

                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k':1.5
                },

                nonbonded_force_func=forces.polynomial_repulsive,
                nonbonded_force_kwargs={
                    'trunc':3.0,
                },

                except_bonds=True,

                # Remove DSB extra bonds (induce DSBs) after a defined number of steps
                extra_bonds = dsb_extra_bonds if iteration < dsbIteration
                else None
                
            )
        )
        
        # initializing milker; adding bonds
        # copied from addBond
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        milker.setParams(activeParams, inactiveParams)
        
        # this step actually puts all bonds in and sets first bonds to be what they should be
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                    blocks=restartSimulationEveryBlocks)

        if iteration==0:
            a.local_energy_minimization() 
        else:
            a._apply_forces()
        
        for i in range(restartSimulationEveryBlocks):        
            if i % saveEveryBlocks == (saveEveryBlocks - 1):  
                a.do_block(steps=steps)
            else:
                a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)
            if i < restartSimulationEveryBlocks - 1: 
                curBonds, pastBonds = milker.step(a.context)  # this updates bonds
        data = a.get_data()  # save data and step, and delete the simulation
        del a
        
        reporter.blocks_only = True  # Write output hdf5-files only for blocks

        print('Simulation iteration', iteration + 1, 'of', simInitsTotal, 'is done!')

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    reporter.dump_data()

### Create hmaps showing contacts after DSB (filtering out pre-DSB data) ###
# note that this outputs the mean hmap, not the total contacts. To get total contacts (e.g. for downstream analysis, pooling hmaps together across instances, etc.), use the saved trajectory files to generate new hmaps. 
if dsb_time % saveEveryBlocks != 0:
    print(f"Warning: dsb_time ({dsb_time}) is not divisible by saveEveryBlocks ({saveEveryBlocks}). Skipping contact processing.")
else:
    STARTS = list(range(0, N, N1))
    MAP_SIZE = N1
    CUTOFF = contacts_cutoff
    SAVE_DIR = "hmap_results"

    os.makedirs(SAVE_DIR, exist_ok=True)

    def filter_uris_by_dsb_time(uris, dsb_time):
        block_cutoff = dsb_time // saveEveryBlocks
        return [uri for uri in uris if int(uri.split("::")[-1]) > block_cutoff]

    def compute_and_save_hmap(traj_path, dsb_time=dsb_time):
        uris = list_URIs(traj_path)
        uris = filter_uris_by_dsb_time(uris, dsb_time)
        if not uris:
            print(f"No URIs to process in {traj_path} after DSB time.")
            return None

        hmap = monomerResolutionContactMapSubchains(
            filenames=uris,
            mapStarts=STARTS,
            mapN=MAP_SIZE,
            cutoff=CUTOFF,
            n=1,
            loadFunction=lambda x: load_URI(x)["pos"]
        )
        np.save(os.path.join(SAVE_DIR, f"hmap_{os.path.basename(traj_path)}.npy"), hmap)
        return hmap

    def main():
        paths = [f"trajectory{cycle}" for cycle in simulation_cycles]

        hmaps = []
        for path in paths:
            hmap = compute_and_save_hmap(path)
            if hmap is not None:
                hmaps.append(hmap)

        if hmaps:
            mean_hmap = np.mean(np.stack(hmaps), axis=0)
            np.save(os.path.join(SAVE_DIR, "mean_hmap.npy"), mean_hmap)
        else:
            print("No hmaps were generated; mean hmap was not saved.")

    if __name__ == "__main__":
        main()