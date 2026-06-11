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
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
import h5py 
import glob

from parameters import cpu_num, N1, M, LIFETIME, SEPARATION, SEPARATION_AFTER_DSB, trajectoryLength, simulation_cycles, steady_state_relax_1D, dsb_time, delay_before_FPT, ctcf_sites, LIFETIME_CTCF, dsb_site, dsb_present, dsb_ends_connected, B_CTCF, contacts_cutoff # Import parameters

##########################################################################################################
### 1D simulation helper functions. Several functions adapted from https://github.com/open2c/polychrom ###
### note that "CTCF" is a flag used for all boundary elements, not just CTCFs                          ###
##########################################################################################################

class leg(object):
    def __init__(self, pos, attrs={"stalled":False, "captured":False, "CTCF":False}):
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
        return 1 / args["LIFETIME_CTCF"]
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


def capture(cohesin, occupied, args, CTCFSites):
    for side in [1, -1]:
        # get probability of capture or otherwise it is 0 
        if np.random.random() < args["Capture"][side].get(cohesin[side].pos, 0):  
            cohesin[side].attrs["captured"] = True  # captured a cohesin at BE    
            if cohesin[side].pos in CTCFSites:
                cohesin[side].attrs["CTCF"] = True # mark if the BE was a CTCF site
    return cohesin 


def release(cohesin, occupied, args):
    
    """
    AN opposite to capture - releasing cohesins from BE 
    """
    
    if not cohesin.any("captured"):
        return cohesin  # no BE: no release necessary 
        
    # attempting to release either side 
    for side in [-1, 1]: 
        if (np.random.random() < args["Release"][side].get(cohesin[side].pos, 0)) and (cohesin[side].attrs["captured"]):
            cohesin[side].attrs["captured"] = False 
            cohesin[side].attrs["CTCF"] = False # Just in case the BE was a CTCF
    return cohesin 


def translocate(cohesins, occupied, args, CTCFSites):
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
    
    # then we try to capture and release them by BE sites 
    for i in range(len(cohesins)):
        cohesins[i] = capture(cohesins[i], occupied, args, CTCFSites)
        cohesins[i] = release(cohesins[i], occupied, args)
    
    # finally we translocate, and mark stalled cohesins because 
    # the unloadProb needs this 
    for i in range(len(cohesins)):
        cohesin = cohesins[i] 
        for leg in [-1,1]: 
            if not cohesin[leg].attrs["captured"]: 
                # cohesins that are not at BEs and cannot move are labeled as stalled 
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
        if attrs["captured"]:
            return 3
        return 1
    ar = np.zeros(args["N"])
    for i in cohesins:
        ar[i.left.pos] = state(i.left.attrs)
        ar[i.right.pos] = state(i.right.attrs)  
    return ar 

#############
### setup ###
#############

N = N1 * M
LEFNum = N // SEPARATION # Initial LEF count

if SEPARATION_AFTER_DSB:
    post_dsb_LEFnum = N // SEPARATION_AFTER_DSB
    total_LEFNum = post_dsb_LEFnum
else:
    total_LEFNum = LEFNum

CTCFs = ctcf_sites

dsb_sites = dsb_site # ex: [549,550, 1149,1150, 1849,1850, 2649,2650, 3149,3150]

CTCFSites = []

ctcfLeftRelease = {}
ctcfRightRelease = {}
ctcfLeftCapture = {}
ctcfRightCapture = {}

for i in range(M):
    # CTCFs in pre-DSB regime
    for CTCF in CTCFs:
        ctcf_pos = i * N1 + CTCF 
        CTCFSites.append(ctcf_pos)
        ctcfLeftCapture[ctcf_pos] = B_CTCF  # 50% capture probability 
        ctcfLeftRelease[ctcf_pos] = 0.0000  # hold it for ~5000 blocks on average
        ctcfRightCapture[ctcf_pos] = B_CTCF
        ctcfRightRelease[ctcf_pos] = 0.0000

args = {}
args["Release"] = {-1:ctcfLeftRelease, 1:ctcfRightRelease}
args["Capture"] = {-1:ctcfLeftCapture, 1:ctcfRightCapture}        
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_CTCF"] = LIFETIME_CTCF # Change in lifetime when at CTCF (not DSB) position
args["LIFETIME_STALLED"] = LIFETIME  # change in lifetime when stalled 
    
# Positions of DSBs and CTCFs and capture probabilities for post-DSB regime.
dsbLeftCapture = copy.deepcopy(ctcfLeftCapture)
dsbLeftRelease = copy.deepcopy(ctcfLeftRelease)
dsbRightCapture = copy.deepcopy(ctcfRightCapture)
dsbRightRelease = copy.deepcopy(ctcfRightRelease)
if dsb_present:
    DSBs = [dsb_site - 1, dsb_site]
    for i in range(M):
        for dsb in DSBs:
            dsb_pos = i * N1 + dsb 
            dsbLeftCapture[dsb_pos] = 1.
            dsbLeftRelease[dsb_pos] = 0.
            dsbRightCapture[dsb_pos] = 1.
            dsbRightRelease[dsb_pos] = 0.

##################################################
### helper functions for 3D molecular dynamics ###
##################################################

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

def run_cycle(cycle, args, CTCFSites,
              N, N1, M, LEFNum, total_LEFNum,
              steady_state_relax_1D, trajectoryLength,
              dsb_time, dsbLeftRelease, dsbRightRelease,
              dsbLeftCapture, dsbRightCapture,
              post_dsb_LEFnum, SEPARATION_AFTER_DSB,
              dsb_ends_connected, dsb_present, dsb_site,
              delay_before_FPT, contacts_cutoff):

    save_folder_name = f"trajectory{cycle}"
    os.makedirs(save_folder_name, exist_ok=True)

    # 1D LEF dynamics
    occupied = np.zeros(N)
    occupied[0] = 1
    occupied[-1] = 1
    cohesins = []

    for i in range(LEFNum):
        loadOne(cohesins, occupied, args)

    for _ in range(steady_state_relax_1D):
        translocate(cohesins, occupied, args, CTCFSites)

    #################################################################
    ### Save 1D file with LEF positions throughout the simulation ###
    #################################################################

    with h5py.File(f"{save_folder_name}/LEFPositions.h5", mode='w') as myfile:
        dset = myfile.create_dataset(
            "positions",
            shape=(trajectoryLength, total_LEFNum, 2),
            dtype=np.float32,
            compression="gzip"
        )
        steps = 50
        assert (dsb_time % steps) == 0
        pre_dsb_bins = np.linspace(0, dsb_time, steps, dtype=int)

        for st, end in zip(pre_dsb_bins[:-1], pre_dsb_bins[1:]):
            cur = []
            for i in range(st, end):
                translocate(cohesins, occupied, args, CTCFSites)
                positions = [(c.left.pos, c.right.pos) for c in cohesins]
                if SEPARATION_AFTER_DSB:
                    positions = positions + [(np.nan, np.nan)] * (post_dsb_LEFnum - LEFNum)
                cur.append(positions)
            dset[st:end] = np.array(cur)

        # Switch args to include DSBs
        args["Release"] = {-1: dsbLeftRelease, 1: dsbRightRelease}
        args["Capture"] = {-1: dsbLeftCapture, 1: dsbRightCapture}

        if SEPARATION_AFTER_DSB:
            for _ in range(post_dsb_LEFnum - LEFNum):
                loadOne(cohesins, occupied, args)

        post_dsb_bins = np.linspace(dsb_time, trajectoryLength, steps, dtype=int)
        for st, end in zip(post_dsb_bins[:-1], post_dsb_bins[1:]):
            cur = []
            for i in range(st, end):
                translocate(cohesins, occupied, args, CTCFSites)
                positions = [(c.left.pos, c.right.pos) for c in cohesins]
                cur.append(positions)
            dset[st:end] = np.array(cur)

        myfile.attrs["N1"] = N1
        myfile.attrs["M"] = M
        myfile.attrs["N"] = N
        myfile.attrs["LEFNum"] = LEFNum

    ###########################
    ### 3D Simulation setup ###
    ###########################

    folder = save_folder_name
    myfile = h5py.File(f"{folder}/LEFPositions.h5", mode='r')
    N1 = myfile.attrs["N1"]
    M = myfile.attrs["M"]
    N = myfile.attrs["N"]
    LEFNum = myfile.attrs["LEFNum"]
    LEFpositions = myfile["positions"]

    steps = 750
    stiff = 1
    dens = 0.1
    box = (N / dens) ** 0.33
    data = grow_cubic(N, int(box) - 2)
    block = 0

    saveEveryBlocks = 100
    restartSimulationEveryBlocks = 100

    dsbTime = dsb_time
    dsbIteration = dsbTime / restartSimulationEveryBlocks

    smcBondWiggleDist = 0.2
    smcBondDist = 0.5

    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal = (trajectoryLength) // restartSimulationEveryBlocks

    # DSB setup
    dsb_extra_bonds = []
    dsb_polymer_chains = []
    breakpoints = []
    if not dsb_ends_connected and dsb_present and dsb_site > 0:
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
        dsb_polymer_chains = [(0, None, 0)]

    milker = bondUpdater(LEFpositions)
    reporter = HDF5Reporter(folder=folder, max_data_length=100,
                            overwrite=True, blocks_only=False)

    fpt_reached = {
        'left': np.full((M, N1), False),
        'right': np.full((M, N1), False)
    }
    first_pass_time = {
        'left': np.full((M, N1), np.inf),
        'right': np.full((M, N1), np.inf)
    }

    ###############################
    ### 3D simulation main loop ###
    ###############################

    for iteration in range(simInitsTotal):
        a = Simulation(
            platform="CUDA",
            integrator="variableLangevin",
            error_tol=0.01,
            GPU="0",
            collision_rate=0.03,
            N=len(data),
            reporters=[reporter],
            PBCbox=[box, box, box],
            precision="mixed"
        )
        a.set_data(data)

        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=dsb_polymer_chains,
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={'bondLength': 1.0,
                                   'bondWiggleDistance': 0.05},
                angle_force_func=forces.angle_force,
                angle_force_kwargs={'k': 1.5},
                nonbonded_force_func=forces.polynomial_repulsive,
                nonbonded_force_kwargs={'trunc': 3.0},
                except_bonds=True,
                extra_bonds=dsb_extra_bonds if iteration < dsbIteration else None
            )
        )

        # Initialize milker
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale
        milker.setParams({"length": bondDist, "k": kbond},
                         {"length": bondDist, "k": 0})
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                     blocks=restartSimulationEveryBlocks)

        if iteration == 0:
            a.local_energy_minimization()
        else:
            a._apply_forces()

        for i in range(restartSimulationEveryBlocks):
            if i % saveEveryBlocks == (saveEveryBlocks - 1):
                a.do_block(steps=steps)
            else:
                a.integrator.step(steps)

            if i + iteration * restartSimulationEveryBlocks >= dsb_time + delay_before_FPT: # check for first passage and record FPTs
                pos = a.get_data()
                for j in range(M):
                    offset = j * N1
                    left_dsb_pos = pos[offset + dsb_site - 1]
                    right_dsb_pos = pos[offset + dsb_site]

                    for mon in range(N1):
                        mon_pos = pos[offset + mon]
                        if not fpt_reached['left'][j, mon]:
                            dist = np.linalg.norm(mon_pos - left_dsb_pos)
                            if dist < contacts_cutoff:
                                fpt_reached['left'][j, mon] = True
                                first_pass_time['left'][j, mon] = iteration * restartSimulationEveryBlocks + i
                        if not fpt_reached['right'][j, mon]:
                            dist = np.linalg.norm(mon_pos - right_dsb_pos)
                            if dist < contacts_cutoff:
                                fpt_reached['right'][j, mon] = True
                                first_pass_time['right'][j, mon] = iteration * restartSimulationEveryBlocks + i

            if i < restartSimulationEveryBlocks - 1:
                milker.step(a.context)

        data = a.get_data()
        del a
        reporter.blocks_only = True
        print(f"[Cycle {cycle}] Iteration {iteration + 1}/{simInitsTotal} done")
        time.sleep(0.2)

    reporter.dump_data()
    os.makedirs(f"{save_folder_name}/fpt_output", exist_ok=True)
    np.save(f"{save_folder_name}/fpt_output/first_pass_time_left.npy", first_pass_time['left'])
    np.save(f"{save_folder_name}/fpt_output/first_pass_time_right.npy", first_pass_time['right'])
    np.save(f"{save_folder_name}/fpt_output/fpt_reached_left.npy", fpt_reached['left'])
    np.save(f"{save_folder_name}/fpt_output/fpt_reached_right.npy", fpt_reached['right'])

    return cycle  

# Main multiprocess driver 
def run_all(simulation_cycles,
            N, N1, M,
            LEFNum, total_LEFNum,
            steady_state_relax_1D,
            trajectoryLength,
            dsb_time, dsbLeftRelease, dsbRightRelease,
            dsbLeftCapture, dsbRightCapture,
            post_dsb_LEFnum,
            SEPARATION_AFTER_DSB,
            dsb_ends_connected, dsb_present, dsb_site,
            delay_before_FPT,
            contacts_cutoff):
    
    with ProcessPoolExecutor(max_workers=simulation_cycles) as executor:
        futures = [
            executor.submit(
                run_cycle,
                cycle, args, CTCFSites,
                N, N1, M, LEFNum, total_LEFNum,
                steady_state_relax_1D, trajectoryLength,
                dsb_time, dsbLeftRelease, dsbRightRelease,
                dsbLeftCapture, dsbRightCapture,
                post_dsb_LEFnum, SEPARATION_AFTER_DSB,
                dsb_ends_connected, dsb_present, dsb_site,
                delay_before_FPT, contacts_cutoff
                )
            for cycle in range(simulation_cycles)
        ]
        for f in as_completed(futures):
            print(f"Cycle {f.result()} finished.")

if __name__ == "__main__":
    run_all(simulation_cycles,
            N, N1, M,
            LEFNum, total_LEFNum,
            steady_state_relax_1D,
            trajectoryLength,
            dsb_time, dsbLeftRelease, dsbRightRelease,
            dsbLeftCapture, dsbRightCapture,
            post_dsb_LEFnum,
            SEPARATION_AFTER_DSB,
            dsb_ends_connected, dsb_present, dsb_site,
            delay_before_FPT,
            contacts_cutoff)