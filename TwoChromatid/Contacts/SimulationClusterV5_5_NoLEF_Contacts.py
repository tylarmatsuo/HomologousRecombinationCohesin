import pickle
import os
import time
import numpy as np
import pandas as pd
import polychrom
import copy
import multiprocessing as mp

from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file

import simtk.openmm 
import os 
import shutil

import random
import warnings
import h5py 
import glob

from scipy.stats import binom

from parameters import n_process_cores, N_strand, N1, M, LIFETIME, SEPARATION, SEPARATION_AFTER_DSB, trajectoryLength, simulation_cycles, steady_state_relax_1D, dsb_time, ctcf_sites, cohesion_sites, Stop_Walkoff, LIFETIME_CTCF, dsb_site, dsb_ends_connected, cohesion_recruitment, kbp_offset_dsb_cohesion, homology_binding, homology_site, p_bond, distance_threshold, B_cohesion, B_ctcf, use_random_tads, num_tads_per_strand, min_spacing

#############################################################################
### Parameters                                                            ###
### Only N_strand=="two" is currently fully supported                     ###
###                                                                       ###
### Because that interactions between the DSB site and sister chromatid   ###
### control cohesive clamp recruitment, and cohesive clamps interact with ###
### LEFs, this code cannot fully precalculate LEF trajectories like in    ###
### single-chromatid simulations.                                         ###
#############################################################################

if N_strand == "two": # Two Strand

    N = N1 * M * 2 # multiplied by 2 for two strands

    LEFNum = 0 # Initial LEF count

    if SEPARATION_AFTER_DSB:
        post_dsb_LEFnum = 0
        total_LEFNum = post_dsb_LEFnum
    else:
        total_LEFNum = LEFNum

    steps = 750   # MD steps per step of cohesin
    stiff = 1
    dens = 0.1
    box = (N / dens) ** 0.33  # density = 0.1.
    block = 0  # starting block 

    # new parameters because some things changed 
    saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
    restartSimulationEveryBlocks = 100

    dsbTime = dsb_time # Block at which DSBs are introduced (adjust as needed)
    dsbIteration = dsbTime/restartSimulationEveryBlocks # Iteration at which DSBs are introduced

    # parameters for smc bonds
    smcBondWiggleDist = 0.2
    smcBondDist = 0.5

    # assertions for easy managing code below 
    assert (trajectoryLength % restartSimulationEveryBlocks) == 0 
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal  = (trajectoryLength) // restartSimulationEveryBlocks

### Parameter Set for One Strand Simulation
if N_strand == "one":

    N = N1 * M

    LEFNum = 0 # Initial LEF count

    if SEPARATION_AFTER_DSB:
        post_dsb_LEFnum = 0
        total_LEFNum = post_dsb_LEFnum
    else:
        total_LEFNum = LEFNum

    steps = 750   # MD steps per step of cohesin
    stiff = 1
    dens = 0.1
    box = (N / dens) ** 0.33  # density = 0.1.
    block = 0  # starting block 

    # new parameters because some things changed 
    saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
    restartSimulationEveryBlocks = 100

    dsbTime = dsb_time # Block at which DSBs are introduced (adjust as needed)
    dsbIteration = dsbTime/restartSimulationEveryBlocks # Iteration at which DSBs are introduced

    # parameters for smc bonds
    smcBondWiggleDist = 0.2
    smcBondDist = 0.5

    # assertions for easy managing code below 
    assert (trajectoryLength % restartSimulationEveryBlocks) == 0 
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0

    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal  = (trajectoryLength) // restartSimulationEveryBlocks


#########################################################
### Initial Setup                                     ###
#########################################################

### 1D ###
def generate_random_tads(N1, num_tads, min_spacing):
    attempts = 0
    while attempts < 1000:
        tads = sorted(np.random.choice(np.arange(min_spacing, N1 - min_spacing), size=num_tads, replace=False))
        if all(np.diff(tads) >= min_spacing):
            return tads
        attempts += 1
    raise RuntimeError("Couldn't generate TADs with required spacing")

if use_random_tads:
    # M sets of cohesion sites, one per strand *pair*
    pair_cohesion_sites = [
        generate_random_tads(N1, num_tads_per_strand, min_spacing)
        for _ in range(M)
    ]
else:
    pair_cohesion_sites = [cohesion_sites for _ in range(M)]

LeftRelease = {}
RightRelease = {}
LeftCapture = {}
RightCapture = {}

for i in range(M * 2): # multiplied by 2 as same process applies to both strands
    pair_index = i // 2
    for cohesion in pair_cohesion_sites[pair_index]:
        pos = i * N1 + cohesion
        LeftCapture[pos] = B_cohesion  
        LeftRelease[pos] = 0  # set to 0 as no literature found yet giving proper value
        RightCapture[pos] = B_cohesion
        RightRelease[pos] = 0
    
    for ctcf in ctcf_sites:
        pos = i * N1 + ctcf
        LeftCapture[pos] = B_ctcf  
        LeftRelease[pos] = 0
        RightCapture[pos] = B_ctcf
        RightRelease[pos] = 0

    for stop in Stop_Walkoff: # Stops LEFs from walking off ends of strands
        stop_pos = i * N1 + stop
        LeftCapture[stop_pos] = 1.0
        LeftRelease[stop_pos] = 0
        RightCapture[stop_pos] = 1.0
        RightRelease[stop_pos] = 0
       
args = {}
args["Release"] = {-1:LeftRelease, 1:RightRelease}
args["Capture"] = {-1:LeftCapture, 1:RightCapture}   
args["N1"] = N1     
args["M"] = M
args["N"] = N 
args["LIFETIME"] = LIFETIME
args["LIFETIME_CAPTURED"] = LIFETIME_CTCF # Change in lifetime when at BE
args["LIFETIME_STALLED"] = LIFETIME  # change in lifetime when stalled 

### 3D ###

# Set extra bonds (linking DSB sites pre-break) and polymer chains, as well as hard code cohesion extra bonds
dsb_strand_chains = []
breakpoints = []
cohesion_only_extra_bonds = []
dsb_extra_bonds = []

if N_strand == "two":
    for i in range(2 * M):
        breakpoints.append(i * N1)

        if i % 2 == 0:
            pair_index = i // 2
            cohesion_only_extra_bonds.extend([
                (i * N1 + site, i * N1 + site + N1)
                for site in pair_cohesion_sites[pair_index]
            ])

            if not dsb_ends_connected and dsb_site and dsb_site > 0:
                break_pos = i * N1 + dsb_site
                dsb_extra_bonds.append((break_pos - 1, break_pos))
                breakpoints.append(break_pos)

elif N_strand == "one":
    if not dsb_ends_connected and dsb_site and dsb_site > 0:
        breakpoints.append(0)
        for i in range(M):
            instance_offset = i * N1
            break_pos = instance_offset + dsb_site
            dsb_extra_bonds.append((break_pos - 1, break_pos))
            breakpoints.append(break_pos)

breakpoints = sorted(set(breakpoints))

if breakpoints:
    dsb_strand_chains = [(breakpoints[i], breakpoints[i + 1], 0)
                         for i in range(len(breakpoints) - 1)]
    dsb_strand_chains.append((breakpoints[-1], None, 0))
else:
    dsb_strand_chains = [(0, None, 0)]  # one continuous chain

all_extra_bonds = cohesion_only_extra_bonds + dsb_extra_bonds

post_dsb_extra_bonds = copy.deepcopy(cohesion_only_extra_bonds) # useful for storage of additional bonds added later

#################################################################################
### Helper Functions                                                          ###
### Several functions adapted from https://github.com/open2c/polychrom        ###
### note that "CTCF" is a flag used for all boundary elements, not just CTCFs ###
#################################################################################

class leg(object):
    def __init__(self, pos, attrs={"stalled":False, "CTCF":False}):
        """
        A leg has two important attribues: pos (positions) and attrs (a custom list of attributes)
        """
        self.pos = pos
        self.attrs = dict(attrs)

class cohesive_cohesin(object):
    def __init__(self, leg1, leg2):
        self.damaged = leg1
        self.template = leg2
   
    def any(self, attr):
        return self.damaged.attrs[attr] or self.template.attrs[attr]
    
    def all(self, attr):
        return self.damaged.attrs[attr] and self.template.attrs[attr]    
    
    def __getitem__(self, item):
        if item == -1:
            return self.damaged
        elif item == 1:
            return self.template 
        else:
            raise ValueError()

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
    Defines unload probability based on a state of cohesin 
    """
    if cohesin.any("CTCF"):
        return 1 / args["LIFETIME_CAPTURED"]
    elif cohesin.any("stalled"):
        return 1 / args["LIFETIME_STALLED"]
    return 1 / args["LIFETIME"]  

def generateLoadingPositions(N, N1, LEFNum, LEFLifetime, blocks,
                              confidence=0.99, N_strand=N_strand, return_all=False):
    """
    Generate precomputed LEF loading sites for one interval.
    """
    expected_loads = blocks * LEFNum / LEFLifetime

    # Probability of a site being occupied
    p_occ = 2 * LEFNum / N
    p_valid = (1 - p_occ) ** 2

    # Estimate how many positions to draw to have `expected_loads` successes
    max_trials = int(1.5 * expected_loads) 
    num_trials = int(np.ceil(binom.ppf(confidence, max_trials, p_valid)))

    # Generate trial positions
    max_left_index = N - 1 
    candidates = np.random.randint(0, max_left_index, size=num_trials)

    # Remove straddling pairs
    if N_strand == "two":
        left = candidates
        right = candidates + 1
        invalid = (left // N1) != (right // N1)
        final = list(candidates[~invalid])
    else:
        final = list(candidates)

    if return_all:
        return final, {
            "expected_loads": expected_loads,
            "p_occ": p_occ,
            "p_valid": p_valid,
            "num_trials": num_trials,
            "yielded": len(final),
        }
    return final

    
def loadOne(cohesins, occupied, args): 
    """
    A function to load one cohesin 
    """
    while True:
        a = np.random.randint(args["N"])
        if (occupied[a] == 0) and (occupied[a+1] == 0):
            if N_strand == "two":
                if a // N1 == (a + 1) // N1:
                    occupied[a] = 1
                    occupied[a+1] = 1 
                    cohesins.append(cohesin(leg(a), leg(a+1)))
                    break
            elif N_strand == "one":
                occupied[a] = 1
                occupied[a+1] = 1 
                cohesins.append(cohesin(leg(a), leg(a+1)))
                break
            else:
                raise ValueError('N_strand was not defined so the loadOne function does not know what to do.')
            
def loadOneFromList(cohesins, occupied, load_locations, LEFDeficit):
    '''
    Load a cohesin onto a site from a pre-defined list, which may be randomly generated
    '''
    idx = 0
    while idx < len(load_locations):
        a = load_locations[idx]
        if (occupied[a] == 0) and (occupied[a+1] == 0):
            if N_strand == "two" and a // N1 != (a + 1) // N1:
                idx += 1
                continue
            occupied[a] = 1
            occupied[a+1] = 1 
            cohesins.append(cohesin(leg(int(a)), leg(int(a+1))))
            load_locations.pop(idx)
            return load_locations, LEFDeficit  # success
        else:
            idx += 1

    LEFDeficit += 1 # if all locations fail or none are left
    return load_locations, LEFDeficit

def loadOneOnSpecificStrand(cohesins, occupied, args, strand=0): 
    """
    A function to load one cohesin on a specific chromatid

    chromatid input is an integer, with 0 specifying the primary (broken) chromatid and 1 the secondary
    """
    N1 = args["N1"]
    M = args["M"]
    ranges = [(i * 2 * N1, i * 2 * N1 + N1) for i in range(M)]
    
    if strand == 1:
        ranges = [(start + N1, end + N1) for start, end in ranges]
    
    valid_positions = []
    for start, end in ranges:
        valid_positions.extend(range(start, end - 1))  # Exclude the last position as it pairs with the next
    
    while True: # While loop ensures function will not stop running until valid location to load cohesin is found and cohesin is loaded
        a = np.random.choice(valid_positions)
        if (occupied[a] == 0) and (occupied[a + 1] == 0):
            occupied[a] = 1
            occupied[a + 1] = 1
            cohesins.append(cohesin(leg(a), leg(a + 1)))
            break


def capture(cohesin, occupied, args):
    for side in [1, -1]:
        # get probability of capture or otherwise it is 0 
        if np.random.random() < args["Capture"][side].get(cohesin[side].pos, 0):  
            cohesin[side].attrs["CTCF"] = True  # captured a cohesin at CTCF     
    return cohesin 


def release(cohesin, occupied, args):
    if not cohesin.any("CTCF"):
        return cohesin  # no boundary element: no release necessary 
        
    # attempting to release either side 
    for side in [-1, 1]: 
        if (np.random.random() < args["Release"][side].get(cohesin[side].pos, 0)) and (cohesin[side].attrs["CTCF"]):
            cohesin[side].attrs["CTCF"] = False 
    return cohesin 

def translocate(cohesins, cohesions, occupied, load_locations, use_load_locations, LEFDeficit, N, args):
    """
    This function describes everything that happens with cohesins - 
    loading/unloading them and stalling against each other 
    
    It relies on the functions defined above: unload probability, capture/release. 

    LEFs may be located on either chromatid
    """
    # first we try to unload cohesins and free the matching occupied sites 
    cohesins_to_delete = []

    for i in range(len(cohesins)):
        prob = unloadProb(cohesins[i], args)
        if np.random.random() < prob:
            occupied[cohesins[i].left.pos] = 0 
            occupied[cohesins[i].right.pos] = 0 
            cohesins_to_delete.append(i)
            if use_load_locations:
                load_locations, LEFDeficit = loadOneFromList(cohesins, occupied, load_locations=load_locations, LEFDeficit=LEFDeficit)
            else:
                loadOne(cohesins=cohesins, occupied=occupied, args=args)
    
    for i in sorted(cohesins_to_delete, reverse=True):
        if i < len(cohesins):
            del cohesins[i]
    
    # then we try to capture and release them by CTCF sites 
    for i in range(len(cohesins)):
        cohesins[i] = capture(cohesins[i], occupied, args)
        cohesins[i] = release(cohesins[i], occupied, args)

    if False: # Disabled and replaced by extrinsic motor only because actual diffusion rate of cohesive cohesin is very low
        # translocate cohesive cohesions on template strand by random walk process
        for i in range(len(cohesions)):
            cohesion = cohesions[i]
            direction = np.random.choice([-1, 1])
            destination = cohesion[1].pos + direction
            if np.random.random() < args["Capture"][direction].get(cohesion[1].pos, 0):
                continue # Do nothing, stopped by CTCF during this random walk step
            elif occupied[destination] != 0 or destination >= N or destination <= 0:
                continue # Do nothing, stalled this random walk step
            else:
                occupied[cohesion[1].pos] = 0
                occupied[destination] = 1
                cohesion[1].pos = destination
            cohesions[i] = cohesion
    
    for i in range(len(cohesions)): # This method checks both the damaged and template chromatids, however the extra damaged chromatid check shouldn't do any harm
        cohesions[i] = capture(cohesions[i], occupied, args)
        cohesions[i] = release(cohesions[i], occupied, args)
        
    
    # finally we translocate, and mark stalled cohesins because 
    # the unloadProb needs this 
    def try_push_cohesion(cohesion, leg, occupied, N, cohesions):
        """
        Attempt to push cohesive cohesin mediated cohesion one step in leg direction.
        Returns True if this cohesive cohesin (and any chain ahead of it) moved; False if blocked.
        """
        node = cohesion[1]
        pos = node.pos
        nxt = pos + leg

        # blocked by boundary element or off-chromatin
        if node.attrs.get("CTCF", False):
            return False
        if nxt < 0 or nxt >= N:
            return False

        # occupied: can only push if the occupier is another cohesive cohesin we can push
        if occupied[nxt] != 0:
            occupant = None
            for other in cohesions:
                if other is cohesion:
                    continue
                if other[1].pos == nxt:
                    occupant = other
                    break
            if occupant is None:
                return False
            # try to push the occupant first; if that succeeds, move this cohesive cohesin too
            if not try_push_cohesion(occupant, leg, occupied, N, cohesions):
                return False

        # move this cohesive cohesin into nxt
        occupied[pos] = 0
        occupied[nxt] = 1
        node.pos = nxt
        return True

    for i, cohesin in enumerate(cohesins):
        for leg in (-1, 1):
            leg_node = cohesin[leg]

            # skip if at a boundary element
            if leg_node.attrs.get("CTCF", False):
                continue

            cur = leg_node.pos
            nxt = cur + leg

            # don't walk off ends
            if nxt < 0 or nxt >= N:
                continue

            if occupied[nxt] == 0:
                # free step
                leg_node.attrs["stalled"] = False
                occupied[cur] = 0
                occupied[nxt] = 1
                leg_node.pos = nxt
            else:
                # if occupied, check if it's a cohesive clamp we can push
                pushed = False
                for cohesion in cohesions:
                    if cohesion[1].pos == nxt:
                        # if the cohesive clamp itself is blocked by a CTCF or boundary, it can't be pushed
                        if cohesion[1].attrs.get("CTCF", False):
                            pushed = False
                        else:
                            if try_push_cohesion(cohesion, leg, occupied, N, cohesions):
                                # now move the cohesin into the freed spot
                                leg_node.attrs["stalled"] = False
                                occupied[cur] = 0
                                occupied[nxt] = 1
                                leg_node.pos = nxt
                                pushed = True
                        break

                if not pushed:
                    leg_node.attrs["stalled"] = True

        cohesins[i] = cohesin
    
    return load_locations, LEFDeficit

### 3D Helper Functions ###

def add_dsb_homology_bonds(current_bonds, positions, dsb_location=dsb_site, homology_location=homology_site,
                           N2=N1*2, M=M, distance_threshold=6, p_bond=1):
    '''For DSB-homology binding'''

    new_bonds = []

    for strand_pair in range(M):
        for side in ["left", "right"]:
            offset = - 1 if side == "left" else 0 # Subtract 1 on left side because the dsb location is given as the right monomer index. You can also add additional offsets here if desired.

            try:
                dsb_index = strand_pair * N2 + dsb_location + offset
                hom_index = strand_pair * N2 + homology_location + offset
                dsb_pos = positions[dsb_index]
                hom_pos = positions[hom_index]
            except IndexError:
                continue  # Out of bounds

            dist = np.linalg.norm(dsb_pos - hom_pos)
            if dist < distance_threshold and random.random() < p_bond:
                bond = (min(dsb_index, hom_index), max(dsb_index, hom_index))
                if bond not in current_bonds:
                    new_bonds.append(bond)

    return new_bonds

def add_dsb_template_bonds(current_bonds, positions, cohesions, occupied, dsb_location=dsb_site,
                           N2=N1*2, M=M, distance_threshold=2, p_bond=1, kbp_offset=0, banned_M = []):
    '''Used for cohesive clamp recruitment'''

    for strand_pair in range(M):
        if strand_pair in banned_M:
            continue
        for side in ["left", "right"]:
            offset = -kbp_offset - 1 if side == "left" else kbp_offset # Subtract 1 on left side because the dsb location is given as the right monomer index

            dsb_index = strand_pair * N2 + dsb_location + offset
            if occupied[dsb_index] != 0: # Only one cohesin or cohesive cohesin at a single site
                continue

            dsb_pos = positions[dsb_index]

            candidate_indices = list(range(N1))
            random.shuffle(candidate_indices)

            for idx in candidate_indices:
                template_idx = strand_pair * N2 + idx + N1
                if occupied[template_idx] != 0: # Can not bind on top of a cohesin/cohesive cohesion
                    continue
                template_pos = positions[template_idx]
                dist = np.linalg.norm(dsb_pos - template_pos)
                if dist < distance_threshold and random.random() < p_bond:
                    bond = (min(dsb_index, template_idx), max(dsb_index, template_idx))
                    if bond not in current_bonds:
                        occupied[dsb_index] = 1
                        occupied[template_idx] = 1 
                        cohesions.append(cohesive_cohesin(leg(dsb_index), leg(template_idx)))
                        break

    return cohesions

### for multiprocessing to predict bonds that may need to be activated ###
def generate_all_lef_bonds_strand(strand_start, N1, max_steps, N_strand="two", total_length=N):
    """
    Generates LEF bonds which might form during the simulation for use in initialization.
    
    Parameters:
        strand_start (int): Starting index of the strand
        N1 (int): Length of one polymer instance
        max_steps (int): Max growth (in bonds) from any starting site
        N_strand (str): "one" for continuous polymer, "two" for per-strand LEFs
        total_length (int): Needed if N_strand == "one"
        
    Returns:
        set: Set of (left, right) tuples representing possible bonds
    """
    strand_bonds = set()

    if N_strand == "two":
        strand_end = strand_start + N1
        index_range = range(strand_start, strand_end - 1)
        max_right = strand_end
    elif N_strand == "one":
        strand_start = 0
        strand_end = total_length
        index_range = range(strand_start, strand_end - 1)
        max_right = total_length
    else:
        raise ValueError(f"Invalid N_strand: {N_strand}")

    for i in index_range:
        for l in range(max_steps + 1):
            left = i - l
            if left < strand_start:
                break
            for r in range(max_steps + 1):
                right = i + 1 + r
                if right >= max_right:
                    break
                strand_bonds.add((left, right))

    return strand_bonds

def predict_bonds_window(left, right, N1, max_steps, N, N_strand):
    bonds = set()

    if N_strand == "one":
        min_left = max(0, left - max_steps)
        max_right = min(N - 1, right + max_steps)
    elif N_strand == "two":
        strand_start = (left // N1) * N1
        strand_end = strand_start + N1
        min_left = max(strand_start, left - max_steps)
        max_right = min(strand_end - 1, right + max_steps)
    else:
        return bonds

    for l in range(min_left, left + 1):
        for r in range(right, max_right + 1):
            bonds.add((l, r))
    return bonds

def predict_bonds_from_sites_list(candidate_list, N1, max_steps, N, N_strand, n_process=1):
    with mp.Pool(n_process) as pool:
        results = pool.starmap(
            predict_bonds_window,
            [(left, left + 1, N1, max_steps, N, N_strand) for left in candidate_list]
        )
    return set().union(*results)

def estimate_bonds_from_existing_lefs(lef_positions, N1, max_steps, N, N_strand, n_process=1):
    with mp.Pool(n_process) as pool:
        results = pool.starmap(
            predict_bonds_window,
            [(left, right, N1, max_steps, N, N_strand) for (left, right) in lef_positions]
        )
    return set().union(*results)



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

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict

    def setup(self, bondForce,  candidate_LEF_loading_sites, use_candidate_LEF_sites=False, blocks=100, smcStepsPerBlock=1, N1=N1, N2=N1*2, M=M, kbp_offset=0, n_process=1, iteration=None):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block'
        :homology_binding: homology binding?
        :N2: system size
        :M: number of systems in simulation
        :iteration: simulation iteration
        :return:
        """


        if False: #len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        loaded_positions = self.LEFpositions
        self.bondForce = bondForce
        #allBonds = [[(int(loaded_positions[j][0]), int(loaded_positions[j][1])) for j in range(len(loaded_positions))]]
        existing_LEFs = estimate_bonds_from_existing_lefs(loaded_positions, N1, blocks, N=N, N_strand=N_strand, n_process=n_process)

        if use_candidate_LEF_sites:
            future_LEFs = predict_bonds_from_sites_list(candidate_LEF_loading_sites, N1, blocks, N=N, N_strand=N_strand, n_process=n_process)
        else:
            args = [
                (strand_id * N1, N1, blocks, N_strand, N)
                for strand_id in range(2 * M)
            ]

            with mp.Pool(n_process) as pool:
                results = pool.starmap(generate_all_lef_bonds_strand, args)

            future_LEFs = set().union(*results)

        all_possible_bonds = existing_LEFs | future_LEFs
        
        # precalculate trans bonds
        potential_trans_bonds = set()
        if iteration > dsbIteration:
            for strand_pair in range(M):
                if cohesion_recruitment:
                    for offset in [-kbp_offset - 1, kbp_offset]:
                        dsb_index_offset = strand_pair * N2 + dsb_site + offset
                        for i in range(N1):
                            trans_index = strand_pair * N2 + N1 + i
                            bond = (min(dsb_index_offset, trans_index), max(dsb_index_offset, trans_index))
                            if bond not in post_dsb_extra_bonds: # in case it was already there for a homology bond
                                potential_trans_bonds.add(bond)
                if homology_binding:
                    # For homology bonds without offset
                    for side in [- 1, 0]:
                        dsb_index = strand_pair * N2 + dsb_site + side
                        hom_index = strand_pair * N2 + homology_site + side
                        bond = (min(dsb_index, hom_index), max(dsb_index, hom_index))
                        if bond not in post_dsb_extra_bonds: # in case it was already there for a homology bond
                            potential_trans_bonds.add(bond)

        # Initialize full unique bond list
        active_bonds = all_possible_bonds
        inactive_trans_bonds = potential_trans_bonds - active_bonds

        self.allBonds = list(all_possible_bonds)
        self.uniqueBonds = list(active_bonds | inactive_trans_bonds)

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = loaded_positions
        self.bondToInd = {}

        for bond in self.uniqueBonds:
            if bond in self.curBonds:
                paramset = self.activeParamDict
            else:
                paramset = self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset)
            self.bondInds.append(ind)
            self.bondToInd[bond] = ind

        self.curtime += blocks 
        
        return self.curBonds,[]

    def step(self, context, homology_binding=False, positions=None, args=None, iteration=None, distance_threshold=6, p_bond=1, kbp_offset=0, cohesions=[], banned_M=[], verbose=False):
        """
        Update the bonds to the next step.
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        Contains parameters for homology search and binding, and to update extra bonds to include the new dsb-homology bond
        """
        
        ### New 5/9 Homology Search
        if homology_binding and iteration>dsbIteration:
            new_homology_bonds = add_dsb_homology_bonds(
                current_bonds=self.curBonds, 
                positions=positions, 
                distance_threshold=distance_threshold, 
                p_bond=p_bond
                )
            
            for bond in new_homology_bonds:
                if bond not in post_dsb_extra_bonds:
                    if bond not in self.bondToInd:
                        raise RuntimeError(f"Tried to activate bond {bond} that wasn't pre-added in setup!")
                    
                    args["Capture"][-1][np.max(bond)] = 1
                    args["Capture"][1][np.max(bond)] = 1
                    args["Release"][-1][np.max(bond)] = 0
                    args["Release"][1][np.max(bond)] = 0

                    this_M = np.min(bond) // (N1 * 2)

                    cohesions_to_delete = []

                    for i in range(len(cohesions)):
                        if cohesions[i].damaged.pos // (N1 * 2) == this_M:
                            occupied[cohesions[i].damaged.pos] = 0
                            occupied[cohesions[i].template.pos] = 0
                            
                            # Get rid of cohesions on same M
                            self.LEFpositions = [bond for bond in self.LEFpositions if bond != (cohesions[i].damaged.pos, cohesions[i].template.pos)]
                            cohesions_to_delete.append(i)
                            
                    for i in sorted(cohesions_to_delete, reverse=True):
                        if i < len(cohesions):
                            del cohesions[i]      

                    banned_M.append(this_M)

                    #if kbp_offset != 0:
                    #    args["Capture"][-1][np.min(bond)] = 1
                    #    args["Capture"][1][np.min(bond)] = 1
                    #    args["Release"][-1][np.min(bond)] = 0
                    #    args["Release"][1][np.min(bond)] = 0

                    post_dsb_extra_bonds.append(bond)

                    ind = self.bondToInd[bond]
                    self.bondForce.setBondParameters(ind, bond[0], bond[1], **self.activeParamDict)

                    print('New homology bond!', bond)
        
        pastBonds = self.curBonds
        self.curBonds = self.LEFpositions  # getting current bonds

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
        return self.curBonds, pastBonds, args, cohesions, banned_M
    
#################
### Main Loop ###
#################

for cycle in range(simulation_cycles):
    save_folder_name = f"trajectory{cycle}"
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)

    folder = save_folder_name

    LEFpositions = []
    occupied = np.zeros(N)
    occupied[0] = 1
    occupied[-1] = 1 
    cohesins = []
    cohesions = []
    banned_M = []
    LEFDeficit = 0

    data = grow_cubic(N, int(box) - 2)  # creates a compact conformation 

    milker = bondUpdater(LEFpositions)

    for i in range(LEFNum):
        loadOne(cohesins,occupied, args)
    for _ in range(steady_state_relax_1D):
        translocate(cohesins, cohesions, occupied, load_locations = None, use_load_locations = False, LEFDeficit=LEFDeficit, N=N, args = args)
    
    milker.LEFpositions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]

    reporter = HDF5Reporter(folder=folder, max_data_length=100, overwrite=True, blocks_only=False)

    post_dsb_extra_bonds = copy.deepcopy(cohesion_only_extra_bonds) # this variable will also track trans bonds, if applicable

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
                chains= dsb_strand_chains, #[(0, None, False)],

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

                # Remove DSB extra bonds (induce DSBs) after a set number of iterations
                extra_bonds = all_extra_bonds if iteration < dsbIteration
                else post_dsb_extra_bonds
                
            )
        )
        
        # initializing milker; adding bonds
        # copied from addBond
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        bondDist = smcBondDist * a.length_scale

        if iteration == dsbIteration:
            # Update DSBs to induce DSBs and have them act as BEs
            if dsb_site:
                DSBs = [dsb_site - 1, dsb_site]
                if N_strand == "two":
                    for i in range(M * 2): # Multiplied by two for two chromatin strands
                        if i % 2 == 0: # Don't add DSB site on even indexed segments, as those segments are on the secondary strand
                            for dsb in DSBs:
                                dsb_pos = i * N1 + dsb
                                args["Capture"][-1][dsb_pos] = 1.0
                                args["Release"][-1][dsb_pos] = 0.0
                                args["Capture"][1][dsb_pos] = 1.0
                                args["Release"][1][dsb_pos] = 0.0
                elif N_strand == "one":
                    for i in range(M):
                        for dsb in DSBs:
                                dsb_pos = i * N1 + dsb
                                args["Capture"][-1][dsb_pos] = 1.0
                                args["Release"][-1][dsb_pos] = 0.0
                                args["Capture"][1][dsb_pos] = 1.0
                                args["Release"][1][dsb_pos] = 0.0

            # If separation changes occur after the DSB due to LEF recruitment, add more LEFs to account for this
            if SEPARATION_AFTER_DSB:
                post_dsb_newLEF = post_dsb_LEFnum - LEFNum
                for _ in range(post_dsb_newLEF):
                    loadOne(cohesins,occupied, args) 
                milker.LEFpositions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
                milker.LEFpositions.extend((cohesion.damaged.pos, cohesion.template.pos) for cohesion in cohesions)

        activeParams = {"length":bondDist,"k":kbond}
        inactiveParams = {"length":bondDist, "k":0}
        milker.setParams(activeParams, inactiveParams)

        if SEPARATION_AFTER_DSB:
            if iteration < dsbIteration:
                this_iteration_LEFnum = LEFNum
            elif iteration >= dsbIteration:
                this_iteration_LEFnum = post_dsb_LEFnum
        else:
            this_iteration_LEFnum = LEFNum
        
        if LEFDeficit > 0:
            for _ in range(LEFDeficit):
                loadOne(cohesins, occupied, args)
            milker.LEFpositions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            milker.LEFpositions.extend((cohesion.damaged.pos, cohesion.template.pos) for cohesion in cohesions)
            print(LEFDeficit, "LEFs (", LEFDeficit/this_iteration_LEFnum, "percent of total LEFs) could not be loaded during the last iteration and were returned to the simulation now. If this number is a large proportion of total LEFs, candidate site list may not be long enough.")
        LEFDeficit = 0

        candidate_LEF_loading_sites = generateLoadingPositions(N, N1=N1, LEFNum=this_iteration_LEFnum, LEFLifetime=LIFETIME, N_strand=N_strand, blocks=restartSimulationEveryBlocks)
        use_from_list = len(candidate_LEF_loading_sites) < (N - 1)

        if use_from_list:
            print(f"Using loadOneFromList with {len(candidate_LEF_loading_sites)} candidate sites.")
        else:
            print("Using loadOne (full sampling).")

        time1=time.time()
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                    blocks=restartSimulationEveryBlocks, 
                    N1=N1,
                    N2=N1*2,
                    M=M,
                    iteration=iteration,
                    kbp_offset=kbp_offset_dsb_cohesion,
                    candidate_LEF_loading_sites = candidate_LEF_loading_sites,
                    use_candidate_LEF_sites = use_from_list,
                    n_process=n_process_cores
                    )
        time2=time.time()
        print("Finished setup!", len(cohesions), "cohesions are present and", len(post_dsb_extra_bonds) - len(cohesion_only_extra_bonds), "DSB-Homology bonds have been formed.")

        if iteration==0:
            a.local_energy_minimization() 
        else:
            a._apply_forces()
        
        for i in range(restartSimulationEveryBlocks):        
            # Translocate cohesins and cohesive cohesins and update their positions
            candidate_LEF_loading_sites, LEFDeficit = translocate(cohesins, cohesions, occupied, load_locations = candidate_LEF_loading_sites, use_load_locations = use_from_list, LEFDeficit=LEFDeficit, N=N, args = args)
            if cohesion_recruitment and iteration > dsbIteration:
                cohesions = add_dsb_template_bonds(current_bonds=milker.curBonds, positions=a.get_data(), cohesions=cohesions, occupied=occupied, distance_threshold=distance_threshold, p_bond=p_bond, kbp_offset=kbp_offset_dsb_cohesion, banned_M=banned_M)
            milker.LEFpositions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            milker.LEFpositions.extend((cohesion.damaged.pos, cohesion.template.pos) for cohesion in cohesions)

            curBonds, pastBonds, args, cohesions, banned_M = milker.step(a.context, homology_binding=homology_binding, positions=a.get_data(), args=args, iteration=iteration, distance_threshold=4, p_bond=1, kbp_offset=kbp_offset_dsb_cohesion, cohesions=cohesions, banned_M=banned_M, verbose=False) 
            milker.LEFpositions = [(cohesin.left.pos, cohesin.right.pos) for cohesin in cohesins]
            milker.LEFpositions.extend((cohesion.damaged.pos, cohesion.template.pos) for cohesion in cohesions)

            if i % saveEveryBlocks == (saveEveryBlocks - 1):  
                a.do_block(steps=steps)
            else:
                a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster), but note that this approach effecitvely determines how often the simulation can check for chromatin interactions for cohesive clamp recruitment, homology interaction, etc.
            
        data = a.get_data()  # save data and step, and delete the simulation
        del a
        
        reporter.blocks_only = True  # Write output hdf5-files only for blocks

        print('Simulation iteration', iteration + 1, 'of', simInitsTotal, 'is done!')

        time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)

    reporter.dump_data()