n_process_cores = 4

N_strand = "two" # This should be left as "two", with one chromatid simulations being used for "one", as this version is not designed for use as a single-chromatid simulation anymore. String defining number of strands, either "one" or "two".

N1 = 4000 # size of one strand of chromatin 
M = 50 # repeat system M times

LIFETIME = 60 # equivalent to one half of processivity/monomer size, as each subunit moves one monomer per timestep
SEPARATION = 120
SEPARATION_AFTER_DSB = 120 # Optional post-DSB separation representing additional LEF recruitment; set to None if this feature isn't needed
trajectoryLength = 6000

dsb_time = 1000

simulation_cycles = 1
steady_state_relax_1D = 10000

Stop_Walkoff = [0,3999] # Stops LEFs from walking off end of strand by blocking them from translocating at ends

# Define cohesion linker sites (symmetrical between strands)
cohesion_sites = [] # overridden by random TAD generation, see below

ctcf_sites = []
LIFETIME_CTCF = 4 * LIFETIME

dsb_site = 2000 # Hyper-effective TAD boundary, defines right side of DSB

# Define DSB site (only second value needed (so 300 for a DSB site between 299 and 300))
dsb_ends_connected = True # if you don't want DSB ends to separate

cohesion_recruitment = True # cohesive cohesin is recruited to the DSB site to facilitate cohesion, mediating the trans search process

kbp_offset_dsb_cohesion = 4
p_bond = 1 # generic probability for a bond to form between the damaged and template strand by cohesion recruitment at the DSB site
distance_threshold = 2 # threshold for a trans contact to form

# Homology search
homology_binding = True
homology_capture_probability = 1.0 # Set 1
homology_site = 6000 # Optional for homology search features; in this case, N1 represents the full two-strand system (so twice the value of N1 as defined above). As with the DSB site, only list the right side here.

# Boundary element strengths
B_cohesion = 0.5
B_ctcf = 0.5

# Random TAD generation
use_random_tads = True
num_tads_per_strand = 8
min_spacing = 0
