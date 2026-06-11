n_process_cores = 4

N_strand = "two" # String defining number of strands, either "one" or "two". "one" is deprecated.

N1 = 2000 # size of one strand of chromatin 
M = 5 # repeat system M times

LIFETIME = 60 # equivalent to one half of processivity/monomer size, as each subunit moves one monomer per timestep
SEPARATION = 120
SEPARATION_AFTER_DSB = 120 # Optional post-DSB separation representing extra LEF recruitment post-DSB; set to None if this feature isn't needed
trajectoryLength = 23000

dsb_time = 1000

delay_before_fpt = 30

simulation_cycles = 1
steady_state_relax_1D = 10000

Stop_Walkoff = [0,1999] # Stops LEFs from walking off end of strand by blocking them from moving

# Define cohesive cohesin linker sites (symmetrical between strands)
cohesion_sites = [] # overridden by random TAD generation

ctcf_sites = []
LIFETIME_CTCF = 4 * LIFETIME # this also applies to DSBs, as the same flag is used.

dsb_site = 1000 # Hyper-effective TAD boundary
dsb_present = True

# Define DSB site (only second value needed (so 300 for a DSB site between 299 and 300))
dsb_ends_connected = True # if you don't want DSB ends to separate

cohesion_recruitment = True # cohesive cohesin is recruited to the DSB site to mediate the trans search process through sister chromatid cohesion

kbp_offset_dsb_homology = 4
p_bond = 1 # generic probability for a bond to form between the damaged and template strand by cohesion recruitment at the DSB site
distance_threshold = 2 # threshold for a trans contact to form

# Homology search
homology_binding = False # should be kept as False in FPT simulations
homology_capture_probability = 1.0 # Describes probability of an LEF stalling at a DSB-end-bound homology site.
homology_site = 150 # Optional for homology search features; in this case, N1 represents the full two-strand system (so twice the value of N1 as defined above). As with the DSB site, only list the right side here.

# Boundary element strengths
B_cohesion = 0.5
B_ctcf = 0.5

contacts_cutoff = 5

# Random TAD generation
use_random_tads = True
num_tads_per_strand = 4
min_spacing = 0
