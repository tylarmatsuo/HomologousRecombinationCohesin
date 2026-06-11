N1 = 3000 # size of one strand of chromatin 
M = 50 # repeat system M times

LIFETIME = 60 # equivalent to one half of processivity/monomer size, as each subunit moves one monomer per timestep
SEPARATION = 120
SEPARATION_AFTER_DSB = None # Optional post-DSB separation; set to None if this feature isn't needed

# Simulation Timescales
simulation_cycles = 10
steady_state_relax_1D = 10000 # Run 1D simulation for this long before recording positions
trajectoryLength = 6000 # Total time to run 3D simulation, including dsb_time but not including steady_state_relax_1D
dsb_time = 1000 # Let simulation evolve for this long before inducing DSB and beginning to record

ctcf_sites = []

dsb_site = None # Hyper-effective TAD boundary

# Define DSB site (only second value needed (so 300 for a DSB site between 299 and 300))
dsb_ends_connected = False # if you don't want DSB ends to separate

# Boundary element strengths
B_CTCF = 0.5

# Contacts map distance threshold
contacts_cutoff = 5