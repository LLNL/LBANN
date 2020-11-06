"""Dataset for offline random walks.

This script loads graph walks that have been generated offline. It is
intended to be imported by the Python data reader and used in a
Skip-Gram algorithm. Data samples are sequences of vertex indices,
corresponding to a graph walk and negative samples.

"""
import configparser
import os
import os.path
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd

current_file = os.path.realpath(__file__)
app_dir = os.path.dirname(os.path.dirname(current_file))
sys.path.append(os.path.join(app_dir))
import utils

# ----------------------------------------------
# Configuration
# ----------------------------------------------

# Load config file
config_file = os.getenv('LBANN_NODE2VEC_CONFIG_FILE')
if not config_file:
    raise RuntimeError(
        'No configuration file provided in '
        'LBANN_NODE2VEC_CONFIG_FILE environment variable')
if not os.path.exists(config_file):
    raise FileNotFoundError(f'Could not find config file at {config_file}')
config = configparser.ConfigParser()
config.read(os.path.join(app_dir, 'default.config'))
config.read(config_file)

# Options from config file
num_vertices = config.getint('Graph', 'num_vertices', fallback=0)
walk_file = config.get('Walks', 'file', fallback=None)
walk_length = config.getint('Walks', 'walk_length', fallback=0)
num_walks = config.getint('Walks', 'num_walks', fallback=0)
epoch_size = config.getint('Skip-gram', 'epoch_size')
num_negative_samples = config.getint('Skip-gram', 'num_negative_samples')
noise_distribution_exp = config.getfloat('Skip-gram', 'noise_distribution_exp')

# Configure RNG
rng_pid = None
def initialize_rng():
    """Initialize NumPy random seed if needed.

    Seed should be initialized independently on each process. We
    reinitialize if we detect a process fork.

    """
    global rng_pid
    if rng_pid != os.getpid():
        rng_pid = os.getpid()
        np.random.seed()

# Determine MPI rank
if 'OMPI_COMM_WORLD_RANK' not in os.environ:
    warnings.warn(
        'Could not detect MPI environment variables. '
        'We expect that LBANN is run with '
        'Open MPI or one of its derivatives.',
        warning.RuntimeWarning,
    )
mpi_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', default=0))
mpi_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', default=1))

# ----------------------------------------------
# Walks
# ----------------------------------------------

# Load walks from file
# Note: Each MPI rank loads a subset of the walk file
if not walk_file:
    raise RuntimeError(f'No walk file specified in {config_file}')
if not num_walks:
    with open(walk_file, 'r') as f:
        result = subprocess.run(['wc','-l'], stdin=f, stdout=subprocess.PIPE)
    num_walks = int(result.stdout.decode('utf-8'))
local_num_walks = utils.ceildiv(num_walks, mpi_size)
start_walk = local_num_walks * mpi_rank
end_walk = min(local_num_walks * (mpi_rank + 1), num_walks)
local_num_walks = end_walk - start_walk
walks = pd.read_csv(
    walk_file,
    delimiter=' ',
    header=None,
    dtype=np.int64,
    skiprows=start_walk,
    nrows=local_num_walks,
)
walks = walks.to_numpy()

# Check that walk data is valid
if not walk_length:
    walk_length = walks.shape[1]
if not num_vertices:
    num_vertices = np.amax(walks) + 1
assert walks.shape[0] > 0, f'Did not load any walks from {walk_file}'
assert walks.shape[0] == local_num_walks, \
    f'Read {walks.shape[0]} walks from {walk_file}, ' \
    f'but expected to read {local_num_walks}'
assert walks.shape[1] == walk_length, \
    f'Found walks of length {walks.shape[1]} in {walk_file}, ' \
    f'but expected a walk length of {walk_length}'
assert num_vertices > 0, \
    f'Walks in {walk_file} have invalid vertex indices'

# Randomly sample local walks
walk_index_cache = []
walk_index_cache_pos = 0
def choose_walk():
    """Randomly choose one of the local walks.

    Random numbers are generated in batches to amortize Python
    overheads.

    """
    global walk_index_cache, walk_index_cache_pos
    if walk_index_cache_pos >= len(walk_index_cache):
        initialize_rng()
        walk_index_cache_pos = 0
        walk_index_cache = np.random.randint(
            local_num_walks,
            size=1024,
            dtype=np.int64,
        )
        record_walk_visits(walks[walk_index_cache])
    index = walk_index_cache[walk_index_cache_pos]
    walk_index_cache_pos += 1
    return walks[index]

# ----------------------------------------------
# Negative sampling
# ----------------------------------------------

# Keep track how often we visit each vertex
# Note: Start with one visit per vertex for Laplace smoothing.
visit_counts = np.ones(num_vertices, dtype=np.int64)
total_visit_count = num_vertices
def record_walk_visits(walks):
    """Record visits to vertices in a graph walk.

    Recomputes negative sampling distribution

    """
    global visit_counts, total_visit_count
    visit_counts[walks] += np.int64(1)
    total_visit_count += len(walks) * walk_length

# Probability distribution for negative sampling
noise_cdf = np.zeros(num_vertices, dtype=np.float64)
noise_visit_count = 0
def update_noise_distribution():
    """Recomputes negative sampling probability distribution, if needed.

    The distribution is recomputed if enough new vertices have been
    visited.

    """
    global noise_visit_count
    if total_visit_count > 2*noise_visit_count:

        # Reset negative samples
        global negative_samples_cache_pos, negative_samples_cache
        negative_samples_cache = []
        negative_samples_cache_pos = 0

        # Update noise distribution if there are enough new visits
        global noise_cdf
        np.float_power(
            visit_counts,
            noise_distribution_exp,
            out=noise_cdf,
            dtype=np.float64,
        )
        np.cumsum(noise_cdf, out=noise_cdf)
        noise_cdf *= np.float64(1 / noise_cdf[-1])
        noise_visit_count = total_visit_count

# Initial negative sampling distribution is uniform
update_noise_distribution()

# Generate negative sampling
negative_samples_cache = []
negative_samples_cache_pos = 0
def generate_negative_samples():
    """Generate negative samples.

    Returns an array with num_negative_samples entries. Negative
    samples are generated in batches to amortize Python overheads.

    """
    global negative_samples_cache, negative_samples_cache_pos
    if negative_samples_cache_pos >= len(negative_samples_cache):
        initialize_rng()
        rands = np.random.uniform(size=1024*num_negative_samples)
        negative_samples_cache_pos = 0
        negative_samples_cache = np.searchsorted(noise_cdf, rands)
    pos_start = negative_samples_cache_pos
    pos_end = negative_samples_cache_pos + num_negative_samples
    negative_samples_cache_pos = pos_end
    return negative_samples_cache[pos_start:pos_end]

# ----------------------------------------------
# Sample access functions
# ----------------------------------------------

def get_sample(*args):
    """Get a single data sample.

    A data sample consists of a graph walk and several negative
    samples. Input arguments are ignored.

    """
    walk = choose_walk()
    update_noise_distribution()
    negative_samples = generate_negative_samples()
    return np.concatenate((negative_samples, walk))

def num_samples():
    """Number of samples per "epoch".

    This data reader samples randomly with replacement, so epochs are
    not meaningful. However, it is convenient to group data samples
    into "epochs" so that we don't need to change LBANN's data model.

    """
    return epoch_size

def sample_dims():
    """Dimensions of a data sample."""
    return (num_negative_samples + walk_length,)
