export AL_PROGRESS_RANKS_PER_NUMA_NODE=2
export OMP_NUM_THREADS=8
export MV2_USE_RDMA_CM=0

# This should be a checkpointed lenet model
MODEL_LOC="path/to/checkpointed/model"

./Main $MODEL_LOC
./Main $MODEL_LOC --dist
./Main $MODEL_LOC --conduit
