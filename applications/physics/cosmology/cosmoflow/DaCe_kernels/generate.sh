#!/bin/sh

# Frontier environment setup
source $LMOD_PKG/init/sh
module load amd/5.5.1
export LIBRARY_PATH="$(which hipcc | xargs dirname)/../lib:$LIBRARY_PATH"
export CC=`which amdclang`
export CXX=`which amdclang++`
export DACE_compiler_extra_cmake_args="-DHIP_CLANG_PATH=${ROCM_PATH}/llvm/bin"


generate_conv3d() {
    # Explicit GEMM
    python generate_egemm_fwd.py explicit 2048 conv $*
}

# Dynamic
#B="-1"
B=2

# 128
#45ms=9.1+5.98+29.98ms vs 17.4
#generate_conv3d $B 4 128 128 128 8388608 2097152 16384 128 1 16 4 3 3 3 $B 16 128 128 128 33554432 2097152 16384 128 1 1 1 1 1 1 1 1 1 1 1
#39ms=6.4+4.97+27.6ms vs 8.2
#generate_conv3d $B 16 64 64 64 4194304 262144 4096 64 1 32 16 3 3 3 $B 32 64 64 64 8388608 262144 4096 64 1 1 1 1 1 1 1 1 1 1 1
#11.6ms=1.5+1.3+8.7ms vs. 2.1
#generate_conv3d $B 32 32 32 32 1048576 32768 1024 32 1 64 32 3 3 3 $B 64 32 32 32 2097152 32768 1024 32 1 1 1 1 1 1 1 1 1 1 1
#134us vs 198us
#generate_conv3d $B 64 16 16 16 262144 4096 256 16 1 128 64 3 3 3 $B 128 8 8 8 65536 512 64 8 1 1 1 1 2 2 2 1 1 1 1
#91us vs 126us
#generate_conv3d $B 128 4 4 4 8192 64 16 4 1 256 128 3 3 3 $B 256 4 4 4 16384 64 16 4 1 1 1 1 1 1 1 1 1 1 1
#446us vs 112us
#generate_conv3d $B 256 2 2 2 2048 8 4 2 1 256 256 3 3 3 $B 256 2 2 2 2048 8 4 2 1 1 1 1 1 1 1 1 1 1 1

# 128 MLPerf
# 20+12+29=61ms vs. 32? + 65
#generate_conv3d $B 4 128 128 128 8388608 2097152 16384 128 1 32 4 3 3 3 $B 32 128 128 128 67108864 2097152 16384 128 1 1 1 1 1 1 1 1 1 1 1
# # 18+10.6+11.5=40 vs. 30
# generate_conv3d $B 32 64 64 64 8388608 262144 4096 64 1 64 32 3 3 3 $B 64 64 64 64 16777216 262144 4096 64 1 1 1 1 1 1 1 1 1 1 1
# # # # 4+3.5+1.5=9.2 vs. 10
# generate_conv3d $B 64 32 32 32 2097152 32768 1024 32 1 128 64 3 3 3 $B 128 32 32 32 4194304 32768 1024 32 1 1 1 1 1 1 1 1 1 1 1
# # # # 0.9+0.3+1.7=3.0 vs. 3.6
# generate_conv3d $B 128 16 16 16 524288 4096 256 16 1 256 128 3 3 3 $B 256 16 16 16 1048576 4096 256 16 1 1 1 1 1 1 1 1 1 1 1
# # # # 0.2+1.038+0.088=1.3 vs. 1.5
# generate_conv3d $B 256 8 8 8 131072 512 64 8 1 512 256 3 3 3 $B 512 8 8 8 262144 512 64 8 1 1 1 1 1 1 1 1 1 1 1

# 512 MLPerf + distconv
generate_conv3d $B 4 66 512 512 69206016 17301504 262144 512 1 32 4 3 3 3 $B 32 64 512 512 553648128 17301504 262144 512 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 32 34 256 256 71303168 2228224 65536 256 1 64 32 3 3 3 $B 64 32 256 256 142606336 2228224 65536 256 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 64 18 128 128 18874368 294912 16384 128 1 128 64 3 3 3 $B 128 16 128 128 37748736 294912 16384 128 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 128 10 64 64 5242880 40960 4096 64 1 256 128 3 3 3 $B 256 8 64 64 10485760 40960 4096 64 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 256 6 32 32 1572864 6144 1024 32 1 512 256 3 3 3 $B 512 4 32 32 3145728 6144 1024 32 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 512 4 16 16 524288 1024 256 16 1 512 512 3 3 3 $B 512 2 16 16 524288 1024 256 16 1 0 1 1 1 1 1 1 1 1 1 fwd
generate_conv3d $B 512 3 8 8 98304 192 64 8 1 512 512 3 3 3 $B 512 1 8 8 98304 192 64 8 1 0 1 1 1 1 1 1 1 1 1 fwd

# TODO
# generate_conv3d $B 512 3 8 8 98304 192 64 8 1 512 512 3 3 3 $B 512 3 8 8 98304 192 64 8 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 512 3 8 8 98304 192 64 8 1 512 512 3 3 3 $B 512 3 8 8 98304 192 64 8 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 512 4 16 16 524288 1024 256 16 1 512 512 3 3 3 $B 512 4 16 16 524288 1024 256 16 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 512 4 16 16 524288 1024 256 16 1 512 512 3 3 3 $B 512 4 16 16 524288 1024 256 16 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 256 6 32 32 1572864 6144 1024 32 1 512 256 3 3 3 $B 512 6 32 32 3145728 6144 1024 32 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 256 6 32 32 1572864 6144 1024 32 1 512 256 3 3 3 $B 512 6 32 32 3145728 6144 1024 32 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 128 10 64 64 5242880 40960 4096 64 1 256 128 3 3 3 $B 256 10 64 64 10485760 40960 4096 64 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 128 10 64 64 5242880 40960 4096 64 1 256 128 3 3 3 $B 256 10 64 64 10485760 40960 4096 64 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 64 18 128 128 18874368 294912 16384 128 1 128 64 3 3 3 $B 128 18 128 128 37748736 294912 16384 128 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 64 18 128 128 18874368 294912 16384 128 1 128 64 3 3 3 $B 128 18 128 128 37748736 294912 16384 128 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 32 34 256 256 71303168 2228224 65536 256 1 64 32 3 3 3 $B 64 34 256 256 142606336 2228224 65536 256 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 32 34 256 256 71303168 2228224 65536 256 1 64 32 3 3 3 $B 64 34 256 256 142606336 2228224 65536 256 1 1 1 1 1 1 1 1 1 1 1 bwddata
# generate_conv3d $B 4 66 512 512 69206016 17301504 262144 512 1 32 4 3 3 3 $B 32 66 512 512 553648128 17301504 262144 512 1 1 1 1 1 1 1 1 1 1 1 bwdfilt
# generate_conv3d $B 4 66 512 512 69206016 17301504 262144 512 1 32 4 3 3 3 $B 32 66 512 512 553648128 17301504 262144 512 1 1 1 1 1 1 1 1 1 1 1 bwddata