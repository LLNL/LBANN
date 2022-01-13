# source /etc/bashrc

#!/bin/bash -l
#module load cmake/3.9.2
#salloc -N1 -t 600 python -m pytest -s --junitxml=results.xml
#cd bamboo/compiler_tests
#python -m pytest -s test_compiler.py -k 'test_compiler_build_script' --junitxml=results.xml
#salloc -N1 -t 600 python -m pytest -s test_compiler.py -k 'test_compiler_build_script' --junitxml=results.xml
export SPACK_ROOT=/g/g19/vanessen/spack.git; . $SPACK_ROOT/share/spack/setup-env.sh
${CI_PROJECT_DIR}/scripts/build_lbann.sh -d -l bamboo --test --clean-build -j $(($(nproc)+2)) -- +deterministic +vision +numpy
exit 0
