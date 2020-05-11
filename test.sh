#!/bin/bash

flags="-O2 -std=c++17"
defs="-D BENCHMARK=1"

if [ $# -ne 1 ]; then 
    echo "Invalid arguments. Expected comma-separated points."
    echo "Usage: test.sh points"
	exit 1
fi;

# compile and run
mpic++ --prefix /usr/local/share/OpenMPI $flags $defs -o vid vid.cpp
mpirun --prefix /usr/local/share/OpenMPI vid $1

# clean
rm -f vid
