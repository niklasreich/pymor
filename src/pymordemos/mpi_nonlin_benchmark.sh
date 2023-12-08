#!/bin/bash
python -m nonlinear_reaction_batch_script
mpiexec -n 2 python -m nonlinear_reaction_batch_script
mpiexec -n 3 python -m nonlinear_reaction_batch_script
mpiexec -n 4 python -m nonlinear_reaction_batch_script
mpiexec -n 5 python -m nonlinear_reaction_batch_script
mpiexec -n 6 python -m nonlinear_reaction_batch_script
mpiexec -n 7 python -m nonlinear_reaction_batch_script
mpiexec -n 8 python -m nonlinear_reaction_batch_script
mpiexec -n 9 python -m nonlinear_reaction_batch_script
mpiexec -n 10 python -m nonlinear_reaction_batch_script
mpiexec -n 11 python -m nonlinear_reaction_batch_script
mpiexec -n 12 python -m nonlinear_reaction_batch_script
python -m nonlinear_reaction_batch_script --batchsize 2
python -m nonlinear_reaction_batch_script --batchsize 3
python -m nonlinear_reaction_batch_script --batchsize 4
python -m nonlinear_reaction_batch_script --batchsize 5
python -m nonlinear_reaction_batch_script --batchsize 6
python -m nonlinear_reaction_batch_script --batchsize 7
python -m nonlinear_reaction_batch_script --batchsize 8
python -m nonlinear_reaction_batch_script --batchsize 9
python -m nonlinear_reaction_batch_script --batchsize 10
python -m nonlinear_reaction_batch_script --batchsize 11
python -m nonlinear_reaction_batch_script --batchsize 12