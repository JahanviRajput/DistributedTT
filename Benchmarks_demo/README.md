# Demos

We choose some benchmarks to demonstrate the working of our method **DiPTS** (**Di**stributed **P**robabilistic **TE**nsor **S**ampling)

## Description

The files in this folder run independently, the functions are defined for each function within the file. Running the file will return results for the function as described in the file name. We run all of these for the `d=5`, `m=1000` , `n=11`, `k_gd=1`. Within `linear_regression.py` we model a simple regression problem as an optimization problem and find a solutions using the original PROTES algorithm. 

In `noise_functions` we demonstrate how noise is being incorporated into the black boxes for the selected functions. 
 
