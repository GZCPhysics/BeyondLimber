# BeyondLimber

These are the Python scripts for calculation based on our approximation

Major functioning scripts are:
1. Decomposition.py
2. Cosmology.py
3. Calculation.py
4. Calculation_RSD.py

The method of using these scripts are described in example.ipynb

The full-sky angular power spectra are pre-calculated and stored in:
1. /Full-sky/ (.txt type for clustering, .npy type for RSD)

Note that all the angular power spectra are calculated based on the power spectrum given in /cosmo_params/pk_camb_planck18.txt

Required the packages are: 
### numpy, scipy, mpmath, Colossus, pyCAMB, CLASS.

The choice of cosmology related functions could be replaced with anything users are familiar with.

All the flat-sky calculations are based on Gaussian window function. In practice, as long as you have two mesh-grids of targeting sources on Windows, Growth factors... you can use **power_calc_sampling**-like functions in **Calculation.py** and **Calculation_RSD.py** to calculate.
