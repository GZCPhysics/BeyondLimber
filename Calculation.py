import numpy as np
import time
from scipy.integrate import simps
import Decomposition as DC
import Cosmology as Csm

#########################################################
#Initializing the special function tools
Nmax = DC.Nmax
func_real_list = DC.func_real_list
func_imag_list = DC.func_imag_list
nu_n_array = DC.nu_n_array

#########################################################
#Precalculating all the cosmology terms
sampling_cosmo = Csm.Sampling() #Here we can vary the cosmological parameters
khmin = 1e-8
khmax = 52.0
Nmax = 200
c_n_array = sampling_cosmo.CoeffTransfer(sampling_cosmo.default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)[:, 0]
print('Linear Power Spectrum at z=0 expanded. The number of expansion terms is: %d'%len(c_n_array))


def power_calc_sampling(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array):
    '''
    Params:
    l: The multiple
    n: The order of our approximation, usually order 0 will be good enough
    chi_chi, dchi_dchi: The 2D mesh-grid of the chi (dchi) parameter. 
                        The n_row is the same as length of dchi array, 
                        while the n_columns is the same as length of chi array.
    D1_D1, D2_D2: The mesh-grid of growth factor. The same shape as chi_chi.
    Wg1_Wg1, Wg2_Wg2: The mesh-grid of several window functions.

    Return:
    The angular power spetrum at mutiple l.
    '''
    xx = dchi_dchi*l/(chi_chi+0.5*dchi_dchi)
    Cl_array_array = np.array([ (c_n_array[i+int(Nmax/2)+1]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(dchi_dchi)**(-nu_n_array[int(Nmax/2)+1+i]-1) for i in range(int(Nmax/2))])
    Cl_array = np.sum(Cl_array_array, axis=0)

    Simp_array = D1_D1*D2_D2*2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    results = simps(simps(Simp_array, chi_chi[0, :]), dchi_dchi[:, 0])
    return results

def full_calc_sampling(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_n_array = c_n_array):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = sampling_cosmo.mesh_grid_generator(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array
