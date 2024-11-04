import numpy as np
import time
from scipy.integrate import simps
import Decomposition as DC
import Cosmology as Csm
from scipy.integrate import quad
import importlib
importlib.reload(Csm)
importlib.reload(DC)

#########################################################
#Precalculating all the cosmology terms
#The evaluation of the coefficient decomposition is very fast
sampling_cosmo = Csm.Sampling() #Here we can vary the cosmological parameters, but now we just use the default params
khmin = 1e-8
khmax = 52.0
Nmax = 200
#c_n_array = sampling_cosmo.CoeffTransfer(sampling_cosmo.default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)[:, 0]
#nu_n_array = sampling_cosmo.CoeffTransfer(sampling_cosmo.default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)[:, 1]
nu_n_array = np.load('/Users/cheng/Documents/Research_at_Cambridge/Limber/1705Python/Package/cosmo_params/nu_n_array.npy')
#print('Linear Power Spectrum at z=0 expanded. The number of expansion terms is: %d'%len(c_n_array))

#########################################################
#Initializing the special function tools
#Here we do the pre-calulcation of all these modifed functions, given the ith expansion term
#For lensing case, we just need to call the lensing functions instead.
def special_func_generator(x_min = 1e-7, x_max = 1e7, N_sample = 2000, nu_n_array=nu_n_array):
    '''
    This generates the set of 201 M2i(x) functions defined in the paper, specially for the matter clustering case.

    NOTICE: This function has to be run first before any following calculations

    Params:
    x_min, x_max: the minimum and maximum of l*delta_chi/(chi+0.5*delta_chi), in order to cover all the target interpolation range
    N_sample: 2000 is the default number of sampling points, which is precise enough for all calculations

    Returns:
    2 lists of Python function objects, corresponding to real and imaginary part of M2i(x)

    '''
    x_test = 10** np.array([np.log10(x_min) + np.log10(x_max/x_min)/N_sample*i for i in range(N_sample)])
    #Sampling in log scale, changing the sampling points will significantly change the time of evaluation
    Nmax_in = int(len(nu_n_array)-1)
    #nu_n_array = DC.nu_n_array
    print('New Import')
    print('Number of interpolation sampling points are:', N_sample)
    print('Modified arguments are in the interval of', '[', x_min, ',', x_max, ']')
    print('Creating the modified functions...')
    start = time.time()
    func_real_list, func_imag_list = DC.special_func2_interp(nu_n_array, x_test, Nmax = Nmax_in) #These are the functions using M2i(x) as mentioned in the paper
    end = (time.time()-start)/60
    print('Function tools activated.')
    print('Time consumed:', end, 'min')
    return func_real_list, func_imag_list

def special_func_generator_CMBlensing(x_min = 1e-6, x_max = 1e5, N_sample = 2000, nu_n_array=nu_n_array):
    '''
    The same as special_func_generator, but specially built up for CMB lensing initial power spectrum

    '''

    x_test = 10** np.array([np.log10(x_min) + np.log10(x_max/x_min)/N_sample*i for i in range(N_sample)])
    #Sampling in log scale, changing the sampling points will significantly change the time of evaluation
    Nmax_in = int(len(nu_n_array)-1)
    #nu_n_array = DC.nu_n_array
    print('Number of interpolation sampling points are:', N_sample)
    print('Modified arguments are in the interval of', '[', x_min, ',', x_max, ']')
    print('Creating the modified functions...')
    start = time.time()
    func_real_list, func_imag_list = DC.special_func_lensing_interp(nu_n_array, x_test, Nmax = Nmax_in) #These are the functions using M2i(x) as mentioned in the paper
    end = (time.time()-start)/60
    print('Function tools activated.')
    print('Time consumed:', end, 'min')
    return func_real_list, func_imag_list



#############################################################################################################

def power_calc_sampling(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_array, nu_n_array, func_real_list, func_imag_list):
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
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *l/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde = l/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)

    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list[int(Nmax/2)](np.abs(xx))+1j*func_imag_list[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    '''
    Cl_array_array = np.array([ (c_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(dchi_dchi)**(-nu_n_array[i]-1) for i in range(int(Nmax)+1)])
    '''
    #Cl_array_array[np.isnan(Cl_array_array)]=0.
    #Cl_array = np.sum(Cl_array_array, axis=0)

    Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2 * (dchi_dchi/chi_chi)**n
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps( intover_dchi, chi_chi[0, :] ) 

    return results

def full_calc_sampling(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_array, nu_n_array, func_real_list, func_imag_list):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_array: the decomposed coefficients array of initial power spectrum

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = sampling_cosmo.mesh_grid_generator(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_array, nu_n_array, func_real_list, func_imag_list).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)

#############################################################################################################
#Calculation with geometric corrections
def power_calc_sampling_mod(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_array, nu_n_array, func_real_list, func_imag_list):
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
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    l_tilde = np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)

    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list[int(Nmax/2)](np.abs(xx))+1j*func_imag_list[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    '''
    Cl_array_array = np.array([ (c_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    
    Cl_array = np.sum(Cl_array_array, axis=0)
    '''
    Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2 * (dchi_dchi/chi_chi)**n

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_sampling_mod(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_array, nu_n_array, func_real_list, func_imag_list):
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
    power_array = [power_calc_sampling_mod(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_array, nu_n_array, func_real_list, func_imag_list).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)
###################################################################################################################
#Function for CMB lensing
def power_calc_sampling_CMBlensing(l, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    Params:
    l: The multiple
    n: The order of our approximation, usually order 0 will be good enough
    chi_chi, dchi_dchi: The 2D mesh-grid of the chi (dchi) parameter. 
                        The n_row is the same as length of dchi array, 
                        while the n_columns is the same as length of chi array.
    D1_D1, D2_D2: The mesh-grid of growth factor. The same shape as chi_chi.
    Wg1_Wg1, Wg2_Wg2: The mesh-grid of several window functions.
    F1_Wg1, F2_F2: The mesh-grid of potential factors.

    Return:
    The angular power spetrum at mutiple l.
    '''
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *l/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde = l/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)

    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list1[int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1-4)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list1[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1-4) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    '''
    Cl_array_array = np.array([ (c_array[i]*(func_real_list1[i](np.abs(xx))+1j*func_imag_list1[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1-4) for i in range(int(Nmax)+1)])
    
    #Cl_array_array[np.isnan(Cl_array_array)]=0.
    Cl_array = np.sum(Cl_array_array, axis=0) #*(1-(dchi_dchi/2/chi_chi))**2/(1+(dchi_dchi/2/chi_chi))**2
    '''
    Simp_array = F1_F1*F2_F2*D1_D1*D2_D2*Cl_array*W1_W1*W2_W2/(chi_chi)**2
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_sampling_CMBlensing(l_array, Nchi, Ndchi, z_source, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2 = sampling_cosmo.mesh_grid_generator_CMBlensing(Nchi, Ndchi, z_source)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_CMBlensing(li, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array

def power_calc_sampling_CMBlensing_mod(l, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    Params:
    l: The multiple
    n: The order of our approximation, usually order 0 will be good enough
    chi_chi, dchi_dchi: The 2D mesh-grid of the chi (dchi) parameter. 
                        The n_row is the same as length of dchi array, 
                        while the n_columns is the same as length of chi array.
    D1_D1, D2_D2: The mesh-grid of growth factor. The same shape as chi_chi.
    Wg1_Wg1, Wg2_Wg2: The mesh-grid of several window functions.
    F1_Wg1, F2_F2: The mesh-grid of potential factors.

    Return:
    The angular power spetrum at mutiple l.
    '''
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    l_tilde = np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list1[int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1-4)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list1[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1-4) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    '''
    Cl_array_array = np.array([ (c_array[i]*(func_real_list1[i](np.abs(xx))+1j*func_imag_list1[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1-4) for i in range(int(Nmax)+1)])
    

    Cl_array = np.sum(Cl_array_array, axis=0)#*(1-(dchi_dchi/2/chi_chi))**2/(1+(dchi_dchi/2/chi_chi))**2
    '''
    Simp_array = F1_F1*F2_F2*D1_D1*D2_D2*Cl_array*W1_W1*W2_W2/(chi_chi)**2
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_sampling_CMBlensing_mod(l_array, Nchi, Ndchi, z_source, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2 = sampling_cosmo.mesh_grid_generator_CMBlensing(Nchi, Ndchi, z_source)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_CMBlensing_mod(li, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array

###################################################################################################################
#Without radial derivative
def power_calc_sampling_CMBlensing_wo_rad(l, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    This is the calculation of approximation of lensing potential without the radial derivative
    Params:
    The same as other functions
    Only that for the func_real_list1 and func_imag_list1 we shall use the matter power spectrum ones

    Return:
    The angular power spetrum at mutiple l.
    '''
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    l_tilde = np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list1[int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list1[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    Simp_array = F1_F1*F2_F2*D1_D1*D2_D2*Cl_array.real*(chi_chi)**2 * (1-(dchi_dchi/2/chi_chi)**2)**1/ (l*(l+1))**2 *W1_W1*W2_W2 
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_sampling_CMBlensing_wo_rad(l_array, Nchi, Ndchi, z_source, c_array, nu_n_array, func_real_list1, func_imag_list1):
    
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2 = sampling_cosmo.mesh_grid_generator_CMBlensing(Nchi, Ndchi, z_source)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_CMBlensing_wo_rad(li, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array

###################################################################################################################
#Lensing X clustering

def power_calc_sampling_gk_mod(l, chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1):
    '''
    Params:
    l: The multiple
    n: The order of our approximation, usually order 0 will be good enough
    chi_chi, dchi_dchi: The 2D mesh-grid of the chi (dchi) parameter. 
                        The n_row is the same as length of dchi array, 
                        while the n_columns is the same as length of chi array.
    D1_D1, D2_D2: The mesh-grid of growth factor. The same shape as chi_chi.
    Wg1_Wg1, Wg2_Wg2: The mesh-grid of several window functions.
    F1_Wg1, F2_F2: The mesh-grid of potential factors.

    Return:
    The angular power spetrum at mutiple l.
    '''
    Nmax = int(len(nu_n_array)-1)
    xx = dchi_dchi *np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    l_tilde = np.sqrt(l*(l+1))/chi_chi/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    Cl_array_0 = (c_array[int(Nmax/2)]*(func_real_list1[int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[int(Nmax/2)]+1)
    Cl_array_array = np.array([ (c_array[i+int(Nmax/2)]*(func_real_list1[i+int(Nmax/2)](np.abs(xx))+1j*func_imag_list1[i+int(Nmax/2)](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i+int(Nmax/2)]+1) for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = Cl_array_0 + 2*np.sum(Cl_array_array, axis=0)
    
    #Simp_array = F1_F1*F2_F2*D1_D1*D2_D2*Cl_array*W1_W1*W2_W2/(chi_chi-0.0*dchi_dchi)**2/2*l*(l+1) #Normal
    Simp_array = F1_F1*F2_F2*D1_D1*D2_D2*Cl_array*W1_W1*W2_W2*(chi_chi+0.5*dchi_dchi)/(chi_chi-0.5*dchi_dchi)/2 #No rad
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_sampling_gk(l_array, z1, sigma1, Nchi, Ndchi, z_source, c_array, nu_n_array, func_real_list1, func_imag_list1):
    
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, W1_W1, W2_W2, F1_F1, F2_F2 = sampling_cosmo.mesh_grid_generator_CMBlensing(Nchi, Ndchi, z_source)
    chi1_chi1 = chi_chi-0.5*dchi_dchi
    Wg1_Wg1 = sampling_cosmo.Wg(chi1_chi1, sampling_cosmo.default_cosmo.chi(z1), sigma1/sampling_cosmo.default_cosmo.HH(z1))
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_gk_mod(li, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, W2_W2, 1., F2_F2, c_array, nu_n_array, func_real_list1, func_imag_list1).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array


###################################################################################################################
#Limber's approximation

def Pk_potent(kh):

    return sampling_cosmo.default_cosmo.Plin(kh)

def Pk_potent2(kh):

    return sampling_cosmo.default_cosmo.Plin(kh)/kh**2

def Pk_potent1(kh):

    return sampling_cosmo.default_cosmo.Plin(kh)/kh**4

def Power_spectrum(kh_par, l, chi, dchi):

    kh = np.sqrt(kh_par**2 + (l/(chi))**2)

    return Pk_potent(kh)

def Power_spectrum_CMBlensing(kh_par, l, chi, dchi):

    kh = np.sqrt(kh_par**2 + (l/(chi))**2)

    return Pk_potent1(kh)

def Kernel_Limber(l, chi, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2):
    
    W1 = sampling_cosmo.Wg(chi, chi_avg1, chi_sigma1)
    W2 = sampling_cosmo.Wg(chi, chi_avg2, chi_sigma2)
    D1 = sampling_cosmo.D_class(chi)
    C = Power_spectrum(kh_par=0, l=l, chi=chi, dchi=0)

    return D1*D1*W1*W2*C
    
def Cn_Limber(l, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2, chi_min, chi_max):

    def integrand(chi):
        return Kernel_Limber(l,chi, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2)/(chi**2)
    
    return quad(integrand, chi_min, chi_max)[0]


def Limber_CMBlensing(l, chi_min, chi_star):

    def integrand(chi):
        W1 = sampling_cosmo.Wlensing(chi, chi_star)
        #D1 = sampling_cosmo.default_cosmo.Dg_norm(chi)
        D1 = sampling_cosmo.D_class(chi)
        F1 = sampling_cosmo.default_cosmo.Psi_normalizer(chi)

        return sampling_cosmo.default_cosmo.Plin(l/chi)*chi**2/l**4 * (W1**2) * (D1**2) * (F1**2)
    
    return quad(integrand, chi_min, chi_star)[0]

def Limber_gk(l, chi_avg, chi_sigma, chi_min, chi_star):

    def integrand(chi):
        W1 = sampling_cosmo.Wlensing(chi, chi_star)
        W2 = sampling_cosmo.Wg(chi, chi_avg, chi_sigma)
        D1 = sampling_cosmo.D_class(chi)
        F1 = sampling_cosmo.default_cosmo.Psi_normalizer(chi)

        return sampling_cosmo.default_cosmo.Plin(l/chi) * (W1*W2) * (D1**2) * (F1**1)
    
    return quad(integrand, chi_min, chi_star)[0]


'''
#These two functions are used to calculate the galaxy clustering power spectrum using the old sampling method

def power_calc_sampling(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array):

    xx = dchi_dchi*l/(chi_chi+0.5*dchi_dchi)
    
    Cl_array_array = np.array([ (c_n_array[i+int(Nmax/2)+1]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(dchi_dchi)**(-nu_n_array[int(Nmax/2)+1+i]-1) for i in range(int(Nmax/2))])
    
    #Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        #np.abs(dchi_dchi)**(-nu_n_array[i]-1) for i in range(int(Nmax)+1)])
    
    Cl_array = np.sum(Cl_array_array, axis=0)

    Simp_array = D1_D1*D2_D2*2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    results = simps(simps(Simp_array, chi_chi[0, :]), dchi_dchi[:, 0])
    return results

def full_calc_sampling(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_n_array = c_n_array):
   
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = sampling_cosmo.mesh_grid_generator_old(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)

def power_calc_sampling_modify(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array):
    
    #With geometric modification, we only need to re-evaluate the \tilde{\ell} by a factor of sqrt(l*(l+1))/sqrt(1-delta**2)
    #All the rest of calculation will remain the same.
    delta_delta = 0.5*(dchi_dchi/chi_chi)

    xx = dchi_dchi*np.sqrt(l*(l+1))/(chi_chi+0.5*dchi_dchi)/(1-delta_delta)  
    
    Cl_array_array = np.array([ (c_n_array[i+int(Nmax/2)+1]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(dchi_dchi)**(-nu_n_array[int(Nmax/2)+1+i]-1) for i in range(int(Nmax/2))])
    
    #Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        #np.abs(dchi_dchi)**(-nu_n_array[i]-1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    Simp_array = D1_D1*D2_D2*2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    results = simps(simps(Simp_array, chi_chi[0, :]), dchi_dchi[:, 0])
    return results

def full_calc_sampling_modify(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_n_array = c_n_array):

    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = sampling_cosmo.mesh_grid_generator(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_modify(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)
'''