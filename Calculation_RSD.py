import numpy as np
import time
from scipy.integrate import simps
import Decomposition as DC
import Cosmology as Csm
from scipy.integrate import quad

#########################################################
#Precalculating all the cosmology terms
#The evaluation of the coefficient decomposition is very fast
sampling_cosmo = Csm.Sampling() #Here we can vary the cosmological parameters, but now we just use the default params
khmin = 1e-8
khmax = 52.0
Nmax = 200
#c_n_array = sampling_cosmo.CoeffTransfer(sampling_cosmo.default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)[:, 0]
#nu_n_array = sampling_cosmo.CoeffTransfer(sampling_cosmo.default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)[:, 1]
#print('Linear Power Spectrum at z=0 expanded. The number of expansion terms is: %d'%len(c_n_array))

#Generating special functions:
def special_func_rsd(x_min, x_max, N_sample, Nmax, c_n_array, nu_n_array):

    '''
    Params: 
    x_min, x_max: the lower and upper limit of x=dchi*l/chi/sqrt(1-delta^2)
    N_sample: the number of sampling points of x, which is sampled on log-scale
    Nmax: the number of FFTLog decomposition points
    c_n_array, nu_n_array: the coefficient and frequency array of FFTLog decomposition
    
    Generate six arrays of special functions needed in RSD
    '''

    x_test = np.array(list(10**np.array([np.log10(x_min) + np.log10(x_max/x_min)/N_sample*i for i in range(N_sample+1)])))
    print('Number of interpolation sampling points are:', N_sample)
    print('Modified arguments are in the interval of', '[', x_min, ',', x_max, ']')
    print('Creating the modified functions...')

    start = time.time()
    K2i_list = [DC.K2i(x_test, nui) for nui in nu_n_array]
    K3i_list = [DC.K3i(x_test, nui) for nui in nu_n_array]
    K4i_list = [DC.K4i(x_test, nui) for nui in nu_n_array]
    #func_real_list4, func_imag_list4 = special_func_interp(nu_n_array, x_test, Nmax, M4i)
    end = (time.time()-start)/60
    print('Function tools activated.')
    print('Time consumed:', end, 'min')
    print('Linear Power Spectrum at z=0 expanded. The number of expansion terms is: %d'%len(c_n_array))
    func_real_list2, func_imag_list2 = DC.M2i_interp(x_test, nu_n_array, K2i_list, Nmax)
    func_real_list3, func_imag_list3 = DC.M3i_interp(x_test, nu_n_array, K2i_list, K3i_list, Nmax)
    func_real_list4, func_imag_list4 = DC.M4i_interp(x_test, nu_n_array, K2i_list, K3i_list, K4i_list, Nmax)

    return func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4


#After generating special functions

def curl_Cl_RSD_mesh(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    This function generally produce the unmodified version of curly Cl for 0th 1st and 2nd terms in the RSD function
    '''
    l_mod = l
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde = l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def curl_Cl_RSD_mesh_mod(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    Similar to curl_Cl_RSD_mesh, but with the same geometric modification for all three parts
    '''
    l_mod = np.sqrt(l*(l+1))
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde = l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def curl_Cl_RSD_mesh_mod2(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    curly Cl of pure clustering, using func_real_list2, and func_imag_list2
    '''
    l_mod = np.sqrt(l*(l+1))
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde =l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def curl_Cl_RSD_mesh_mod31(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    curly Cl of f1*clustering, using func_real_list3, and func_imag_list3
    '''
    l_mod = np.sqrt(l*(l+1))
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/(chi_chi-0.5*dchi_dchi)
    l_tilde = l_mod/(chi_chi-0.5*dchi_dchi)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def curl_Cl_RSD_mesh_mod32(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    curly Cl of f2*clustering, using func_real_list3, and func_imag_list3
    '''
    l_mod = np.sqrt(l*(l+1))
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/(chi_chi+0.5*dchi_dchi)
    l_tilde = l_mod/(chi_chi+0.5*dchi_dchi)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def curl_Cl_RSD_mesh_mod4(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list, func_imag_list):
    '''
    curly Cl of f1*f2*clustering, using func_real_list4, and func_imag_list4
    '''
    l_mod = np.sqrt(l*(l+1))
    Nmax = int(len(c_n_array)-1)
    xx = dchi_dchi*l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    l_tilde = l_mod/chi_chi/np.sqrt(1-(0.5*dchi_dchi/chi_chi)**2)
    Cl_array_array = np.array([ (c_n_array[i]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(l_tilde)**(nu_n_array[i]+1) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array

def power_calc_sampling_RSD(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2, curl_Cl_RSD_mesh, c_n_array, nu_n_array,\
    func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4):
    '''
    Params:
    l: The multiple
    n: The order of our approximation, usually order 0 will be good enough
    chi_chi, dchi_dchi: The 2D mesh-grid of the chi (dchi) parameter. 
                        The n_row is the same as length of dchi array, 
                        while the n_columns is the same as length of chi array.
    D1_D1, D2_D2: The mesh-grid of growth factor. The same shape as chi_chi.
    Wg1_Wg1, Wg2_Wg2: The mesh-grid of several window functions.
    f1_f1, f2_f2: The mesh-grid of lnD/lna
    func_real_listi, func_imag_listi: special function lists for curly Cl calculation  

    Return:
    The angular power spetrum at mutiple l.
    '''
    #xx = dchi_dchi *l/(chi_chi)/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    #l_tilde = l/(chi_chi)/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    Cl_array2 = curl_Cl_RSD_mesh(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list2, func_imag_list2)
    Cl_array31 = curl_Cl_RSD_mesh(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list3, func_imag_list3)
    Cl_array32 = curl_Cl_RSD_mesh(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list3, func_imag_list3)
    Cl_array4 = curl_Cl_RSD_mesh(l, chi_chi, dchi_dchi,c_n_array, nu_n_array, func_real_list4, func_imag_list4)

    #Cl_array = Cl_array2 + f1_f1*Cl_array31+f2_f2*Cl_array32 + f1_f1*f2_f2*Cl_array4
    Cl_0 = Cl_array2
    Cl_1 = f1_f1*Cl_array31+f2_f2*Cl_array32
    Cl_2 = f1_f1*f2_f2*Cl_array4

    #Simp_array = D1_D1*D2_D2*Cl_array.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    #intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    #results = simps(intover_dchi, chi_chi[0, :])

    Simp0_array = D1_D1*D2_D2*Cl_0.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover0_dchi = np.array([simps(Simp0_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results0 = simps(intover0_dchi, chi_chi[0, :])

    Simp1_array = D1_D1*D2_D2*Cl_1.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover1_dchi = np.array([simps(Simp1_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results1 = simps(intover1_dchi, chi_chi[0, :])

    Simp2_array = D1_D1*D2_D2*Cl_2.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover2_dchi = np.array([simps(Simp2_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results2 = simps(intover2_dchi, chi_chi[0, :])

    results = results0 + results1 + results2
    return results, results0, results1, results2

def full_calc_sampling_RSD(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, curl_Cl_RSD_mesh, c_n_array, nu_n_array, \
    func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2 = sampling_cosmo.mesh_grid_generator_RSD(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_RSD(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2, curl_Cl_RSD_mesh, c_n_array, nu_n_array,\
        func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4) for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)

def power_calc_sampling_RSD_mod(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2, curl_Cl_RSD_mesh, c_n_array, nu_n_array, \
    func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4):
    
    #xx = dchi_dchi *l/(chi_chi)/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    #l_tilde = l/(chi_chi)/np.sqrt(1-(dchi_dchi/2/chi_chi)**2)
    Cl_array2 = curl_Cl_RSD_mesh_mod2(l, chi_chi, dchi_dchi, c_n_array, nu_n_array, func_real_list2, func_imag_list2)
    Cl_array31 = curl_Cl_RSD_mesh_mod31(l, chi_chi, dchi_dchi, c_n_array, nu_n_array, func_real_list3, func_imag_list3)
    Cl_array32 = curl_Cl_RSD_mesh_mod32(l, chi_chi, dchi_dchi, c_n_array, nu_n_array, func_real_list3, func_imag_list3)
    Cl_array4 = curl_Cl_RSD_mesh_mod4(l, chi_chi, dchi_dchi, c_n_array, nu_n_array, func_real_list4, func_imag_list4)

    #Cl_array = Cl_array2 + f1_f1*Cl_array31+f2_f2*Cl_array32 + f1_f1*f2_f2*Cl_array4
    Cl_0 = Cl_array2
    Cl_1 = f1_f1*Cl_array31+f2_f2*Cl_array32
    Cl_2 = f1_f1*f2_f2*Cl_array4

    #Simp_array = D1_D1*D2_D2*Cl_array.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    #intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    #results = simps(intover_dchi, chi_chi[0, :])
    
    Simp0_array = D1_D1*D2_D2*Cl_0.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover0_dchi = np.array([simps(Simp0_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results0 = simps(intover0_dchi, chi_chi[0, :])

    Simp1_array = D1_D1*D2_D2*Cl_1.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover1_dchi = np.array([simps(Simp1_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results1 = simps(intover1_dchi, chi_chi[0, :])

    Simp2_array = D1_D1*D2_D2*Cl_2.real*Wg1_Wg1*Wg2_Wg2/(chi_chi)**2
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover2_dchi = np.array([simps(Simp2_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results2 = simps(intover2_dchi, chi_chi[0, :])

    results = results0 + results1 + results2

    return results, results0, results1, results2

def full_calc_sampling_RSD_mod(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, curl_Cl_RSD_mesh, c_n_array, nu_n_array,\
    func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2 = sampling_cosmo.mesh_grid_generator_RSD(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling_RSD_mod(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2, curl_Cl_RSD_mesh, c_n_array,nu_n_array,\
        func_real_list2, func_imag_list2, func_real_list3, func_imag_list3, func_real_list4, func_imag_list4) for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)


def G(chi, chi_star, chi_sigma):

    window = sampling_cosmo.Wg(chi, chi_star, chi_sigma)
    growth = sampling_cosmo.D_class(chi)

    return window * growth 

def Gf(chi, chi_star, chi_sigma):

    return G(chi, chi_star, chi_sigma)*sampling_cosmo.f_class(chi)

def G_Limber_RSD(l, chi, chi_star, chi_sigma):

    term1 = G(chi, chi_star, chi_sigma)
    term2 = Gf(chi, chi_star, chi_sigma)
    term3 = Gf( (2*l-3)/(2*l+1)*chi, chi_star, chi_sigma)
    term4 = Gf( (2*l+5)/(2*l+1)*chi, chi_star, chi_sigma)

    coef2 = (2*l**2 + 2*l - 1)/(2*l-1)/(2*l+3)
    coef3 = (l-1)*l / (2*l-1) / np.sqrt((2*l-3)*(2*l+1))
    coef4 = (l+1)*(l+2)/(2*l+3)/np.sqrt((2*l+5)*(2*l+1))

    return term1 + coef2 * term2 - coef3*term3 - coef4*term4

def Pk_potent1(kh):

    return sampling_cosmo.default_cosmo.Plin(kh)

def Power_spectrum_RSD_intpart(kh_par, l, chi):

    kh = np.sqrt(kh_par**2 + (l/(chi))**2)

    return Pk_potent1(kh)

def CRSD_Limber(l, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2, chi_min, chi_max):

    def integrand(chi):

        Geff1 = G_Limber_RSD(l, chi, chi_avg1, chi_sigma1)
        Geff2 = G_Limber_RSD(l, chi, chi_avg2, chi_sigma2)
        pk = Power_spectrum_RSD_intpart(0, l+0.5, chi)

        return Geff1 * Geff2 * pk / chi**2
    
    return quad(integrand, chi_min, chi_max)[0]