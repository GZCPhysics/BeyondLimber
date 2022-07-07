
import numpy as np
import time
from scipy.integrate import simps
from mpmath import hyp2f1, gamma, rgamma
np_hyp2f1 = np.vectorize(hyp2f1)

c_n_array = np.load('./c_n_array')
nu_n_array = np.load('./nu_n_array.npy')
Nmax=200

def I_l_cluster(l, nu, t):
    '''
    This is the special function derived from integral of spherical harmonics
    t<1
    '''
    term1 = 2**(nu+2)*gamma(l+0.5*(nu+3))*rgamma(0.5*(-nu))/gamma(l+1.5)
    term2 = hyp2f1(0.5*(nu+2), l+0.5*(nu+3), l+1.5, t**2)

    number = np.pi/4 * term1*term2 * t**l
    #return np.array([float(num.real) for num in number]), np.array([float(num.imag) for num in number])
    return float(number.real)+1j*float(number.imag)
def I_l_cluster_extra(l,nu,t):
    '''
    This is the analytical extrapolation of the above equation for t>1 case
    '''
    
    number = t**(-nu)*I_l_cluster(l,nu,1/t)
    return number.real

np_I_l_cluster = np.vectorize(I_l_cluster)
np_I_l_cluster_extra = np.vectorize(I_l_cluster_extra)

def I_l_mesh(l, nu, chi_chi, dchi_dchi):

    chi1 = chi_chi + 0.5*dchi_dchi
    chi2 = chi_chi - 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi1[:int(Ndchi/2),:]/chi2[:int(Ndchi/2),:] #dchi negative part
    tt2 = chi2[int(Ndchi/2):,:]/chi1[int(Ndchi/2):,:] #dchi positive part

    mesh1 = chi2[:int(Ndchi/2),:]**(-3-nu)*np_I_l_cluster(l,nu,tt1)
    mesh2 = chi1[int(Ndchi/2):,:]**(-3-nu)*np_I_l_cluster(l,nu,tt2)

    return np.array(list(mesh1)+list(mesh2))

def curly_C_cluster(l, chi, dchi_array, c_n_array):
    '''
    Sum over all the complex frequencies to get the final curly C
    '''
    chi1 = chi + 0.5*dchi_array
    chi2 = chi - 0.5*dchi_array
    tt = chi2/chi1
    Cl_array_array = np.array([c_n_array[i]*chi1**(-3-nu_n_array[i])*np_I_l_cluster(l, nu_n_array[i], tt) for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array.real

def curly_C_cluster_mesh(l, chi_chi, dchi_dchi, c_n_array):
    
    Cl_array_0 = I_l_mesh(l,nu_n_array[int(Nmax/2)],chi_chi, dchi_dchi)*c_n_array[int(Nmax/2)]
    Cl_array_array = np.array([I_l_mesh(l,nu_n_array[int(Nmax/2)+i],chi_chi, dchi_dchi)*c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_array = 2*np.sum(Cl_array_array, axis=0)+Cl_array_0
    return Cl_array.real


def I_l_lensing(l, nu, t):
    '''
    This is the special function derived from integral of spherical harmonics
    t<1
    '''
    term1 = 2**(nu-2)*gamma(l+0.5*(nu-1))/gamma(0.5*(4-nu))/gamma(l+1.5)
    term2 = np_hyp2f1(0.5*(nu-2), l+0.5*(nu-1), l+1.5, t**2)

    number = np.pi/4 * term1*term2 * t**l
    return float(number.real)+1j*float(number.imag)

np_I_l_lensing = np.vectorize(I_l_lensing)

def curly_C_lensing(l, chi, dchi_array):
    '''
    Sum over all the complex frequencies to get the final curly C
    '''
    chi1 = chi + 0.5*dchi_array
    chi2 = chi - 0.5*dchi_array
    tt = chi2/chi1
    Cl_array_array = np.array([c_n_array[i]*chi1**(1-nu_n_array[i])*(I_l_lensing(l, nu_n_array[i], tt)[0] + 1j*I_l_lensing(l, nu_n_array[i], tt)[1] )for i in range(int(Nmax)+1)])
    Cl_array = np.sum(Cl_array_array, axis=0)

    return Cl_array.real

def I_l_lensing_mesh(l, nu, chi_chi, dchi_dchi):

    chi1 = chi_chi + 0.5*dchi_dchi
    chi2 = chi_chi - 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi1[:int(Ndchi/2),:]/chi2[:int(Ndchi/2),:] #dchi negative part
    tt2 = chi2[int(Ndchi/2):,:]/chi1[int(Ndchi/2):,:] #dchi positive part

    mesh1 = chi2[:int(Ndchi/2),:]**(1-nu)*np_I_l_lensing(l,nu,tt1)
    mesh2 = chi1[int(Ndchi/2):,:]**(1-nu)*np_I_l_lensing(l,nu,tt2)

    return np.array(list(mesh1)+list(mesh2))

def curly_C_lensing_mesh(l, chi_chi, dchi_dchi, c_n_array):
    
    #Cl_array_0 = I_l_mesh(l,nu_n_array[int(Nmax/2)],chi_chi, dchi_dchi)*c_n_array[int(Nmax/2)]
    Cl_array_array = np.array([I_l_lensing_mesh(l,nu_n_array[i],chi_chi, dchi_dchi)*c_n_array[i] for i in range(int(Nmax)+1)])
    
    Cl_array = np.sum(Cl_array_array, axis=0)
    return Cl_array.real


def power_calc_full(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array):
    
    Cl_array = curly_C_cluster_mesh(l, chi_chi, dchi_dchi, c_n_array)

    Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*2/np.pi
    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])
    return results

def full_calc_full(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi, c_n_array = c_n_array):
    '''
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above
    c_n_array: the decomposed coefficients array

    Return:
    An list of angular power spectrum given l_array
    '''
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = mesh_grid_generator3(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_full(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return np.array(power_array)