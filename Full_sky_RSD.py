import numpy as np
import time
from scipy.integrate import simps
from mpmath import hyp2f1, gamma, rgamma
import sys
import multiprocessing as mp

np_hyp2f1 = np.vectorize(hyp2f1)

c_n_array = np.load('./Mesh_Planck18/c_n_array.npy')
nu_n_array = np.load('./Mesh_Planck18/nu_n_array.npy')
Nmax=200

#Open_file_name = sys.argv[1]

#We separately consider the hypter-geomrtric functions and its prefactors, since the former are the most time-consuming part of evaluation

#Note that we have the symmetry over chi1_chi1 and chi2_chi2 mesh, we do not need to calibrate different chi1/chi2 and chi2/chi1, just consider the difference between frequencies

def Hyp_1(l, nu, t):

    number = hyp2f1(0.5*(nu+2), l+0.5*(nu+3), l+1.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_2(l, nu, t):

    number = hyp2f1(0.5*(nu), l+0.5*(nu+1), l+1.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_3(l, nu, t):

    number = hyp2f1(0.5*(nu), l+0.5*(nu+3), l+1.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_4(l, nu, t):

    number = hyp2f1(0.5*(nu+2), l+0.5*(nu+3), l+2.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_5(l, nu, t):

    number = hyp2f1(0.5*(nu), l+0.5*(nu+3), l+2.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_6(l, nu, t):

    number = hyp2f1(0.5*(nu-2), l+0.5*(nu+1), l+1.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_7(l, nu, t):

    number = hyp2f1(0.5*(nu), l+0.5*(nu+1), l+2.5, t**2)

    return float(number.real)+1j*float(number.imag)

def Hyp_8(l, nu, t):

    number = hyp2f1(0.5*(nu-2), l+0.5*(nu-1), l+1.5, t**2)

    return float(number.real)+1j*float(number.imag)

#Now we vectorize the above functions:
np_Hyp1 = np.vectorize(Hyp_1)
np_Hyp2 = np.vectorize(Hyp_2)
np_Hyp3 = np.vectorize(Hyp_3)
np_Hyp4 = np.vectorize(Hyp_4)
np_Hyp5 = np.vectorize(Hyp_5)
np_Hyp6 = np.vectorize(Hyp_6)
np_Hyp7 = np.vectorize(Hyp_7)
np_Hyp8 = np.vectorize(Hyp_8)

def Pre_1(l, nu):

    number = np.pi/4 * 2**(nu+2)*gamma(l+0.5*(nu+3))*rgamma(0.5*(-nu))/gamma(l+1.5)

    return float(number.real)+1j*float(number.imag)

def Pre_2(l, nu):

    number = np.pi/4 * 2**(nu)*gamma(l+0.5*(nu+1))*rgamma(0.5*(-nu+2))/gamma(l+1.5)

    return float(number.real)+1j*float(number.imag)

def Pre_3(l, nu):

    number = np.pi/4 * 2**(nu+1)*gamma(l+0.5*(nu+3))*rgamma(0.5*(-nu+2))/gamma(l+1.5)

    return float(number.real)+1j*float(number.imag)

def Pre_4(l, nu):

    number = np.pi/4 * 2**(nu+1)*gamma(l+0.5*(nu+3))*rgamma(0.5*(-nu))/gamma(l+2.5)

    return float(number.real)+1j*float(number.imag)

def Pre_5(l, nu):

    number = np.pi/4 * 2**(nu)*gamma(l+0.5*(nu+3))*rgamma(0.5*(-nu+2))/gamma(l+2.5)

    return float(number.real)+1j*float(number.imag)

def Pre_6(l, nu):

    number = np.pi/4 * 2**(nu-1)*gamma(l+0.5*(nu+1))*rgamma(0.5*(-nu+4))/gamma(l+1.5)

    return float(number.real)+1j*float(number.imag)

def Pre_7(l, nu):

    number = np.pi/4 * 2**(nu-1)*gamma(l+0.5*(nu+1))*rgamma(0.5*(-nu+2))/gamma(l+2.5)

    return float(number.real)+1j*float(number.imag)

def Pre_8(l, nu):

    number = np.pi/4 * 2**(nu-2)*gamma(l+0.5*(nu-1))*rgamma(0.5*(-nu+4))/gamma(l+1.5)

    return float(number.real)+1j*float(number.imag)


def I_half_mesh1(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.1 and 1st part of Eq.1
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_1(l, nu) * tt1**l * np_Hyp1(l, nu, tt1)

    return I_mesh1

def I_half_mesh2(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.2,3 and 1st part of Eq.2,3
    '''
    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part chi1>chi2
    I_mesh1 = Pre_2(l, nu) * tt1**l * np_Hyp2(l, nu, tt1)

    return I_mesh1

def I_half_mesh3(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.5 and 1st part of Eq.4
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_3(l, nu) * tt1**l * np_Hyp3(l, nu, tt1)

    return I_mesh1

def I_half_mesh4(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.4 and 1st part of Eq.5
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_4(l, nu) * tt1**(l+1) * np_Hyp4(l, nu, tt1)

    return I_mesh1

def I_half_mesh5(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.6 and 1st part of Eq.6
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_5(l, nu) * tt1**(l+1) * np_Hyp5(l, nu, tt1)

    return I_mesh1

def I_half_mesh6(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.8 and 1st part of Eq.7
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_6(l, nu) * tt1**l * np_Hyp6(l, nu, tt1)

    return I_mesh1

def I_half_mesh7(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.7 and 1st part of Eq.8
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_7(l, nu) * tt1**(l+1) * np_Hyp7(l, nu, tt1)

    return I_mesh1

def I_half_mesh8(l, nu, chi_chi, dchi_dchi):
    '''
    Used for 2nd part of Eq.9 and 1st part of Eq.9
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    I_mesh1 = Pre_8(l, nu) * tt1**l * np_Hyp8(l, nu, tt1)

    return I_mesh1

def save_half_mesh(l, chi_chi, dchi_dchi, func_id, path):

    if func_id == 1:
        I_half = I_half_mesh1
    if func_id == 2:
        I_half = I_half_mesh2
    if func_id == 3:
        I_half = I_half_mesh3
    if func_id == 4:
        I_half = I_half_mesh4
    if func_id == 5:
        I_half = I_half_mesh5
    if func_id == 6:
        I_half = I_half_mesh6
    if func_id == 7:
        I_half = I_half_mesh7
    if func_id == 8:
        I_half = I_half_mesh8

    Cl_array = np.array([I_half(l,nu_n_array[int(Nmax/2)+i], chi_chi, dchi_dchi) for i in range(int(Nmax/2)+1)])

    np.save(path+'I_func%d_l%d'%(func_id, l), Cl_array)

def Equation1(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path1):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path1) #With shape(101, 50, 50)
    I_mesh2_array = I_mesh1_array[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-3-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-3-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-3-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-3-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = Cl_array*(1+f1_f1)*(1+f2_f2)*Wg1_Wg1*Wg2_Wg2*2/np.pi

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation2(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path2):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path2) #With shape(101, 50, 50)
    I_mesh2_array = I_mesh1_array[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = -(2/np.pi)*Cl_array*(1+f1_f1)*f2_f2*Wg1_Wg1*Wg2_Wg2*l*(l-1)/chi2**2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation3(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path2):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path2) #With shape(101, 50, 50)
    I_mesh2_array = I_mesh1_array[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = -(2/np.pi)*Cl_array*(1+f2_f2)*f1_f1*Wg1_Wg1*Wg2_Wg2*l*(l-1)/chi1**2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation4(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path3, path4):
    '''
    path4 is for upper half mesh of mesh4
    path3 is for upper half mesh of mesh3, need rotate to the lower part
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path4) #With shape(101, 50, 50)
    I_mesh2_array = np.load(path3)[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-2-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-2-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-2-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-2-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = -(2/np.pi)*2*Cl_array*(1+f1_f1)*f2_f2*Wg1_Wg1*Wg2_Wg2/chi2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation5(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path3, path4):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path3) #With shape(101, 50, 50)
    I_mesh2_array = np.load(path4)[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-2-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-2-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-2-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-2-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = -(2/np.pi)*2*Cl_array*(1+f2_f2)*f1_f1*Wg1_Wg1*Wg2_Wg2/chi1

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation6(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path5):
    '''
    path5 is for upper half mesh of mesh5
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path5) #With shape(101, 50, 50)
    I_mesh2_array = I_mesh1_array[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = (2/np.pi)*4*Cl_array*f1_f1*f2_f2*Wg1_Wg1*Wg2_Wg2/chi1/chi2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation7(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path6, path7):
    '''
    path7 is for upper half mesh of mesh7
    path6 is for upper half mesh of mesh6, need rotate to the lower part
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path7) #With shape(101, 50, 50)
    I_mesh2_array = np.load(path6)[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-0-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-0-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-0-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-0-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = (2/np.pi)*2*Cl_array*f1_f1*f2_f2*Wg1_Wg1*Wg2_Wg2*l*(l-1)/chi2/chi1**2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation8(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path6, path7):
    '''
    path7 is for upper half mesh of mesh7, need rotate to the lower part
    path6 is for upper half mesh of mesh6
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path6) #With shape(101, 50, 50)
    I_mesh2_array = np.load(path7)[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(-0-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(-0-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(-0-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(-0-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = (2/np.pi)*2*Cl_array*f1_f1*f2_f2*Wg1_Wg1*Wg2_Wg2*l*(l-1)/chi2**2/chi1

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results

def Equation9(l, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path8):
    '''
    path5 is for upper half mesh of mesh5
    '''

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    I_mesh1_array = np.load(path8) #With shape(101, 50, 50)
    I_mesh2_array = I_mesh1_array[:, ::-1,:]

    Cl_up_0 = I_mesh1_array[0,:,:] * chi1[:int(Ndchi/2),:]**(1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_up_array = np.array([I_mesh1_array[i,:,:] * chi1[:int(Ndchi/2), :]**(1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])

    Cl_low_0 = I_mesh2_array[0,:,:] * chi2[int(Ndchi/2):,:]**(1-nu_n_array[int(Nmax/2)]) * c_n_array[int(Nmax/2)]
    Cl_low_array = np.array([I_mesh2_array[i,:,:] * chi2[int(Ndchi/2):, :]**(1-nu_n_array[int(Nmax/2)+i]) * c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    
    Cl_up = 2*np.sum(Cl_up_array, axis=0)+Cl_up_0
    Cl_low = 2*np.sum(Cl_low_array, axis=0)+Cl_low_0

    Cl_array = np.array(list(Cl_up)+list(Cl_low)).real

    Simp_array = (2/np.pi)*Cl_array*f1_f1*f2_f2*Wg1_Wg1*Wg2_Wg2* l**2 * (l-1)**2 /chi1**2/chi2**2

    intover_dchi = np.array([simps(Simp_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results = simps(intover_dchi, chi_chi[0, :])

    return results


def I_l_mesh1(l, nu, chi_chi, dchi_dchi):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    tt2 = chi1[int(Ndchi/2):,:]/chi2[int(Ndchi/2):,:] #dchi positive part

    I_mesh1 = Pre_1(l, nu) * tt1**l * np_Hyp1(l, nu, tt1)
    I_mesh2 = I_mesh1[::-1,:]

    mesh1 = chi1[:int(Ndchi/2),:]**(-3-nu)*I_mesh1
    mesh2 = chi2[int(Ndchi/2):,:]**(-3-nu)*I_mesh2

    return np.array(list(mesh1)+list(mesh2))

def I_l_mesh2(l, nu, chi_chi, dchi_dchi):

    chi1 = chi_chi - 0.5*dchi_dchi
    chi2 = chi_chi + 0.5*dchi_dchi
    Ndchi = len(dchi_dchi[:,0])
    Nchi = len(chi_chi[0,:])

    tt1 = chi2[:int(Ndchi/2),:]/chi1[:int(Ndchi/2),:] #dchi negative part
    tt2 = chi1[int(Ndchi/2):,:]/chi2[int(Ndchi/2):,:] #dchi positive part

    I_mesh1 = Pre_2(l, nu) * tt1**l * np_Hyp2(l, nu, tt1)
    I_mesh2 = I_mesh1[::-1,:]

    mesh1 = chi1[:int(Ndchi/2),:]**(-1-nu)*I_mesh1
    mesh2 = chi2[int(Ndchi/2):,:]**(-1-nu)*I_mesh2

    return np.array(list(mesh1)+list(mesh2))


def curly_C_mesh(l, chi_chi, dchi_dchi, I_l_mesh, c_n_array):
    
    Cl_array_0 = I_l_mesh(l,nu_n_array[int(Nmax/2)]-4,chi_chi, dchi_dchi)*c_n_array[int(Nmax/2)]
    Cl_array_array = np.array([I_l_mesh(l,nu_n_array[int(Nmax/2)+i]-4,chi_chi, dchi_dchi)*c_n_array[int(Nmax/2)+i] for i in range(1, int(Nmax/2)+1)])
    Cl_array = 2*np.sum(Cl_array_array, axis=0)+Cl_array_0
    #Cl_array_array = np.array([I_l_mesh(l,nu_n_array[i]-4,chi_chi, dchi_dchi)*c_n_array[i] for i in range(int(Nmax)+1)])
    
    #Cl_array = np.sum(Cl_array_array, axis=0)
    return Cl_array.real
    
def power_calc_separate(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, G1_G1, G2_G2, I_l_mesh, c_n_array):
    
    Cl_array = curly_C_mesh(l, chi_chi, dchi_dchi, I_l_mesh, c_n_array)

    W1_W1 = D1_D1 + Wg1_Wg1*(l*(l+1)-2)/(chi_chi-0.5*dchi_dchi)**2 
    W2_W2 = D2_D2 + Wg2_Wg2*(l*(l+1)-2)/(chi_chi+0.5*dchi_dchi)**2 

    Simp0_array = Cl_array*W1_W1*W2_W2*2/np.pi
    Simp1_array = Cl_array*W1_W1*G2_G2*2/np.pi
    Simp2_array = Cl_array*G1_G1*W2_W2*2/np.pi
    Simp3_array = Cl_array*G1_G1*G2_G2*2/np.pi

    #Simp_array = D1_D1*D2_D2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    intover_dchi0 = np.array([simps(Simp0_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results0 = simps(intover_dchi0, chi_chi[0, :])

    intover_dchi1 = np.array([simps(Simp1_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results1 = simps(intover_dchi1, chi_chi[0, :])

    intover_dchi2 = np.array([simps(Simp2_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results2 = simps(intover_dchi2, chi_chi[0, :])

    intover_dchi3 = np.array([simps(Simp3_array[:,i], dchi_dchi[:,i]) for i in range(len(chi_chi[0,:]))])
    results3 = simps(intover_dchi3, chi_chi[0, :])

    results = results0+results1+results2+results3

    file = open(Open_file_name,'a')
    file.write('%d,%s,%s,%s,%s,%s'%(l,str(results),str(results0), str(results1), str(results2), str(results3)))
    file.write('\n')
    file.close()

    #return results0.real, results1.real, results2.real, results3.real

if __name__ == '__main__':

    #We test multiprocessing here
    
    Mesh_file_name = './Mesh_Planck18/Mesh_1125_RSD.npy'
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2 = np.load(Mesh_file_name)
    
    l_list = [2 + 2*i for i in range(7)]
    '''
    path = './Half_mesh_1125/'
    for li in l_list:
        p1 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 1, path))
        p2 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 2, path))
        p3 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 3, path))
        p4 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 4, path))
        p5 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 5, path))
        p6 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 6, path))
        p7 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 7, path))
        p8 = mp.Process(target=save_half_mesh, args=(li, chi_chi, dchi_dchi, 8, path))
    
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        p8.join()
    '''
    Cl_list = []
    for li in l_list:
        path1 = './Half_mesh_1125/I_func1_l%d.npy'%li
        path2 = './Half_mesh_1125/I_func2_l%d.npy'%li
        path3 = './Half_mesh_1125/I_func3_l%d.npy'%li
        path4 = './Half_mesh_1125/I_func4_l%d.npy'%li
        path5 = './Half_mesh_1125/I_func5_l%d.npy'%li
        path6 = './Half_mesh_1125/I_func6_l%d.npy'%li
        path7 = './Half_mesh_1125/I_func7_l%d.npy'%li
        path8 = './Half_mesh_1125/I_func8_l%d.npy'%li

        Cli1 = Equation1(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path1)
        Cli2 = Equation2(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path2)
        Cli3 = Equation3(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path2)
        Cli4 = Equation4(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path3, path4)
        Cli5 = Equation5(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path3, path4)
        Cli6 = Equation6(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path5)
        Cli7 = Equation7(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path6,path7)
        Cli8 = Equation8(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path6,path7)
        Cli9 = Equation9(li, chi_chi, dchi_dchi, f1_f1, f2_f2, Wg1_Wg1, Wg2_Wg2, c_n_array, path8)
        
        print(li, [Cli1,Cli2,Cli3,Cli4,Cli5,Cli6,Cli7,Cli8,Cli9])
        Cli = Cli1+Cli2+Cli3+Cli4+Cli5+Cli6+Cli7+Cli8+Cli9


        Cl_list.append([Cli,Cli1,Cli2,Cli3,Cli4,Cli5,Cli6,Cli7,Cli8,Cli9])
    
    np.save('./Cl_1125_RSD_test', Cl_list)




    
    
    




    

    



    
