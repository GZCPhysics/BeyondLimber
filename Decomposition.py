'''
In this script, we explicitly calculate the special functions used in decomposing linear power spectrum
Note that these calculations do not depend on any choice of cosmology.
'''

from scipy.interpolate import interp1d
import numpy as np
from mpmath import besselk, rgamma, mp
#import time

mp.dps=25
mp.pretty=True

'''
Nmax = 200
nu_n_array = np.load('./nu_n_array.npy')
print('The complex frequency array is read. The corresponding length is:', len(nu_n_array))
'''

np_rgamma = np.vectorize(rgamma)
np_besselk = np.vectorize(besselk)

def M1i(x_i, nu_i):
    '''
    This function calculates the single term when evaluating expanding FT of power spectrum
    According to the note in mpmath, z=0 is a pole for Gamma(z) function, but 1/Gamma should be 0,
    therefore we adapted the alternative rgamma(z) for 1/Gamma in the mpmath package,
    which deals with this pole.

    Params:
    x_i: is the redefined parameter, the same as delta_chi*l_tilde
    nu_i: is the ith complex order from power expansion
    c_i: is the ith coeffecient

    Return:
    A complex number
    '''
    
    number = (2 * x_i)**(0.5*nu_i+0.5)*np_besselk(-0.5-0.5*nu_i, x_i)*np_rgamma(-0.5*nu_i)/np.sqrt(np.pi)
    return np.array([num.real for num in number]), np.array([num.imag for num in number])

def special_func1_interp(nu_n_array, x_array, Nmax, Mi):
    '''
    Contains of a list of interpolated real functions, corresponding to the real part and imaginary part of the modifed function
    '''
    func_real_list = []
    func_imag_list = []
    
    for i in range(1, int(Nmax/2)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i+int(Nmax/2)]
        y_real_array, y_imag_array = Mi(x_array, nu_i)
        func_real_list.append(interp1d(x_array, y_real_array))
        func_imag_list.append(interp1d(x_array, y_imag_array))
    '''
    for i in range(int(Nmax)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i]
        y_real_array, y_imag_array = M1i(x_array, nu_i)
        func_real_list.append(interp1d(x_array, y_real_array))
        func_imag_list.append(interp1d(x_array, y_imag_array))
    '''
    return func_real_list, func_imag_list


def M2i(x_i, nu_i):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    number = (2/x_i)**(0.5*nu_i+0.5)*np_besselk(-0.5-0.5*nu_i, x_i)*np_rgamma(-0.5*nu_i)/np.sqrt(np.pi)
    return [num.real for num in number], [num.imag for num in number]

def special_func2_interp(nu_n_array, x_array, Nmax):
    '''
    Contains of a list of interpolated real functions, corresponding to the real part and imaginary part of the modifed function
    Note that x_array is always non-negative because of the certain symmetry. And x_array[0] must be 0
    ToDo:
    Add a term that could enable the calculation for x=0
    '''
    func_real_list = []
    func_imag_list = []

    for i in range(int(Nmax)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i]
        y_real_array, y_imag_array = M2i(x_array, nu_i)
        #y_real_0, y_imag_0 = M2i(np.array([1e-6]), nu_i) #This should be improved to actual 0 in the short future
        func_real_list.append(interp1d(x_array, y_real_array))
        func_imag_list.append(interp1d(x_array, y_imag_array))
    
    return func_real_list, func_imag_list

#Functions for CMBlensing scenario
#################################################################################################################
def M2i_lensing(x_i, nu_i_l):
    '''
    This function calculates the single term when evaluating expanding FT of power spectrum
    According to the note in mpmath, z=0 is a pole for Gamma(z) function, but 1/Gamma should be 0,
    therefore we adapted the alternative rgamma(z) for 1/Gamma in the mpmath package,
    which deals with this pole.

    Params:
    x_i: is the redefined parameter, the same as delta_chi*l_tilde
    nu_i_l: is the ith complex order from power expansion of matter power spectrum, 
            when we try to calculate lensing, we need to modify it by minus 4.
    c_i: is the ith coeffecient

    Return:
    A complex number
    '''
    nu_i = nu_i_l - 4
    number = (2 / x_i)**(0.5*nu_i+0.5)*np_besselk(-0.5-0.5*nu_i, x_i)*np_rgamma(-0.5*nu_i)/np.sqrt(np.pi)
    return np.array([num.real for num in number]), np.array([num.imag for num in number])

def special_func_lensing_interp(nu_n_array, x_array, Nmax):
    '''
    Contains of a list of interpolated real functions, corresponding to the real part and imaginary part of the modifed function
    '''
    func_real_list = []
    func_imag_list = []
    for i in range(int(Nmax)+1):  #Here we only take whole of the coeffecient array
        nu_i = nu_n_array[i]
        y_real_array, y_imag_array = M2i_lensing(x_array, nu_i)
        func_real_list.append(interp1d(x_array, y_real_array))
        func_imag_list.append(interp1d(x_array, y_imag_array))
    
    return func_real_list, func_imag_list

#Functions for galaxy clustering RSD scenario
#################################################################################################################
def K2i(x_i, nu_i):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    number = np_besselk(-0.5-0.5*nu_i, x_i)*x_i**(-0.5-0.5*nu_i)
    return np.array([complex(num.real+1j*num.imag) for num in number])

def K3i(x_i, nui):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    nu_i=nui-2

    return K2i(x_i, nu_i)

def K4i(x_i, nui):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    nu_i=nui-4
    return K2i(x_i, nu_i)

def M2i_interp(x_array, nu_n_array, K2i_array, Nmax):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    func_real_list = []
    func_imag_list = []
    for i in range(int(Nmax)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i]
        Prefactor = complex(2**(0.5*nu_i+0.5)*np_rgamma(-0.5*nu_i)/np.sqrt(np.pi))
        y_array = Prefactor*K2i_array[i]
        #y_real_array, y_imag_array = Prefactor*K2i_array[i].real, Prefactor*K2i_array[i].imag
        #y_real_0, y_imag_0 = M2i(np.array([1e-6]), nu_i) #This should be improved to actual 0 in the short future
        func_real_list.append(interp1d(x_array, y_array.real))
        func_imag_list.append(interp1d(x_array, y_array.imag))
    
    return func_real_list, func_imag_list

def M3i_interp(x_array, nu_n_array, K2i_array, K3i_array, Nmax):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    func_real_list = []
    func_imag_list = []
    for i in range(int(Nmax)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i]
        Prefactor = -complex(2**(0.5*nu_i-0.5)*np_rgamma(-0.5*nu_i+1)/np.sqrt(np.pi))
        y_array = Prefactor*(K3i_array[i] + nu_i*K2i_array[i])
        #y_real_0, y_imag_0 = M2i(np.array([1e-6]), nu_i) #This should be improved to actual 0 in the short future
        func_real_list.append(interp1d(x_array, y_array.real))
        func_imag_list.append(interp1d(x_array, y_array.imag))
    
    return func_real_list, func_imag_list

def M4i_interp(x_array, nu_n_array, K2i_array, K3i_array, K4i_array, Nmax):
    '''
    For x_i not equal to zero.
    In order to avoid the dchi=0 pole in the following calculation
    '''
    func_real_list = []
    func_imag_list = []
    for i in range(int(Nmax)+1):  #Here we only take half of the coeffecient array, since it is symmetric
        nu_i = nu_n_array[i]
        Prefactor = complex(2**(0.5*nu_i-1.5)*np_rgamma(-0.5*nu_i+2)/np.sqrt(np.pi))
        y_array = Prefactor*(K4i_array[i] + 2*(nu_i-2)*K3i_array[i] + nu_i*(nu_i-2)*K2i_array[i])
        #y_real_0, y_imag_0 = M2i(np.array([1e-6]), nu_i) #This should be improved to actual 0 in the short future
        func_real_list.append(interp1d(x_array, y_array.real))
        func_imag_list.append(interp1d(x_array, y_array.imag))
    
    return func_real_list, func_imag_list


'''
#Here we do the pre-calulcation of all these modifed functions, given the ith expansion term
#For lensing case, we just need to call the lensing functions instead.
x_min = 2e-5
x_max = 500.1/(0.2)
N_sample = 1000
x_test = 10** np.array([np.log10(x_min) + np.log10(x_max/x_min)/N_sample*i for i in range(N_sample)])
 #Sampling in log scale, changing the sampling points will significantly change the time of evaluation

print('Number of interpolation sampling points are:', N_sample)
print('Modified arguments are in the interval of', '[', x_min, ',', x_max, ']')
print('Creating the modified functions...')
start = time.time()
func_real_list, func_imag_list = special_func_interp(nu_n_array, x_test, Nmax = Nmax)
end = (time.time()-start)/60
print('Function tools activated.')
print('Time consumed:', end, 'min')
'''