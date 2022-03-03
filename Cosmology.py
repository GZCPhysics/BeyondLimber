'''
In this script, we define the cosmology class and the corresponding functions that we are going to use in the final calculation.
'''

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import sys, platform, os
import numpy as np
from colossus.cosmology import cosmology
import params as pa
import cmath
from scipy.integrate import quad as scquad
from scipy.integrate import dblquad as scdblquad
import time
from scipy.integrate import simps


camb_path = os.path.realpath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

#Read and generate Linear power spectrum function from files

class MyCosmology(object):

    #Here we defined several default physical parameters as global constant
    #Currently, we have fixed our cosmology
    #This Class contains the cosmology parameters, linear power spectrum, distances and linear growth factor
    #Here we have used pyCAMB package and colussus package to calculate all the cosmology related quantity
    #In order to make it more flexible, say, for CLASS people. They can just modify this scrip by replacing 
    #this class with other that you are familiar with.
    def __init__(self, c=pa.c, zCMB=pa.zCMB, zmax=pa.zmax, h=pa.h, omch2=pa.omch2, ombh2=pa.ombh2, ns=pa.ns, s8=pa.s8):
        self.c = c  #Speed of light, in unit of m/s
        self.zCMB = zCMB 
        self.zmax = zmax
        self.h = h
        self.omch2 = omch2
        self.ombh2 = ombh2
        self.Omc = omch2/h**2
        self.Omb = ombh2/h**2
        self.Omm = omch2/h**2 + ombh2/h**2
        self.ns = ns
        self.s8 = s8

        #Setting up CAMB
        #Set up a new set of parameters for CAMB
        self.pars = camb.CAMBparams()
        #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        self.pars.set_cosmology(H0=self.h*100, ombh2=self.ombh2, omch2=self.omch2, mnu=0.06, omk=0, tau=0.06)
        self.pars.InitPower.set_params(As=2.21536e-9, ns=self.ns, r=0)
        self.pars.set_for_lmax(2500, lens_potential_accuracy=0)
        self.results = camb.get_results(self.pars)
        #Setting up colossus
        self.params = {'flat': True, 'H0': 100*h, 'Om0': self.Omm, 'Ob0': self.Omb, 'sigma8': self.s8, 'ns': self.ns}
        cosmology.addCosmology('myCosmo', self.params)
        self.cosmo = cosmology.setCosmology('myCosmo')

        #self.pkdata = np.loadtxt('/Users/cheng/Documents/Researches_at_Cambridge/Limber/1705 2/Mathematica/PCAMBz0.txt')
        self.pkdata = np.loadtxt('./PCAMBz0 copy.txt')
        self.kh = self.pkdata[:,0]
        self.pk = self.pkdata[:,1]
        self.kappa_array = [-10.+0.01*i for i in range(int((-np.log(self.kh[0])/np.log(10.)+0.01)/0.01)+1)]
        self.lowkh = np.array([10**kappa for kappa in self.kappa_array])
        self.Plowk = np.array([self.pk[0]*(10**kappa/self.kh.min())**self.ns for kappa in self.kappa_array])
        self.kh_new = np.array(list(self.lowkh)+list(self.kh[0:]))
        self.Pk_new = np.array(list(self.Plowk)+list(self.pk[0:]))
        #Initialize the linear power spectrum P(kh) in our given universe
        self.Plin = interp1d(self.kh_new, self.Pk_new)

    def chi(self, z):
        '''
        Here we only consider the radial comoving distance, 
        which is the same as transverse one in a flat universe

        The returned unit is [Mpc/h]
        '''
        return self.cosmo.comovingDistance(z_min=0.0, z_max=z, transverse=False)
    
    def red_at_chi(self, chi_x):
        '''
        Return: 
        
        A function that could give us the redshift of a given comoving distance
        '''

        z = self.results.redshift_at_comoving_radial_distance(chi_x/self.h)
        return z
    
    def a(self, chi):
        '''
        The unit of chi should be [Mpc/h]
        '''
        z = self.red_at_chi(chi)
        return 1/(1+z)
    
    def HH(self, z):
        '''
        Here the returned value has a unit of [h/Mpc]
        '''
        #return results.hubble_parameter(z)/(c*1e-3*h) #Update to this if we could define a camb environment
        return self.cosmo.Hz(z)/(self.c*1e-3*self.h)
    
    #Now we define the growth factors
    def Dg_z(self, z):
        '''
        This is the unnormalized growth factor given a redshift z
        '''
        return self.cosmo.growthFactorUnnormalized(z)

    def Dg_chi(self, chi):
        '''
        The unit of chi here is [Mpc/h].
        Therefore, when applying the function of camb, we need to convert it to the unit of [Mpc]
        '''
        z = self.red_at_chi(chi)
        return self.cosmo.growthFactorUnnormalized(z)

    def Dg_norm(self, chi):
        '''
        This is the normalized growth factor
        '''
        return self.Dg_chi(chi)/self.Dg_z(0.00001)
    
#To make it more flexible in choosing different preferences over class, we separatly write the following calculations in another class
class Window_function(object):
    '''
    Aimed to create a set of different types of window functions.
    For now, we only have Guassian window functions for galaxy surveys 
    '''
    def __init__(self):
        pass
 
    def Wg(self, chi, chi_avg, chi_sigma):
        '''
        Here we consider a Gaussian window function
        '''
        term1 = 1/(np.sqrt(2*np.pi))/chi_sigma
        term2 = np.exp(-(chi-chi_avg)**2/(2*chi_sigma**2))
        return term1*term2

class Sampling(object):

    def __init__(self, c=pa.c, zCMB=pa.zCMB, zmax=pa.zmax, h=pa.h, omch2=pa.omch2, ombh2=pa.ombh2, ns=pa.ns, s8=pa.s8):
        
        #Introducing cosmology, where the parameters could vary and even the choice of cosmology class could be changed
        self.default_cosmo = MyCosmology(c, zCMB, zmax, h, omch2, ombh2, ns, s8)
        self.Wg = Window_function().Wg
    
    def CoeffTransfer(self, P, b, cst, Nmax, kmin, kmax):
        """
        Params:
        P: function to Fourier transform. Usually, we choose the initial power spectrum in default_cosmo
        b: bias
        cst: constant
        k^{-b}*P(k)+cst will be Fourier transformed
        Nmax: number of frequencies, must be a even number
        kmin: minimum wavenumber
        kmax: maximum wavenumber
        
        Returns:
        C[i,0]: ith coefficient of the Fourier transform
        C[i,1]: ith frequency of the Fourier transform
        """
        delta = np.log(kmax/kmin)/(Nmax-1) #Bin width in logk
        n = [i for i in range(Nmax)]
        m = [int(i - Nmax/2) for i in range(Nmax+1)]
        kn = [kmin * np.exp(n[i] * delta) for i in range(Nmax)]
        eta_m = [b + 2*cmath.pi*1j/(Nmax*delta)*m[i] for i in range(Nmax+1)]
        eta_n = [b + 2*cmath.pi*1j/(Nmax*delta)*n[i] for i in range(Nmax)]

        Pn = np.array([ [kn[i], P(kn[i])*(kn[i]/kmin)**(-b)+cst] for i in range(Nmax)])
        cn = np.array([ np.sum([Pn[k,1]*cmath.exp(-2*cmath.pi*1j*n[i]*n[k]/Nmax)/Nmax for k in range(Nmax)]) for i in range(Nmax)])
        cnsym = []
        for j in range(Nmax+1):
            if m[j] < 0: 
                cnsym.append(kmin**(-eta_m[j])*np.conjugate(cn[-int(m[j])]))
            else:
                cnsym.append(kmin**(-eta_m[j])*cn[int(m[j])])
        cnsym[0] = cnsym[0]/2
        cnsym[-1] = cnsym[-1]/2
        result = np.array([ [cnsym[i], eta_m[i]] for i in range(Nmax+1)])

        return result
    
    def mesh_grid_generator(self, z1, z2, sigma1, sigma2, Nchi, Ndchi):
        '''
        Params:
        z1, z2: the redshifts of our objects
        sigma1, sigma2: the dispersion of our window function
        Nchi, Ndchi: fidicually chosen length of the sampling array

        Return:
        mesh-grids of chi_chi, dchi_dchi, 
        and, growth factors D1_D1, D2_D2.
        '''
        chi_avg1 = self.default_cosmo.chi(z1)
        chi_avg2 = self.default_cosmo.chi(z2)
        chi_sigma1 = sigma1/self.default_cosmo.HH(z1)
        chi_sigma2 = sigma2/self.default_cosmo.HH(z2)
        chi_min = max(chi_avg1-4*chi_sigma1, 0.5)
        chi_max = min(chi_avg2+4*chi_sigma2, self.default_cosmo.chi(10.0))
        chi_array = np.array([chi_min + i*(chi_max-chi_min)/Nchi for i in range(Nchi)])
        dchi_array = np.array(list(-10**np.array([-1+(np.log10(chi_max-chi_min)+1)/Ndchi*i for i in range(Ndchi+1)])[::-1])\
                +list(10**np.array([-1+(np.log10(chi_max-chi_min)+1)/Ndchi*i for i in range(Ndchi+1)])))
        #Create the sample grid
        D1_D1 = np.array([[self.default_cosmo.Dg_norm(chi-0.5*dchi) for chi in chi_array] for dchi in dchi_array])
        D2_D2 = np.array([[self.default_cosmo.Dg_norm(chi+0.5*dchi) for chi in chi_array] for dchi in dchi_array])
        chi_chi, dchi_dchi = np.meshgrid(chi_array, dchi_array)
        grid1 = chi_chi-0.5*dchi_dchi
        grid2 = chi_chi+0.5*dchi_dchi
        Wg1_Wg1 = self.Wg(grid1, chi_avg1, chi_sigma1)
        Wg2_Wg2 = self.Wg(grid2, chi_avg2, chi_sigma2)

        return chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2

'''
default_cosmo = MyCosmology()

def CoeffTransfer(P, b, cst, Nmax, kmin, kmax):
    """
    Params:
    P: function to Fourier transform
    b: bias
    cst: constant
    k^{-b}*P(k)+cst will be Fourier transformed
    Nmax: number of frequencies, must be a even number
    kmin: minimum wavenumber
    kmax: maximum wavenumber

    Returns:
    C[i,0]: ith coefficient of the Fourier transform
    C[i,1]: ith frequency of the Fourier transform
    """
    delta = np.log(kmax/kmin)/(Nmax-1) #Bin width in logk
    n = [i for i in range(Nmax)]
    m = [int(i-Nmax/2) for i in range(Nmax+1)]
    kn = [kmin*np.exp(n[i]*delta) for i in range(Nmax)]
    eta_m = [b + 2*cmath.pi*1j/(Nmax*delta)*m[i] for i in range(Nmax+1)]
    eta_n = [b + 2*cmath.pi*1j/(Nmax*delta)*n[i] for i in range(Nmax)]

    Pn = np.array([ [kn[i], P(kn[i])*(kn[i]/kmin)**(-b)+cst] for i in range(Nmax) ])
    cn = np.array([ np.sum([Pn[k,1]*cmath.exp(-2*cmath.pi*1j*n[i]*n[k]/Nmax)/Nmax for k in range(Nmax)]) for i in range(Nmax)])
    cnsym = []
    for j in range(Nmax+1):
        if m[j]<0: 
            cnsym.append(kmin**(-eta_m[j])*np.conjugate(cn[-int(m[j])]))
        else:
            cnsym.append(kmin**(-eta_m[j])*cn[int(m[j])])
    cnsym[0] = cnsym[0]/2
    cnsym[-1] = cnsym[-1]/2
    result = np.array([ [cnsym[i], eta_m[i]] for i in range(Nmax+1)])

    return result

########################################################################################################
#May consider write these into classes in the future
khmin= 1e-8
khmax = 52.
Nmax = 200
CC = CoeffTransfer(default_cosmo.Plin, 0, 0, Nmax, khmin, khmax)
c_n_array = CC[:, 0]
nu_n_array = CC[:, 1]
print('Linear Power Spectrum at z=0 expanded. The number of expansion terms is: %d'%len(c_n_array))

########################################################################################################
def Wg(chi, chi_avg, chi_sigma):
    """
    Here we consider a Gaussian window function
    """
    term1 = 1/(np.sqrt(2*np.pi))/chi_sigma
    term2 = np.exp(-(chi-chi_avg)**2/(2*chi_sigma**2))
    return term1*term2

import time

def mesh_grid_generator(z1, z2, sigma1, sigma2, Nchi, Ndchi):
    """
    Params:
    z1, z2: the redshifts of our objects
    sigma1, sigma2: the dispersion of our window function
    Nchi, Ndchi: fidicually chosen length of the sampling array

    Return:
    mesh-grids of chi_chi, dchi_dchi, 
    and, growth factors D1_D1, D2_D2.
    """
    chi_avg1 = default_cosmo.chi(z1)
    chi_avg2 = default_cosmo.chi(z2)
    chi_sigma1 = sigma1/default_cosmo.HH(z1)
    chi_sigma2 = sigma2/default_cosmo.HH(z2)
    chi_min = max(chi_avg1-4*chi_sigma1, 0.5)
    chi_max = min(chi_avg2+4*chi_sigma2, default_cosmo.chi(10.0))
    chi_array = np.array([chi_min + i*(chi_max-chi_min)/Nchi for i in range(Nchi)])
    dchi_array = np.array(list(-10**np.array([-1+(np.log10(chi_max-chi_min)+1)/Ndchi*i for i in range(Ndchi+1)])[::-1])\
        +list(10**np.array([-1+(np.log10(chi_max-chi_min)+1)/Ndchi*i for i in range(Ndchi+1)])))
    #Create the sample grid
    D1_D1 = np.array([[default_cosmo.Dg_norm(chi-0.5*dchi) for chi in chi_array] for dchi in dchi_array])
    D2_D2 = np.array([[default_cosmo.Dg_norm(chi+0.5*dchi) for chi in chi_array] for dchi in dchi_array])
    chi_chi, dchi_dchi = np.meshgrid(chi_array, dchi_array)
    grid1 = chi_chi-0.5*dchi_dchi
    grid2 = chi_chi+0.5*dchi_dchi
    Wg1_Wg1 = Wg(grid1, chi_avg1, chi_sigma1)
    Wg2_Wg2 = Wg(grid2, chi_avg2, chi_sigma2)

    return chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2

def power_calc_sampling(l, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, c_n_array):
    """
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
    """
    xx = dchi_dchi*l/(chi_chi+0.5*dchi_dchi)
    Cl_array_array = np.array([ (c_n_array[i+int(Nmax/2)+1]*(func_real_list[i](np.abs(xx))+1j*func_imag_list[i](np.abs(xx))))*\
        np.abs(dchi_dchi)**(-nu_n_array[int(Nmax/2)+1+i]-1) for i in range(int(Nmax/2))])
    Cl_array = np.sum(Cl_array_array, axis=0)

    Simp_array = D1_D1*D2_D2*2*Cl_array*Wg1_Wg1*Wg2_Wg2*(dchi_dchi/chi_chi)**n/chi_chi**2
    results = simps(simps(Simp_array, chi_chi[0,:]), dchi_dchi[:,0])
    return results

def full_calc_sampling(l_array, n, z1, z2, sigma1, sigma2, Nchi, Ndchi):
    """
    Params:
    l_array: The array of multiples we have chosen to consider
    The meaning of rest parameters could be found above

    Return:
    An list of angular power spectrum given l_array
    """
    start1 = time.time()
    chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2 = mesh_grid_generator(z1, z2, sigma1, sigma2, Nchi, Ndchi)
    end1 = time.time()-start1
    print('Time for preparing mesh-grids is:', end1, 's')
    start2 = time.time()
    power_array = [power_calc_sampling(li, n, chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2).real for li in l_array]
    end2 = (time.time()-start2)/len(l_array)
    print('Time for calculating each l is:', end2, 's')

    return power_array


#After initialize the functions we need


#func_real, func_imag = special_func_interp(c_n_array, nu_n_array, x_array, Nmax=200)
'''






