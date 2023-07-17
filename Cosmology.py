'''
In this script, we define the cosmology class and the corresponding functions that we are going to use in the final calculation.
'''
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import sys, platform, os
import numpy as np
#from colossus.cosmology import cosmology
sys.path.append('./cosmo_params/')
import params as pa
import cmath
from scipy.integrate import quad as scquad
from scipy.integrate import dblquad as scdblquad
import time
from scipy.integrate import simps
from numpy import diff
from classy import Class
import scipy
import scipy.special

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
    def __init__(self, c=pa.c, zCMB=pa.zCMB, zmax=pa.zmax, h=pa.h, omch2=pa.omch2, ombh2=pa.ombh2, ns=pa.ns, s8=pa.s8, As = pa.As, tau = 0.0561):
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
        self.As = As
        self.tau = tau

        #Setting up CAMB
        #Set up a new set of parameters for CAMB
        self.pars = camb.CAMBparams()
        #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
        self.pars.set_cosmology(H0=self.h*100, ombh2=self.ombh2, omch2=self.omch2, mnu=0.06, omk=0, tau=self.tau)
        self.pars.InitPower.set_params(As=self.As, ns=self.ns, r=0)
        self.pars.set_for_lmax(2500, lens_potential_accuracy=4)
        self.pars.set_dark_energy(w=-1, wa=0, dark_energy_model='fluid')
        self.pars.NonLinear = model.NonLinear_none
        self.pars.set_matter_power(redshifts=[0.,0.8], kmax=200.0*self.h)
        self.results = camb.get_results(self.pars)
        #Linear spectra
        self.kh, _, self._pk = self.results.get_matter_power_spectrum(minkh=1e-4, maxkh=200.0, npoints = 200)
        self.pk = self._pk[0,:]

        #Set up CLASS
        self.LambdaCDM = Class()
        # pass input parameters
        self.LambdaCDM.set({'omega_b':self.ombh2,'omega_cdm':self.omch2,'h':self.h,'A_s':self.As,'n_s':self.ns,'tau_reio':self.tau})
        self.LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':200.0})
        # run class
        self.LambdaCDM.compute()

        #self.pkdata = np.loadtxt('/Users/cheng/Documents/Researches_at_Cambridge/Limber/1705 2/Mathematica/PCAMBz0.txt')
        #self.pkdata = np.loadtxt('/Users/cheng/Documents/Researches_at_Cambridge/Limber/1705Python/Package/cosmo_params/pk_camb_planck18.txt')
        #self.kh = self.pkdata[:,0]
        #self.pk = self.pkdata[:,1]
        self.kappa_array = [-10.+0.01*i for i in range(int((-np.log(self.kh[0])/np.log(10.)+0.01)/0.01)+1)]
        self.lowkh = np.array([10**kappa for kappa in self.kappa_array])
        self.Plowk = np.array([self.pk[0]*(10**kappa/self.kh.min())**pa.ns for kappa in self.kappa_array])
        self.kh_new = np.array(list(self.lowkh)+list(self.kh[0:]))
        self.Pk_new = np.array(list(self.Plowk)+list(self.pk[0:]))
        #Initialize the linear power spectrum P(kh) in our given universe
        self.Plin = interp1d(self.kh_new, self.Pk_new)
        
        self.aa_array = np.e**np.array([np.log(1/1110)+np.log(1110)/10000*i for i in range(10000)])[::-1]
        self.defchi_array = self.chi(1/self.aa_array-1)
    
    def chi(self, z):
        '''
        Here we only consider the radial comoving distance, 
        which is the same as transverse one in a flat universe

        The returned unit is [Mpc/h]
        '''
        return self.results.comoving_radial_distance(z)*self.h
        #return self.cosmo.comovingDistance(z_min=0.0, z_max=z, transverse=False)
    
    def red_at_chi(self, chi_x):
        '''
        Params:
        chi_x: comoving distance with a unit of [Mpc/h]

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
        return self.results.hubble_parameter(z)/(self.c*1e-3*self.h) #Update to this if we could define a camb environment
        #return self.cosmo.Hz(z)/(self.c*1e-3*self.h)
    
    
    def Psi_normalizer(self, chi_x):
        '''
        This is the normalization factor used in evaluating the power spectrum of lensing potential.
        Params:
        chi_x: the comoving distance, with a unit of [Mpc/h]

        Returns:
        3*Omega_m*HH^2/2
        '''
        z_x = self.red_at_chi(chi_x)
        #Omega_m = self.results.get_Omega('cdm', z_x) + self.results.get_Omega('baryon', z_x)+self.results.get_Omega('neutrino', z_x)+self.results.get_Omega('nu', z_x)
        #Omega_m = self.cosmo.Om(z_x)
        Omega_m = self.results.get_Omega('tot', z_x)  - self.results.get_Omega('de', z_x) #- self.results.get_Omega('photon', z_x)

        return 1.5 * Omega_m * self.HH(z_x)**2 / (1+z_x)**2
    
    def Cl_lensing_camb(self, lmax):
        '''
        This is the implemented function in camb, to calulate the modified full-sky angular power spectrum of CMB lensing potential
        '''

        return self.results.get_lens_potential_cls(lmax=lmax) 

    def D_hypergeom(self, avec, csm):
        bg = csm.get_background()
        Om = csm.Omega0_m()
        if '(.)rho_lambda' in bg:
            Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
        else:
            Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        
        x = Ol/Om*avec**3
        D = avec*scipy.special.hyp2f1(1./3.,1,11./6.,-x)
        D_today = scipy.special.hyp2f1(1./3.,1,11./6.,-Ol/Om)
        return D/D_today

    def D_chi_class(self, chi):
        '''
        The linear growth factor as a function of \chi, normalized to D(0) 
        Generated using CLASS package.
        '''
        avec = 1/(1+self.red_at_chi(chi))

        return self.D_hypergeom(avec,self.LambdaCDM)
    
    def f_hypergeom(self, avec,csm):
        bg = csm.get_background()
        Om = csm.Omega0_m()
        if '(.)rho_lambda' in bg:
            Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
        else:
            Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
            
        x = Ol/Om*avec**3
        D = avec*scipy.special.hyp2f1(1./3.,1,11./6.,-x)
        f = 1.-6./11.*x*avec/D*scipy.special.hyp2f1(4./3.,2,17./6.,-x)
        return f

    def f_chi_class(self, chi):
        '''
        The RSD factor as a function of \chi, in principle f = ln D/ ln a.
        Generated using CLASS package.
        '''
    
        avec = 1/(1+self.red_at_chi(chi))
    
        return self.f_hypergeom(avec,self.LambdaCDM)
    
    
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
    
    def Wlensing(self, chi, chi_star):
        '''
        Here we consider the general used lensing window function
        '''

        return -2*(chi_star-chi)/(chi_star*chi)

class Sampling(object):

    def __init__(self, c=pa.c, zCMB=pa.zCMB, zmax=pa.zmax, h=pa.h, omch2=pa.omch2, ombh2=pa.ombh2, ns=pa.ns, s8=pa.s8, As = pa.As):
        
        #Introducing cosmology, where the parameters could vary and even the choice of cosmology class could be changed
        self.default_cosmo = MyCosmology(c, zCMB, zmax, h, omch2, ombh2, ns, s8, As)
        self.Wg = Window_function().Wg
        self.Wlensing = Window_function().Wlensing
        self.chi_array = np.array([0.01 + (9400-0.01)/9000*i for i in range(9001)])
        self.D_class_array = self.default_cosmo.D_chi_class(self.chi_array)#, self.default_cosmo.LambdaCDM)
        self.f_class_array = self.default_cosmo.f_chi_class(self.chi_array)#, self.default_cosmo.LambdaCDM)
        self.D_class = interp1d(self.chi_array, self.D_class_array)
        self.f_class = interp1d(self.chi_array, self.f_class_array)

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
        This is sampling 2D mesh-grid function over chi and delta_chi,
        with self-adapting range of delta_chi

        Params:
        z1, z2: the redshifts of our objects
        sigma1, sigma2: the dispersion of our window function
        Nchi, Ndchi: fidicually chosen length of the sampling array of chi and dchi

        Return:
        mesh-grids of chi1_chi1, chi2_chi2, 
        and, growth factors D1_D1, D2_D2.
        '''
        chi_avg1 = self.default_cosmo.chi(z1)
        chi_avg2 = self.default_cosmo.chi(z2)
        chi_sigma1 = sigma1/self.default_cosmo.HH(z1)
        chi_sigma2 = sigma2/self.default_cosmo.HH(z2)
        chi_min = max(100, chi_avg1-4*chi_sigma1)
        chi_max = min(self.default_cosmo.chi(1090), chi_avg2+chi_sigma2*4)
        chi_array = np .array([chi_min + i*(chi_max-chi_min)/Nchi for i in range(Nchi)])
        dchi_basline = np.arange(2*Ndchi)
        D1_D1 = np.zeros((2*Ndchi, Nchi))
        D2_D2 = np.zeros((2*Ndchi, Nchi))

        chi_chi, dchi_dchi_raw = np.meshgrid(chi_array, dchi_basline)
        dchi_dchi_list = []
        for i in range(Nchi):
            dchi_max = min(2*chi_array[i], 2*(self.default_cosmo.chi(1110)-chi_array[i]))
            dchi_array = list(-10**np.array([-1+(np.log10(dchi_max-1)+1)/Ndchi*j for j in range(Ndchi)])[::-1])\
                +list(10**np.array([-1+(np.log10(dchi_max-1)+1)/Ndchi*j for j in range(Ndchi)]))
            dchi_dchi_list.append(dchi_array)
        dchi_dchi = np.transpose(np.array(dchi_dchi_list))
        grid1 = chi_chi-0.5*dchi_dchi
        grid2 = chi_chi+0.5*dchi_dchi
        '''
        for i in range(len(chi_array)):
            for j in range(len(dchi_basline)):
                grid1_ji = grid1[j,i]
                grid2_ji = grid2[j,i]
                D1_D1[j, i] = self.default_cosmo.Dg_norm(grid1_ji)
                D2_D2[j, i] = self.default_cosmo.Dg_norm(grid2_ji)
        '''
        D1_D1 = self.D_class(grid1)
        D2_D2 = self.D_class(grid2)

        Wg1_Wg1 = self.Wg(grid1, chi_avg1, chi_sigma1)
        Wg2_Wg2 = self.Wg(grid2, chi_avg2, chi_sigma2)

        return chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2

    
    def mesh_grid_generator_RSD(self, z1, z2, sigma1, sigma2, Nchi, Ndchi):
        '''
        Params:
        z1, z2: the redshifts of our objects
        sigma1, sigma2: the dispersion of our window function
        Nchi, Ndchi: fidicually chosen length of the sampling array of chi and dchi

        Return:
        mesh-grids of chi1_chi1, chi2_chi2, 
        and, growth factors D1_D1, D2_D2.
        '''
        chi_avg1 = self.default_cosmo.chi(z1)
        chi_avg2 = self.default_cosmo.chi(z2)
        chi_sigma1 = sigma1/self.default_cosmo.HH(z1)
        chi_sigma2 = sigma2/self.default_cosmo.HH(z2)
        chi_min = max(100, chi_avg1-4*chi_sigma1)
        chi_max = min(self.default_cosmo.chi(499), chi_avg2+chi_sigma2*4)
        chi_array = np.array([chi_min + i*(chi_max-chi_min)/Nchi for i in range(Nchi)])
        dchi_basline = np.arange(2*Ndchi)
        D1_D1 = np.zeros((2*Ndchi, Nchi))
        D2_D2 = np.zeros((2*Ndchi, Nchi))

        chi_chi, dchi_dchi_raw = np.meshgrid(chi_array, dchi_basline)
        dchi_dchi_list = []
        for i in range(Nchi):
            dchi_max = min(2*chi_array[i], 2*(self.default_cosmo.chi(499)-chi_array[i]))
            dchi_array = list(-10**np.array([-1+(np.log10(dchi_max-1)+1)/Ndchi*j for j in range(Ndchi)])[::-1])\
                +list(10**np.array([-1+(np.log10(dchi_max-1)+1)/Ndchi*j for j in range(Ndchi)]))
            dchi_dchi_list.append(dchi_array)
        dchi_dchi = np.transpose(np.array(dchi_dchi_list))
        #print(np.shape(chi_chi), np.shape(dchi_dchi))
        grid1 = chi_chi-0.5*dchi_dchi
        grid2 = chi_chi+0.5*dchi_dchi
        '''
        D1_D1 = self.Dg_func(grid1)
        D2_D2 = self.Dg_func(grid2)
        f1_f1 = self.default_cosmo.f_broad(grid1)
        f2_f2 = self.default_cosmo.f_broad(grid2)
        '''
        D1_D1 = self.D_class(grid1)
        D2_D2 = self.D_class(grid2)
        f1_f1 = self.f_class(grid1)
        f2_f2 = self.f_class(grid2)

        Wg1_Wg1 = self.Wg(grid1, chi_avg1, chi_sigma1)
        Wg2_Wg2 = self.Wg(grid2, chi_avg2, chi_sigma2)
        
        return chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, f1_f1, f2_f2

    def mesh_grid_generator_CMBlensing(self, Nchi, Ndchi, z_source):
        '''
        Generating 2D sampling mesh-grid via self-adapting chi and delta_chi
        Params:
        z1, z2: the redshifts of our objects
        sigma1, sigma2: the dispersion of our window function
        Nchi, Ndchi: fidicually chosen length of the sampling array

        Return:
        mesh-grids of chi_chi, dchi_dchi, 
        and, growth factors D1_D1, D2_D2.
        '''
        chi_min = 1.0
        chi_max = self.default_cosmo.chi(z_source)-1.0
        #chi_array = np.e**np.array([np.log(chi_min) + i*(np.log(chi_max)-np.log(chi_min))/(Nchi-1) for i in range(Nchi)])
        chi_array = np.array([chi_min + i*(chi_max-chi_min)/(Nchi-1) for i in range(Nchi)])
        dchi_basline = np.arange(2*Ndchi+2)
        D1_D1 = np.zeros((2*Ndchi+2, Nchi))
        D2_D2 = np.zeros((2*Ndchi+2, Nchi))
        F1_F1 = np.zeros((2*Ndchi+2, Nchi))
        F2_F2 = np.zeros((2*Ndchi+2, Nchi))

        chi_chi, dchi_dchi_raw = np.meshgrid(chi_array, dchi_basline)
        dchi_dchi_list = []
        for i in range(Nchi):
            dchi_max = min(2*chi_array[i], 2*(self.default_cosmo.chi(z_source)-chi_array[i])) #We can change the maximum comoving distance, but note there are actually upper limit in colossus code
            dchi_array = list(-10**np.array([-2+(np.log10(dchi_max-0.5)+2)/Ndchi*j for j in range(Ndchi+1)])[::-1])\
                +list(10**np.array([-2+(np.log10(dchi_max-0.5)+2)/Ndchi*j for j in range(Ndchi+1)]))
            dchi_dchi_list.append(dchi_array)
        dchi_dchi = np.transpose(np.array(dchi_dchi_list))
        print(np.shape(chi_chi), np.shape(dchi_dchi))
        grid1 = chi_chi-0.5*dchi_dchi
        grid2 = chi_chi+0.5*dchi_dchi
        
        for i in range(len(chi_array)):
            for j in range(len(dchi_basline)):
                grid1_ji = grid1[j, i]
                grid2_ji = grid2[j, i]
                #D1_D1[j, i] = self.D_class(grid1_ji)
                #D2_D2[j, i] = self.D_class(grid2_ji)
                F1_F1[j, i] = self.default_cosmo.Psi_normalizer(grid1_ji)
                F2_F2[j, i] = self.default_cosmo.Psi_normalizer(grid2_ji)
        
        D1_D1 = self.D_class(grid1)
        D2_D2 = self.D_class(grid2)
        Wg1_Wg1 = self.Wlensing(grid1, self.default_cosmo.chi(z_source))
        Wg2_Wg2 = self.Wlensing(grid2, self.default_cosmo.chi(z_source))

        return chi_chi, dchi_dchi, D1_D1, D2_D2, Wg1_Wg1, Wg2_Wg2, F1_F1, F2_F2
    
###################################################################################################
 