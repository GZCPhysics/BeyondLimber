#We import the linear matter power spectrum from FullSkyCalculation.py, along with all the needed cosmological parameters and the growth factor
from ast import Pow
import FullSkyCalculation as FSC
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, interp2d
import cmath
import mpmath as mm
import scipy.fft
#import pyfftlog


#We have tested the consistency of the method of using growth factor to probe linear perturbation theory
#Now, we try to define the 3D power spectrum in Equation(34) of Zvonimir's note
cosmo = FSC.cosmo
def Power_spectrum(kh_par, l, chi, delta_chi):
    '''
    This is the cross power spectrum between two different redshifts. i. e. the power spectrum in Equation (34)

    Params:
    k_par: the Fourier modes along the line of sight [h/Mpc]
    l: l/chi is the Fourier modes on the plane perpendicular to the line of sight, in respect of polar coordinate
    chi: the central comoving radial distance between two redshifts, [Mpc/h]
    delta_chi: the difference between comoving radial distances of the samples, [Mpc/h]

    Return:
    An resulted power spectrum, with a unit of (Mpc/h)^3
    '''

    kh = np.sqrt(kh_par**2 + (l/chi)**2)
    D1 = FSC.Dg_norm(chi - 0.5*delta_chi)
    D2 = FSC.Dg_norm(chi + 0.5*delta_chi)
    #We neglect the power spectrum outside our interpolation range
    if (kh >= FSC.kh_new[0]) & (kh <= FSC.kh_new[-1]):
        p_lin = FSC.Plin(kh)
    else:
        p_lin = 0.

    return D1*D2*p_lin

def kh_par_min_max(l, chi):
    '''
    Decide the maximum and minimum k_par range based on the data interpolation range
    '''

    min_square = (FSC.kh_new[0])**2 - (l/chi)**2
    max_square = (FSC.kh_new[-1])**2 - (l/chi)**2
    if min_square <= 0:
        min = 0
    else:
        min = np.sqrt(min_square)
    if max_square <= 0:
        max = 0
    else:
        max = np.sqrt(max_square)
    return min, max

#Unuseful equation
def C_direct_int(l, chi, delta_chi):
    '''
    Here we do the direct calculation of Fourier transformation over the k_par according to Equation(34)
    '''
    kh_par_min,kh_par_max = kh_par_min_max(l, chi)
    N=1000
    lnk_array = np.array([np.log(1e-8)+np.log(kh_par_max/1e-8)/N*i for i in range(N)])
    k_array = np.e**lnk_array
    def integrand(lnk_par):
        k_par = np.exp(lnk_par)
        #exp_part = mm.exp(1j*k_par*delta_chi)
        exp_part = np.cos(k_par*delta_chi)
        power_part = Power_spectrum(k_par, l, chi, delta_chi)
        return k_par*exp_part*power_part/(2*np.pi)
    real_part = quad(lambda lnk_par: integrand(lnk_par).real, np.log(kh_par_min+1e-33), np.log(kh_par_max))[0]
    #imag_part = quad(lambda k_par: integrand(k_par).imag, k_par_min, k_par_max)

    return real_part#, imag_part

def C_fft(l, chi, delta_chi):
    '''
    Here we use the FFT algorithm to calculate the Equation(34), which formally a 1D Fourier transformation
    N here is the number of discrete points that we need.
    '''
    #Creat power spectrum array in respect of k_par
    kh_par_min, kh_par_max = 0,20 #kh_par_min_max(l, chi)
    if delta_chi>=40:
        dkh_crit = 2*np.pi/delta_chi/10
        N_crit = int(kh_par_max/dkh_crit)
        if N_crit>10000:
            dkh = 2*np.pi/delta_chi/2.5
            N = int(kh_par_max/dkh)
        else:
            dkh = dkh_crit
            N = N_crit
    else:
        dkh = kh_par_max/5000
        N = 5000
    #print('Number of Samples:', N)
    #N = int((k_par_max-k_par_min)/dk)
    #k_par_array =np.array(list(-10**np.array([np.log10(k_par_min+1e-8) + (np.log10(k_par_max/1e-8)*2/N)*i for i in range(int(N/2))]))[::-1]+\
                #list(10**np.array([np.log10(k_par_min+1e-8) + (np.log10(k_par_max/1e-8)*2/N)*i for i in range(int(N/2))])))
    #k_par_array = np.array([-k_par_max + k_par_max*2/N*i for i in range(int(N/2))]+[k_par_min + k_par_max*2/N*i for i in range(int(N/2))])
    kh_par_array = [-dkh*N/2 + dkh*i for i in range(N)]
 
    kh_par = scipy.fft.fftshift(kh_par_array)
    power_array = [Power_spectrum(kh, l, chi, delta_chi) for kh in kh_par]
    C_array = scipy.fft.fftshift(scipy.fft.ifft(power_array))
    w_array = scipy.fft.fftshift(scipy.fft.fftfreq(len(power_array))*2*np.pi/dkh)
    C_array *= [N*dkh*cmath.exp(1j*w*kh_par_min).real/(2*np.pi) for w in w_array]

    C_fft_func = interp1d(w_array, np.real(C_array))
    return C_fft_func(delta_chi) #w_array, C_array #C_fft_func(delta_chi) ##,k_par_array, power_array

def C_fftlog(l, chi, delta_chi, N):

    '''
    Try to use FFTlog algorithm to calculate the Fourier transformation function as defined in Equation(34)
    '''
    logkhmin = 1e-8
    kh_par_min, kh_par_max = kh_par_min_max(l, chi)
    logkhmax = np.log10(kh_par_max)
    mu = -0.5 #Consider the Fourier transformation as integral over cos(k \delta\chi)
    q = 0
    chik = delta_chi+10
    ftopt = 1
    tdir = 1
    logkc = (logkhmin+logkhmax)/2
    nc = (N+1)/2
    dlogk = (logkhmax-logkhmin)/N
    dlnr = dlogk*np.log(10.0)
    kpar_array = 10**(logkc+(np.arange(1,N+1)-nc)*dlogk)
    a_k = np.array([Power_spectrum(kpar, l, chi, delta_chi) for kpar in kpar_array])
    ft, xsave = pyfftlog.fhti(N, mu, dlnr, q, chik, ftopt)
    logfc = np.log10(ft)-logkc
    a_delta_chi = pyfftlog.fftl(a_k.copy(), xsave, 1/(2*np.pi), tdir)
    delta_chi_array = 10**(logfc+(np.arange(1,N+1)-nc)*dlogk)

    return delta_chi_array, a_delta_chi


#We use the same FFT method to calculate the Gaussian function to see if we could get the expected analytical results
def Gaussian_fft(l, chi,  N):
    kh_par_min, kh_par_max = kh_par_min_max(l, chi) 
    dkh = 0.001
    kh_par_array = np.array([kh_par_min + dkh*i for i in range(N)])
    
    power_array = np.array([np.exp(-khpar**2) for khpar in kh_par_array])
    C_array = np.fft.fftshift(np.fft.ifft(power_array))
    w_array = np.fft.fftshift(np.fft.fftfreq(power_array.size)*2*np.pi/dkh)
    C_array *= np.array([N*dkh*cmath.exp(1j*w*kh_par_min).real/(2*np.pi) for w in w_array])
    #delta_chi_array = np.fft.rfftfreq(k_par_array)
    #Here, since we do not have information regarding delta_chi, let's hope the frequency will peak at the delta_chi we have chosen
    #We need to plot these in order to check
    return w_array, C_array#, k_par_array, power_array
#The FFT results seem to converge, we currently take it as a reasonable result, and continue to explore the rest of the equation

#We consider our Fourier transform is correct, then we write down the window function for galaxies

#In order to speed up and check a stable interstep of calculation, we first generate the C(l,\chi,\delta\chi) files on the 2D grids
#in respect of \chi and \delta\chi, in the old calculation, we found that the large steps may resulted in error in final results

def C_files(path, l, chi_array, dchi_array):
    ''' 
    The saved file type is binary file, automatically generate a .npy. 
    When reading these files, we use np.load('./C_ell/C_l_%d.npy'%l)
    '''
    Cl_delta_chi = []
    for delta_chi_x in dchi_array:
        Cl_chi = []
        for chi_x in chi_array:
            Cl_chi.append(C_fft(l, chi_x, delta_chi_x))
        Cl_delta_chi.append(Cl_chi)
    np.save(path, Cl_delta_chi)

    return 'Saved file for l=%d'%l

def Wg(chi, chi_avg, chi_sigma):
    '''
    Here we consider a Gaussian window function
    '''
    term1 = 1/(np.sqrt(2*np.pi))/chi_sigma
    term2 = np.exp(-(chi-chi_avg)**2/(2*chi_sigma**2))
    return term1*term2

#Calculation from reading file
def intergral_kernel_read(path, l, chi, delta_chi, chi_array, dchi_array, n,chi_avg1, chi_sigma1,chi_avg2, chi_sigma2):
    '''
    Params:
    l: the angular moment we want to calculate
    chi, delta_chi: the integral variable, will be used in following dblquad() 
    chi_array, dchi_array: the array we used to generate C files
    n: the order of Cl^n, as defined in Equation(33)

    '''
    #Here we have extended our data to negative delta_chi part, which based on the symetric property of our equation
    delta_extend = list(-dchi_array[1:][::-1])+[0]+ list(dchi_array[1:])
    Cl_array = np.load(path)
    Nx = len(Cl_array[0,:])
    Ny = len(Cl_array[:,0])
    Cl_extend = np.zeros(( Ny*2-1, Nx))
    for i in range(len(Cl_extend[0,:])):
        for j in range(len(Cl_array[:,0])):
            Cl_extend[j,i] = Cl_array[::-1][j,i]
            Cl_extend[Ny-1+j,i] = Cl_array[j,i]
    
    #Now define the whole function and perform 2D integral
    #Get C_\ell(l=2, \chi, \delta\chi) from our data to 
    Cl_chi_dchi = interp2d(chi_array, delta_extend, Cl_extend)
    
    Wg1 = Wg(chi-0.5*delta_chi, chi_avg1, chi_sigma1)
    Wg2 = Wg(chi+0.5*delta_chi, chi_avg2, chi_sigma2)
    Cl = Cl_chi_dchi(chi, delta_chi)

    return Wg1*Wg2*Cl*(delta_chi/chi)**n/chi**2

def Cl_n_read(path, n, l, chi_array, dchi_array,chi_avg1, chi_sigma1,chi_avg2, chi_sigma2):

    delta_extend = list(-dchi_array[1:][::-1])+[0]+ list(dchi_array[1:])

    return dblquad(lambda y, x: intergral_kernel_read(path, l, y, x,chi_array, dchi_array, n, chi_avg1, chi_sigma1,chi_avg2, chi_sigma2), delta_extend[0], delta_extend[-1], chi_array[0], chi_array[-1])[0]

#Direct calculation, which is very slow
def C_l_n_kernel(l, chi, delta_chi, chi_avg1, chi_avg2, chi_sigma1, chi_sigma2):

    W1 = Wg(chi+0.5*delta_chi, chi_avg1, chi_sigma1)
    W2 = Wg(chi-0.5*delta_chi, chi_avg2, chi_sigma2)
    C  = C_fft(l, chi, delta_chi, N=10000)
    return W1*W2*C

def C_n_l(l,n, chi_avg1, chi_avg2, chi_sigma1, chi_sigma2, chi_min, chi_max):

    def integrand(chi, delta_chi):
         return (delta_chi**n/chi**(n+2))*C_l_n_kernel(l, chi, delta_chi, chi_avg1, chi_avg2, chi_sigma1, chi_sigma2)

    return dblquad(integrand, 0, (chi_max-chi_min), chi_min, chi_max)[0]

#We can also define the Limber's approximation in order to test

def Kernel_Limber(l, chi, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2):
    
    kh_min, kh_max = kh_par_min_max(l, chi)
    W1 = Wg(chi, chi_avg1, chi_sigma1)
    W2 = Wg(chi, chi_avg2, chi_sigma2)
    C = Power_spectrum(kh_par=0, l=l, chi=chi, delta_chi=0)
    return W1*W2*C
def Cn_Limber(l, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2, chi_min, chi_max):

    def integrand(chi):
        return Kernel_Limber(l,chi, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2)/(chi**2)
    
    return quad(integrand, chi_min, chi_max)[0]



#################################################################################################
#Test cube

if __name__ == "__main__":

    plt.style.use('dark_background')
    import time
    #Note that z1 is always the lower redshift cluster, while z2 is always the higher redshift cluster
    z1 = 1.0
    z2 = 1.25
    sigma1 = 0.05
    sigma2 = 0.05
    chi_avg1 = FSC.chi(z1)
    chi_avg2 = FSC.chi(z2)
    chi_sigma1 = sigma1/FSC.HH(z1)
    chi_sigma2 = sigma2/FSC.HH(z2)
    chi_min = chi_avg1-5*chi_sigma1
    chi_max = chi_avg2+5*chi_sigma2
    print('Comoving distance of middle part of 1st galaxy clusters is:',chi_avg1,'[Mpc/h]')
    print('Comoving distance of middle part of 2nd galaxy clusters is:',chi_avg2,'[Mpc/h]')
    print('Sigma in comoving unit of cluster 1 is:',chi_sigma1)
    print('Sigma in comoving unit of cluster 2 is:',chi_sigma2)

    #So here we fix the chi in our power spectrum function, while vary the delta chi
    N_chi_array = 25
    chi_array = np.array([chi_min + i*(chi_max-chi_min)/N_chi_array for i in range(N_chi_array)])
    
    N_delta_chi_array = 31
    #delta_chi_array = np.array([50/10*i for i in range(10)]+[50+((chi_max-chi_min)/2-50)/10*i for i in range(11)]) #For z1=z2=1
    #delta_chi_array = np.array([50/15*i for i in range(15)]+[50+((chi_max-chi_min)/2-50)/15*i for i in range(16)])
    delta_chi_array = np.array([50/40*i for i in range(40)]+[50+((chi_max-chi_min)/2-50)/20*i for i in range(21)]) #Here we test to bin the delta_chi array finer
    print(delta_chi_array)
    
    l_array_short = np.array([int(2 + i) for i in range(8)] + [int(10 + 10 * i) for i in range(9)] + [int(100+50*i) for i in range(10)])
    l_array_short_new = np.array([2, 5, 10, 20, 40, 80, 100 ,150, 200, 300, 500])

    for li in [200,5,10,20,40,60,80,100,150,300]:
        start = time.time()
        path = './C_ell_1_125/C_l_%d_finer'%li
        C_files(path, li, chi_array, delta_chi_array)
        end = time.time()-start
        print('time for one l is:', end/60, 'min')

    '''
    ##################################################################################################
    #We test for Limber's approximation
    l_limber_array = np.load('./Limber_z1_z2_1.npy')[0,:]
    Cl_limber_array = np.array([Cn_Limber(l, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2, chi_min, chi_max) for l in l_limber_array])
    np.save('./Limber_z1_1_z2_125.npy',[l_limber_array, Cl_limber_array])

    plt.loglog(l_limber_array, np.abs(Cl_limber_array)*1e7)
    plt.xlim(2,500)
    plt.ylim(0.01,10)
    plt.show()
    '''




