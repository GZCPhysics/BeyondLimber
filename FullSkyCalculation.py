
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.special import hyp2f1
from scipy.misc import derivative
import sys, platform, os

from torch import FloatStorage
#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
'''
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
'''
#In this script, we try to replicate the calculation of Assassi's paper using python
#While their original calculation is performed using Mathematica

######################################################################################
#Setup cosmology using CAMB

c = 3e8 #Speed of light, in unit of m/s
zCMB = 1100
zmax = 1500
h=0.6711
omch2 = 0.12029
ombh2 = 0.02207
Omc = omch2/h**2
Omb = ombh2/h**2
Omm = Omb+Omc
ns = 0.97
s8 = 0.83747596996536
######################################################################################
#Growth factor and linear matter power spectrum
from colossus.cosmology import cosmology
params = {'flat': True, 'H0': 67.11, 'Om0': Omm, 'Ob0': Omb, 'sigma8': s8, 'ns': 0.97}
cosmology.addCosmology('myCosmo', params)
cosmo = cosmology.setCosmology('myCosmo')
######################################################################################
#Geometry 

def chi(z):
    '''
    The returned unit is [Mpc/h]
    '''
    return cosmo.comovingDistance(z_min=0.0, z_max=z, transverse=False)

z_array = np.array([0.000001+0.4*i for i in range(30)])
chi_array = chi(z_array)
red_at_chi = interp1d(chi_array, z_array,kind='linear') #The calculation range is from 0~12, which is sufficient in our examples

def func_z(chi_x):
    '''
    Now 
    The unit of chi here is [Mpc/h].
    Therefore, when applying the function of camb, we need to convert it to the unit of [Mpc]
    '''
    
    z = results.redshift_at_comoving_radial_distance(chi/h)
    return z

def a(chi):

    z = red_at_chi(chi)

    return 1/(1+z)

def HH(z):
    '''
    Here the returned value has a unit of [h/Mpc]
    '''
    #return results.hubble_parameter(z)/(c*1e-3*h)
    return cosmo.Hz(z)/(c*1e-3*h)

chi_CMB = chi(zCMB)
chi_max = chi(zmax)

def Dg_z(z):

    return cosmo.growthFactorUnnormalized(z)

def Dg_chi(chi):
    '''
    The unit of chi here is [Mpc/h].
    Therefore, when applying the function of camb, we need to convert it to the unit of [Mpc]
    '''
    #z = results.redshift_at_comoving_radial_distance(chi/h)
    z = red_at_chi(chi)
    return cosmo.growthFactorUnnormalized(z)

def Dg_norm(chi):

    return Dg_chi(chi)/Dg_0

Dg_0 = Dg_z(0.00001)

'''
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.11, ombh2=0.02207, omch2=0.12029, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2.21536e-9, ns=0.97, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)
results = camb.get_results(pars)

pars.set_matter_power(redshifts=[0.], kmax=0.20637e3*h)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)

#Calculated today's matter power spectrum, and achieve sigma8 in our universe.
kh_origin, z_origin, pk_origin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=0.20637e3, npoints = 728)
s8 = np.array(results.get_sigma8())[0]
'''
pkdata = np.loadtxt('./PCAMBz0.txt')
kh = pkdata[:,0]
pk = pkdata[:,1]
kappa_array = [-10.+0.01*i for i in range(int((-np.log(kh[0])/np.log(10.)+0.01)/0.01)+1)]
lowkh = np.array([10**kappa for kappa in kappa_array])
Plowk = np.array([pk[0]*(10**kappa/kh.min())**ns for kappa in kappa_array])
kh_new = np.array(list(lowkh)+list(kh[0:]))
Pk_new = np.array(list(Plowk)+list(pk[0:]))

Plin = interp1d(kh_new, Pk_new)

######################################################################################
#Preliminary Functions:
#Elementary functions and Special Functions:
#Although we could read gamma function from scipy package,
#in order to follow Assassi's work, we also define the GammaC function to speed up calculation
import cmath
def GammaC(z):
    """
    z: a complex number
    """
    q0 = 75122.6331530 + 0.0j
    q1 = 80916.6278952 + 0.0j
    q2 = 36308.2951477 + 0.0j
    q3 = 8687.24529705 + 0.0j
    q4 = 1168.92649479 + 0.0j
    q5 = 83.8676043424 + 0.0j
    q6 = 2.50662827511 + 0.0j
    if z.real >= 0:
        p1 = (q0 + q1*z + q2*z**2 + q3*z**3 + q4*z**4 + q5*z**5 + q6*z**6)/ \
            (z*(z+1)*(z+2)*(z+3)*(z+4)*(z+5)*(z+6))
        return p1*(z+5.5)**(z+0.5)*cmath.exp(-z-5.5)
    else:
        p1 = (q0 + q1*(1-z) + q2*(1-z)**2 + q3*(1-z)**3 + \
            q4*(1-z)**4 + q5*(1-z)**5 + q6*(1-z)**6)/ \
            ((1-z)*(2-z)*(3-z)*(4-z)*(5-z)*(6-z)*(7-z))
        p2 = p1*(1-z+5.5)**(1-z+0.5)*cmath.exp(-1+z-5.5)
        return cmath.pi/(cmath.sin(cmath.pi*z)*p2)

def GammaRatioC(z1, z2):
    """
    Based on GammaC, calcuate the ratio between two gamma function
    """
    q0 = 0.0075122633153 + 0.0j
    q1 = 0.0809166278952 + 0.0j
    q2 = 0.363082951477 + 0.0j
    q3 = 0.868724529705 + 0.0j
    q4 = 1.16892649479 + 0.0j
    q5 = 0.838676043424 + 0.0j
    q6 = 0.250662827511 + 0.0j

    ratio = 1.0 + 0.0j
    for i in range(1,8):
        ratio = ratio*(z2+i-1)/(z1+i-1)
    ratio = ratio*cmath.exp(z2-z1)
    p1 = (z1+0.5)*cmath.log(z1+5.5)-(z2+0.5)*cmath.log(z2+5.5)
    ratio = ratio*cmath.exp(p1)

    z1t = z1/10.0
    z2t = z2/10.0
    p1 = (q0 + q1*z1t + q2*z1t**2 + q3*z1t**3 + q4*z1t**4 + q5*z1t**5 + q6*z1t**6)
    p2 = (q0 + q1*z2t + q2*z2t**2 + q3*z2t**3 + q4*z2t**4 + q5*z2t**5 + q6*z2t**6)

    ratio = ratio*p1/p2

    return ratio

def Hyp2F1basic(a,b,c,z):
    """
    Calculate the basic hypergeometry function.
    Using power series representation around z=0
    a,b,c,z: all complex numbers
    """
    s = 0. + 0.0j
    p = 1.0 + 0.0j
    eps = 1.0
    n = 0.0
    while(eps>1e-10):
        sold = s
        s += p
        p = p*(a+n)*(b+n)*z/(c+n)/(n+1)
        eps = np.abs((s-sold)/s)
        n += 1
    return s

######################################################################################
#Special functions:
#Here we use the special properties possessed by hypergeometric function to speed up integral

def tminC(l, nu):
    """
    l is an integer
    nu is a complex number
    we try to replace the hypergeometric function with scipy.special version this time
    """
    if l > 60:
        term1 = (1-np.exp(-0.08405923801793776*l))*l**(1.0388189966482335)/ \
            (16.552260860083162+l**(1.0388189966482335))
        term2 = (1 - np.exp(-0.03269491513404876*l))*l**(1.0606484271153198)/ \
            (86.60472131391394 + l**1.0606484271153198)

        tmin = term1 + (1/72)*(-term1+term2)*np.abs(nu.imag)
        return tmin
    else:
        term1 = 0.026189266734847335 - 0.04199333649354753*l + 0.01813725076906472*l**2 -\
             0.0019512662766506912*l**3 + 0.00011476285973931163*l**4 - 4.132495310006262e-6*l**5 + \
                 9.321216021016041e-8*l**6 - 1.2840836892275476e-9*l**7 + 9.874351126449866e-12*l**8 -\
                      3.247034458438453e-14*l**9
        term2 = 0.0050534423514964006 - 0.004245361441903382*l + 0.0009644735508629553*l**2 -\
             0.000029194973960888548*l**3 - 1.197268126576586e-7*l**4 + 3.9232441693781885e-8*l**5 -\
                  1.3806236786152843e-9*l**6 + 2.380296810916152e-11*l**7 - 2.105287890873389e-13*l**8 +\
                       7.627228092016026e-16*l**9
        term3 = 0.014502978209351904 - 0.01218174975881159*l + 0.002817970220966578*l**2 -\
             0.00011942831975390713*l**3 + 1.223432213234367e-6*l**4 + 7.921224590247682e-8*l**5 -\
                  3.5781997384264233e-9*l**6 + 6.634225862490053e-11*l**7 - 6.057230587166174e-13*l**8 +\
                       2.230575708513619e-15*l**9

        tmin = term1 + (1/91512)*(-4223*term1 - 961*term2 + 5184*term3)*np.abs(nu.imag) +\
             (1/91512)*(41*term1 + 31*term2 - 72*term3)*np.abs(nu.imag)**2
        return tmin

def I_l(l, nu, t):
    """
    l: integer for polinomial
    nu: a complex number, indicating power expansion index
    t: a real number, indicating r/chi in the integral
    """
    if t > 1:
        z = 1/t 
        factor = t**(-nu)
    else:
        z = t
        factor = 1
    res = 0.0 + 0.0j
    arg = ((1-z**2)/(2*z))**2
    
    if l >= 5:
        z1min = tminC(l,nu)
    else:
        z1min = 0
    
    if z < z1min:
        res = 0.+0.0j
    else:
        if z <= 0.7:
            res = (np.pi**2 * z**l * 2**(nu-1)/GammaC((3-nu)/2))*\
                GammaRatioC(l+nu/2, l+3/2) * Hyp2F1basic((nu-1)/2, l+nu/2, l+3/2, z**2)
        if (z > 0.7) & (z < 1):
            term1 = GammaRatioC(l+nu/2,l+2-nu/2)*GammaRatioC(1-nu/2, 3/2-nu/2)*\
                Hyp2F1basic(l/2+nu/4, nu/4-(l+1)/2, nu/2, -arg)
            term2 = GammaC(nu-2)*cmath.cos(cmath.pi*nu/2) * arg**(1-nu/2) * \
                Hyp2F1basic(l/2-nu/4+1, 1/2-nu/4-l/2, 2-nu/2, -arg)
            res = np.pi*z**(-nu/2)*(np.sqrt(np.pi)*term1 - 2*term2)
        if z == 1:
            res = np.pi**2 * z**l * 2**(nu-1) * GammaC(2-nu) * GammaRatioC(l+nu/2, 2+l-nu/2)/  \
                GammaC((3-nu)/2)**2
    return factor*res

######################################################################################
#Window functions:

#Galaxy Clustering:

#I am not sure about the operator defined in Mathematica, but we just try applying the operator on 
#window functions
def Wg1(chi, chi_av, sigma_chi):
    """
    Here we define a Gaussian window function
    """
    D_chi = Dg_chi(chi)/Dg_0
    coef = 1/(np.sqrt(2*np.pi)*sigma_chi)
    func = np.exp(-(chi-chi_av)**2/(2*sigma_chi**2))
    return coef*func*D_chi

def Wg2(chi, chi_av, sigma_chi):

    D_chi = Dg_chi(chi)/Dg_0
    return Wg1(chi, chi_av, sigma_chi)*D_chi

def DWg1(chi, chi_av, sigma_chi, l):

    second = derivative(lambda x: Wg1(x, chi_av, sigma_chi), chi, n = 2)
    first = derivative(lambda x: Wg1(x, chi_av, sigma_chi), chi, n = 1)
    zero = Wg1(chi, chi_av, sigma_chi)

    return -(second - 2*first/chi - (l*(l+1)-2)*zero/chi**2)

def DDWg1(chi, chi_av, sigma_chi, l):

    second = derivative(lambda x: DWg1(x, chi_av, sigma_chi, l), chi, n = 2)
    first = derivative(lambda x: DWg1(x, chi_av, sigma_chi, l), chi, n = 1)
    zero = DWg1(chi, chi_av, sigma_chi, l)

    return -(second - 2*first/chi - (l*(l+1)-2)*zero/chi**2)

def DDDWg1(chi, chi_av, sigma_chi, l):

    second = derivative(lambda x: DDWg1(x, chi_av, sigma_chi, l), chi, n = 2)
    first = derivative(lambda x: DDWg1(x, chi_av, sigma_chi, l), chi, n = 1)
    zero = DDWg1(chi, chi_av, sigma_chi, l)

    return -(second - 2*first/chi - (l*(l+1)-2)*zero/chi**2)

def DWg2(chi, chi_av, sigma_chi, l):

    second = derivative(lambda x: Wg2(x, chi_av, sigma_chi), chi, n = 2)
    first = derivative(lambda x: Wg2(x, chi_av, sigma_chi), chi, n = 1)
    zero = Wg2(chi, chi_av, sigma_chi)

    return -(second - 2*first/chi - (l*(l+1)-2)*zero/chi**2)

def DDWg2(chi, chi_av, sigma_chi, l):

    second = derivative(lambda x: DWg2(x, chi_av, sigma_chi, l), chi, n = 2)
    first = derivative(lambda x: DWg2(x, chi_av, sigma_chi, l), chi, n = 1)
    zero = DWg2(chi, chi_av, sigma_chi, l)

    return -(second - 2*first/chi - (l*(l+1)-2)*zero/chi**2)

def Wgalaxy(l, chi, chi_av, sigma_chi, nD, nG):
    """
    Window functions for galaxies
    """
    res = 0.
    if nG == 1:
        if nD == 0:
            res = Wg1(chi, chi_av, sigma_chi)
        if nD == 1:
            res = DWg1(chi, chi_av, sigma_chi,l)
        if nD == 2:
            res = DDWg1(chi, chi_av, sigma_chi,l)
        else:
            res = DDDWg1(chi, chi_av, sigma_chi,l)
    else:
        if nD == 0:
            res = Wg2(chi, chi_av, sigma_chi)
        if nD == 1:
            res = DWg2(chi, chi_av, sigma_chi,l)
        else:
            res = DDWg2(chi, chi_av, sigma_chi,l)
    return res

#Integral Eq (3.5):
GL_chi = 50 #The number of points to perform the integral
chi_tab, w_chi_tab = np.polynomial.legendre.leggauss(GL_chi)

def IntWgalaxy(l, nu, chi_av1, sigma_chi1, chi_av2, sigma_chi2, t):
    """
    Params:
    l: multipole
    nu: complex frequency
    chi_av1, chi_av2: centre of the window function W1 and W2
    sigma_chi1, sigma_chi2: width of the window function W1 and W2
    t: chi2/chi1
    Return:
    a value (real number or an array according to the form of t)
    """
    chi_min = chi_av2 - 5*sigma_chi2
    chi_max = chi_av2 + 5*sigma_chi2
    delta_chi = (chi_max-chi_min)/2.
    chi_av = (chi_max+chi_min)/2.
    sum = 0.0 + 0.0j
    for i in range(GL_chi):
        term1 = w_chi_tab[i]*(delta_chi*chi_tab[i]+chi_av)**(1-nu)
        term2 = Wgalaxy(l, delta_chi*chi_tab[i]+chi_av,chi_av2, sigma_chi2, 1, 1)
        term3 = Wgalaxy(l, (delta_chi*chi_tab[i]+chi_av)*t, chi_av1, sigma_chi1, 1, 1)
        term4 = Wgalaxy(l, (delta_chi*chi_tab[i]+chi_av)/t, chi_av1, sigma_chi1, 1, 1)
        sum += term1*term2*(term3 + t**(nu-2)*term4)
    result = delta_chi * sum
    return result

#Gravitational Lensing: We leave Gravitational lensing calculation for now, and only focus on galaxy clustering
######################################################################################
#Fourier Transform:
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
    delta = np.log(kmax/kmin)/(Nmax-1)
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


######################################################################################
#Now we continue to follow the calculation in Mathematica 
#And, calculate the galaxy clustering power spectrum
def delta_Pk(kh):

    return kh**(-1)*Plin(kh)*np.exp(-(kh/10.)**2)

khmin= 1e-8
khmax = 52.
bias = -1.9

cst = delta_Pk(2*khmin)*(2)**(-bias)

Nmax = 200
NN2 = int(Nmax/2)
CC = CoeffTransfer(delta_Pk, bias, cst, Nmax, khmin, khmax)

c_n = CC[:,0]
nu_n = CC[:,1] 
#print("nu_max = ", np.abs(nu_n[0].imag)), after testing, it worked perfectly

def delta_PFT(kh):

    term1 = CC[int(Nmax/2),0]*kh**(CC[int(Nmax/2),1])
    term2_array = [CC[int(Nmax/2+i),0]*kh**(CC[int(Nmax/2+i),1]) for i in range(1, int(Nmax/2)+1)]
    return (term1+2*np.sum(term2_array)).real

######################################################################################
#C^g_l calculate the angular power spectrum accordingly
GLt = 50
ttab, wttab = np.polynomial.legendre.leggauss(GLt)

def Clg_nu(l, nu, chi_av1, sigma_chi1, chi_av2, sigma_chi2):
    """
    Params:
    l: multipole
    nu: complex frequency
    chi_av1, chi_av2: centre of the window function W1 and W2
    sigma_chi1, sigma_chi2: width of the window function W1 and W2
    Return:
    a value (IntWgalaxy*Il integral over t )
    """
    chi1_min = chi_av1 - 5*sigma_chi1
    chi1_max = chi_av1 + 5*sigma_chi1

    chi2_min = chi_av2 - 5*sigma_chi2
    chi2_max = chi_av2 + 5*sigma_chi2

    if l > 10:
        zmin = tminC(l, nu)
        tmin = max(chi1_min/chi2_max, zmin)
        tmax = min(chi1_max/chi2_min, 1)
    else:
        tmin = 0
        tmax = 1
    
    delta_t = (tmax-tmin)/2
    tav = (tmax+tmin)/2

    res = 0.+0.0j
    for it in range(GLt):
        term1 = wttab[it]*IntWgalaxy(l, nu, chi_av1, sigma_chi1, chi_av2, sigma_chi2,\
             delta_t*ttab[it]+tav)
        term2 = I_l(l, nu+1e-10, delta_t*ttab[it]+tav)
        res += term1*term2
    return delta_t*res

def Clg(l, chi_av1, sigma_chi1, chi_av2, sigma_chi2, d):
    """
    Params:
    l: multipole
    chi_av1, chi_av2: centre of the window function W1 and W2
    sigma_chi1, sigma_chi2: width of the window function W1 and W2
    d: represents the # of powers of k multiplying the powerspectrum
    Return:
    a value (final angular power spectrum)
    """
    res = 0. + 0.0j
    for nn in range(NN2):
        res += 2*c_n[nn]*Clg_nu(l, nu_n[nn]+d, chi_av1, sigma_chi1, chi_av2, sigma_chi2)
    res += c_n[NN2]*Clg_nu(l, nu_n[NN2]+d, chi_av1, sigma_chi1, chi_av2, sigma_chi2)

    return res.real/(2*np.pi**2)

######################################################################################
#Block of Testing
'''
import time
import pandas as pd
if __name__ == "__main__":
    sigma1 = 0.05
    z1 = 1.0
    z2 = 1.0
    l1_array1 = np.array([2+i for i in range(98)])
    l1_array2 = np.array([100+10*i for i in range(41)])
    
    chi_1 = chi(z1)
    chi_2 = chi(z2)
    sigma_chi1 = sigma1/HH(z1)
    sigma_chi2 = sigma1/HH(z2)
    print(chi_1)
    print(sigma_chi1)
    #print(IntWgalaxy(5, 2*np.pi*1j+1, chi_1, sigma_chi1, chi_2, sigma_chi2,0.5))
    k_array = []
    Pk_array = []
    with open('/Users/cheng/Documents/Researches_at_Cambridge/Limber/1705 2/Mathematica/z1_z2_1.txt','r') as file:
        for line in file.readlines():
            k_line, Pk_line = line.split(',')
            k_array.append(float(k_line))
            Pk_array.append(float(Pk_line.strip('/n')))
    file.close()
    Pk_Limber_array = pd.read_csv('/Users/cheng/Documents/Researches_at_Cambridge/Limber/1705 2/Mathematica/z1_z2_1_Limber.csv',sep=',').values
    plt.style.use('dark_background')
    plt.loglog(k_array, np.array(Pk_array)*1e6, label='FullSky')
    plt.loglog(Pk_Limber_array[:,0], Pk_Limber_array[:,1]*1e6, '--', label='Limber')
    
    plt.legend()
    plt.xlim(2,500)
    plt.ylim(0.3,10)
    plt.xlabel(r'l')
    plt.ylabel(r'$C_l^{(g)}\rm\, (\times 10^6)$')
    plt.title(r'$z_1 = z_2=1\ and\ \sigma_z=0.05$')
    plt.show()

    
'''
'''
start1 = time.time()
Clg1_array1 = [Clg(li, chi_1, sigma_chi1, chi_2, sigma_chi2, 0) for li in l1_array1]
Clg1_array2 = [Clg(li, chi_1, sigma_chi1, chi_2, sigma_chi2, 0) for li in l1_array2]

l1_array = np.array(list(l1_array1)+list(l1_array2))
Clg1 = np.array(Clg1_array1+Clg1_array2)
end1 = (time.time()-start1)/60

print('Time for computing one set of Clg is:', end1, 'min')

np.savetxt('./CgT1', [l1_array, Clg1])

plt.style.use('dark_background')
plt.loglog(l1_array, Clg1)
plt.xlabel(r'l')
plt.ylabel(r'$C_l^{(g)}\rm\, (\times 10^6)$')
plt.title(r'$z_1 = z_2=1\ and\ sigma_z=0.05$')

plt.show()
'''

######################################################################################




