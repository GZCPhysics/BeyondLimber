import numpy as np
from matplotlib import pyplot as plt
import FullSkyCalculation as FSC
import OurApproximation as OA

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

#So here we fix the chi in our power spectrum function, while vary the delta chi
N_chi_array = 25
chi_array = np.array([chi_min + i*(chi_max-chi_min)/N_chi_array for i in range(N_chi_array)])
    
N_delta_chi_array = 21
#delta_chi_array = np.array([50/10*i for i in range(10)]+[50+(5*chi_sigma1-50)/10*i for i in range(11)])
#delta_chi_array = np.array([50/15*i for i in range(15)]+[50+((chi_max-chi_min)/2-50)/15*i for i in range(16)])
delta_chi_array = np.array([50/40*i for i in range(40)]+[50+((chi_max-chi_min)/2-50)/20*i for i in range(21)])

Cl_list=[]
Cl_1_list=[]
for li in [2,5,10,20,40,60,80,100,150,200,300]:
    path = './C_ell_1_125/C_l_%d_finer.npy'%li
    Cl_list.append(OA.Cl_n_read(path, 0, li, chi_array, delta_chi_array, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2))
    Cl_1_list.append(OA.Cl_n_read(path, 1, li, chi_array, delta_chi_array, chi_avg1, chi_sigma1, chi_avg2, chi_sigma2))

l_limber_array = np.load('./Limber_z1_1_z2_125.npy')[0,:]
Cl_limber_array = np.load('./Limber_z1_1_z2_125.npy')[1,:]

print(Cl_list)
print(Cl_1_list)

plt.style.use('dark_background')
plt.loglog(l_limber_array, np.abs(Cl_limber_array)*1e7)
plt.loglog([2,5,10,20,40,60,80,100,150,200,300], np.abs(np.array(Cl_list)+np.array(Cl_1_list)/2)*1e7, '*')
plt.xlim(2,500)
plt.ylim(0.01,10)
plt.show()