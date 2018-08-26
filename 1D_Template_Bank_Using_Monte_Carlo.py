
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt # For ploting
import numpy as np # To work with numerical data efficiently
import scipy as scp
from scipy import integrate  
from scipy.misc import derivative
from scipy import random
import sympy as sp
import time


# In[3]:


# We Define most of our constants in this section
c = 299792458.0 #The Speed of Light in m/s
G = 6.67408e-11 # The Gravitional Constant in m3/(kg s^2)
M_sun = 1.9884754153381438e+30 # The mass of the sun in kg
f_0 = float(10) # An initial frequency of 10 Hz
omega_0 = 2*np.pi*f_0 # The inital angular frequency in rad/s

M_min = 1*M_sun # Minimum total mass for a binary neutron star system
M_max = 3*M_sun # Maximum total mass for a binary neutron star system
print(M_min)
print(M_max)


# In[4]:


MM = float(input("What is the minimal match? "))


# In[5]:


# We first need to built templates along the equal mass curve
nu = 1/4 # symmetric mass


# In[6]:


# We define the Newtonian chirp time as a function of the total mass of the system
A_0 = (5/256)*(np.pi*f_0)**(-8/3)
A_1 = c**5
A_2 = G**(-5/3)
def tau_0(M):
    return A_1*A_2*(A_0/nu)*(M**(-5/3))

def Mass(tau_0):
    return (A_1*A_2*(A_0/nu)*(1/tau_0))**(3/5)

print(tau_0(M_min))
print(tau_0(M_max))

print(Mass(tau_0(M_min)))


# In[7]:


# The frequency at which the two objects merge in Hz
# known as the ISCO gravitational-wave frequency
# ISCO stands for innermost stable circular orbit

c_3 = c**3/np.pi*G*6**(3/2)

def f_LSO(M):
    return c_3*M**-1


# In[8]:


# Since our template bank is one dimensional
D = 1 # D is the dimension of our parameter space

# Proper Volume Per Template for a hypercubic lattice
DeltaV = (2*np.sqrt((1-MM)/D))**D

print(DeltaV)


# In[9]:


def g(x):
    return (5*x**(-7/3))/(x**-4 + 2*(1+x**2))
I_7,err = integrate.quad(g,4,np.inf)

print(I_7)


# In[10]:


def psi_0(x):
    return 2*np.pi*f_0*x
                          
def psi_1(x):
    return 2*np.pi*f_0*(3/5)*(x**(-5/3))


# In[52]:


# Metric Calculation Method: Monte Carlo Integration
# Monte Carlo integration is better for higher dimensions
# However, I did it here because it's what was used in the
# research papers.

start = time.time()

f_L = 20 # Hz
f_U = 30 # Hz

L = f_L/f_0
U = f_U/f_0 # limits of integration

# Sampling the signal with a sampling rate of 2*f_LSO Hz
# In this case, we are using the Nyquist sampling rate.
Fs = 2.5*f_U # Engineering Nyquist Sampling rate frequency
F_Nyquist = Fs/2 # Nyquist frequency
delta_t = 1/Fs # Sampling inverval/rate or time between data points

T = 40 # Sampling space or total time interval in seconds
# Note that delta_t = T/n

n = int(round(T/delta_t)) # n is the number of sample points
xrand = random.uniform(L,U,n) # We generate some randome numbers

def h_0(x):
    return g(x)*psi_0(x)
def h_00(x):
    return g(x)*psi_0(x)**2 #psi_0(x)
def h_1(x):
    return g(x)*psi_1(x)
def h_01(x):
    return g(x)*psi_0(x)*psi_1(x)
def h_11(x):
    return g(x)*psi_1(x)*psi_1(x)


# We use Monte Carlo integration to find 
# the moment functionals

integral = 1.0

for i in range(n):
    integral += h_0(xrand[i])

    J_0 = (U - L)/float(n)*integral/I_7

integral = 1.0

for i in range(n):
    integral += h_00(xrand[i])

    J_00 = (U - L)/float(n)*integral/I_7

integral = 1.0
    
for i in range(n):
    integral += h_1(xrand[i])

    J_1 = (U - L)/float(n)*integral/I_7

integral = 1.0

for i in range(n):
    integral += h_01(xrand[i])

    J_01 = (U - L)/float(n)*integral/I_7
    
integral = 1.0
    
for i in range(n):
    integral += h_11(xrand[i])

    J_11 = (U - L)/float(n)*integral/I_7
    

# We can now calculate the gammas which
# we will use to calculate the metric

gamma_00 = (1/2)*(J_00 - J_0*J_0)
gamma_01 = (1/2)*(J_01 - J_0*J_1)
gamma_11 = (1/2)*(J_11 - J_1*J_1)

# In general the metric is defined as
# g_ij = gamma_ij - gamma_0i*gamma_0j/gamma_00

# For our one dimesional space our metric only has one component
g11 = gamma_11 - gamma_01*gamma_01/gamma_00

end = time.time()
print(end - start) # Time it takes to run the program


# In[55]:


# Now since our metric is a scalar
# it's equal to it's determinant
# therefore we can just use g11

V = np.sqrt(g11)*(tau_0(M_min) - tau_0(M_max))

# N is the Number of Templates required to cover
# the desired range of parameters

N = (1/DeltaV)*V
print(N)

dl = np.sqrt(2*((1-MM)/g11))


# In[56]:


# Template Bank Placement Algorithm

# We lay templates assuming for points in which nu = 1/4

y = tau_0(M_max) # Our starting point
f = open("InspiralTemplateList.txt", "w") # Open new data file
while y <= tau_0(M_min):
    f.write( str(y) + "\n"  )
    y = y + dl
else:
    f.write( str(tau_0(M_min)))
    f.close()
    
# Now we print out the contents of the file
print(open('InspiralTemplateList.txt').read())


# In[ ]:




