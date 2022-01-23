#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import legendre
from numpy.polynomial.legendre import Legendre
import matplotlib.ticker as mtick
import itertools


# Legendre polynomial
def leg(n, x): 
    return Legendre(np.concatenate((np.zeros(n), np.array([1]))))(x)

def hbasis(i,x):
#  Evaluates the function Ni at x
    if i==0:
        Ni=0.5 *(1-x)
    elif i==1:
        Ni=0.5 *(1+x)
    else:
        Ni=(np.sqrt(1/(4*(i+1)-6)))*(leg(i,x)-leg(i-2,x))
    return Ni


def stifness_matrix(p):
#evaluates the elemental stifness matrix of size (p+1)x(p+1)
    K=np.zeros((p+1,p+1))
    K[0,0]=K[1,1]=0.5
    K[0,1]=K[1,0]=-0.5
    if p>=1:
        for i in range(2,p+1):
            K[i,i]=1
    return K

def stifness_matrix(p):
#evaluates the elemental stifness matrix of size (p+1)x(p+1)
    K=np.zeros((p+1,p+1))
    K[0,0]=K[1,1]=0.5
    K[0,1]=K[1,0]=-0.5
    if p>=1:
        for i in range(2,p+1):
            K[i,i]=1
    return K

def mass_matrix(p):
# Evaluates the elemental mass matrix of size (p+1)x(p+1)
    G=np.zeros((p+1,p+1))
    G[0,0]=G[1,1]=2/3
    G[0,1]=G[1,0]=1/3
    if p>=2:
        G[0,2]=G[1,2]=G[2,0]=G[2,1]=-1/np.sqrt(6)
        for i in range(2,p+1):
            G[i,i]=2/((2*(i+1)-1)*((2*(i+1)-5)))
    if p>=3:
        G[0,3]=G[3,0]=1/3*np.sqrt(10)
        G[1,3]=G[3,1]=-1/3*np.sqrt(10)
        for i in range(2,p+1):
            if i+2<p+1:
                G[i,i+2]=G[i+2,i]=(-1)/(((2*(i+1)-1)*np.sqrt(((2*(i+1)-3)* \
                ((2*(i+1)+1))))))
    return G

def load_vector(x_k,x_kk,p_k,f):
#Evaluates the elemental load vector
    vals=[]
    for i in range(p_k+1):
        g = lambda t : f((1-t)*x_k/2+(1+t)*x_kk/2)*hbasis(i,t)
        vals.append(integrate.quad(g,-1,1)[0])
    b=np.array(vals)
    return b

def el_stiff(x_k,x_kk,p_k):
    h_k=x_kk-x_k
    Kk=(2/h_k)*stifness_matrix(p_k)
    return Kk

def el_mass(x_k,x_kk,p_k):
    h_k=x_kk-x_k
    Gk=(h_k/2)*mass_matrix(p_k)
    return Gk

def pointer(M,p):
# [P] = pointer(M,p)
# 
# Calculates the pointer matrix P, such that
# P(i,j)=k means that over the ith element
# the jth local basis function corresponds
# to the kth global basis function.
#
# M is the number of elements
# p is the degree vector of size M
# P is M by (max(p)+1)
# 
    pmax=max(p)
    P=np.zeros((M,pmax+1))
    P=P.astype(int)
    for i in range(M):
        P[i,0] = i
        P[i,1] = i+1    
    P[M-1,1]=0
    for i in range(M):
        for j in range(2,1+p[i]):
            P[i,j] = M
            M=M+1
    return P

def global_matrix(x,p):
    #Evaluates the elemental matrices, stiffnes(Kk) and mass(Gk) and then 
    #the global matrix (G+K)
    P=pointer(len(x)-1,p)
    K=np.zeros((sum(p)-1,sum(p)-1))
    G=np.zeros((sum(p)-1,sum(p)-1))
    for k in range(len(x)-1):
        Kk=np.zeros((p[k]+1,p[k]+1))
        Gk=np.zeros((p[k]+1,p[k]+1))
        Kk=el_stiff(x[k],x[k+1],p[k])
        Gk=el_mass(x[k],x[k+1],p[k])     
        for i in range(p[k]+1):
            l=P[k,i]
            for j in range(p[k]+1):
                m=P[k,j]
                if l!=0 and m!=0:
                    K[l-1,m-1]=K[l-1,m-1]+Kk[i,j]
                    G[l-1,m-1]=G[l-1,m-1]+Gk[i,j]
    GL=K+G
    return GL

def el_load(x_k,x_kk,p_k,f):
#evaluates the elemental load vector 
    h_k=x_kk-x_k
    Fk=(h_k/2)*load_vector(x_k,x_kk,p_k,f)
    return Fk

def global_load_vector(x,p,f):
#Evaluates the global load vector  
    P=pointer(len(x)-1,p)
    F=np.zeros((sum(p)-1))
    for k in range(len(x)-1):
        Fk=el_load(x[k],x[k+1],p[k],f)
        for i in range(p[k]+1):
            l=P[k,i]
            if l!=0:
                F[l-1]=F[l-1]+Fk[i]
    return F
            
def fem_solution(x,p,f):
# Evaluates the finite element method solution and the coifficients that 
#generate the solution
    A=global_matrix(x,p)
    b=global_load_vector(x,p,f)
    a=np.linalg.solve(A,b)
    return a

def energy_norm(x,p,f,n):
# Evaluates the energy norm of the finite element method solution and it finds 
# the percentage of the error
    a=fem_solution(x,p,f)
    DOF=len(a)
    enorm = np.dot(a, np.array(global_load_vector(x,p,f)))
    y = lambda z : (z**n-z)*f(z)
    enorm_uex=integrate.quad(y,0,1)[0]
    relE=100*np.sqrt(abs(enorm-enorm_uex)/abs(enorm_uex))
    return enorm,relE,DOF

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a,b)

def solid1d(y,x,M,p,f):
#Evaluates the finite element method solution for every y that belongs to 
#[a,b] interval
    x_intervals = list(pairwise(x))
    k = [0]
    for point in y[1:]:
        for i, interval in enumerate(x_intervals[k[-1]:]):
            if point>=interval[0] and point<interval[1]:
                k.append(k[-1]+i)
                break
    k.append(len(x_intervals)-1)
    ksi=[]
    for k_val, y_i in zip(k,y):
        ksi.append((2*y_i-x_intervals[k_val][0]-x_intervals[k_val][1])/ \
                   (x_intervals[k_val][1]-x_intervals[k_val][0]))
    P=pointer(M,p)
    ufe=[]
    c=fem_solution(x,p,f)
    for k_val, ksi_i in zip(k,ksi):
        proxy=[]
        for i in range(1,p[k_val]+2):
            l=P[k_val,i-1]
            if l!=0:
                proxy.append(c[l-1]*hbasis(i-1,ksi_i))
            else:
                proxy.append(0)
        ufe.append(sum(proxy))
    return ufe
        

def h_unif(M,pmax,a,b):
#It creates a list of the uniform mesh and a list with the degrees of the 
#polynomials which are constant.
    x=[]
    p=[]
    h=(b-a)/M
    for i in range(M+1):
        z=a+i*h
        x.append(z)
    for j in range(M):
        p.append(pmax)  
    return x,p
  
                              
def h_radical(M,pmax,s,a,b):
#It creates a list of the root-s mesh and a list with the degrees of the 
#polynomials which are constant.
    p=[]
    x=[]
    for i in range(M+1):
        z=a+(b-a)*((i/M)**s)
        x.append(z)
    for j in range(M):
        p.append(pmax)
    return x,p

def p_unif(M,pmax,a,b):
#It creates a list of the uniform mesh which is constant and a list with 
#the degrees of the polynomials that goes 1:pmax.
    p=[]
    x=[]
    h=(b-a)/M
    for i in range(M+1):
        z=a+i*h
        x.append(z)
    for j in range(1,pmax+1):
        p.append(j)
    return x,p

def p_geom(M,pmax,q,a,b):
#It creates a list of the geometric-q mesh which is constant and a list with
# the degrees of the polynomials that goes 1:pmax.
    p=[]
    x=[a]
    for i in range(2,M+2):
        z=a+(b-a)*(q**(M-i+1))
        x.append(z)
    for j in range(1,pmax+1):
        p.append(j)
    return x,p
                     
def hp_geom(M,q,a,b):
#It creates a list of the geometric-q mesh and a list with the degrees of the
# polynomials that goes 1:M, M:=number of elements.
    x=[a]
    p=[]
    pmax=M
    for i in range(2,M+2):
        z=a+(b-a)*(q**(M-i+1))
        x.append(z)
    for j in range(1,pmax+1):
        p.append(j)
    return x,p   
                     


# In[2]:


#f(x)=x**n-x
#n=7.1,a=0,b=1
#Graphs the error against DOF  in logaritmic axes
#First it uses h fem with uniform mesh for polynomial degrees 1 and then 2
a=0
b=1
n=7.1
N=5
f = lambda x : (x**n)-x-n*(n-1)*(x**(n-2))
fig, axes = plt.subplots(1,4, figsize=(15,7))
fig.tight_layout()
fig.subplots_adjust(wspace=0.3)
for j in range(1,3):
    pmax=j
    errors=[]
    DOF=[]
    for i in range(1,N+1):
        M=2**i
        x=h_unif(M,pmax,a,b)[0]
        p=h_unif(M,pmax,a,b)[1]
        errors.append(energy_norm(x,p,f,n)[1])
        DOF.append(energy_norm(x,p,f,n)[2])
    axes[j-1].loglog(DOF,errors)
    axes[j-1].set_xlabel("Degrees of freedom")
    axes[j-1].set_ylabel("Error in energy norm(%)")
    axes[j-1].set_title(f"Error for l=7.1, h_unif with N=5,p={j}")
    print(f"the slope is:",(np.log(errors[-1])-np.log(errors[-2]))/ \
          (np.log(DOF[-1])-np.log(DOF[-2])))

#Second it uses p fem with uniform mesh with 1 element and then two elements
#for polynomial degrees 1,....,6   
P_uni_list=[1,4]
for j,Mp in enumerate(P_uni_list):
    pmax=6
    errors=[]
    DOF=[]
    x=p_unif(Mp,pmax,a,b)[0]
    p=p_unif(Mp,pmax,a,b)[1]
    for i in p:
        p_list=[]
        for k in range(Mp):
            p_list.append(i)
        errors.append(energy_norm(x,p_list,f,n)[1])
        DOF.append(energy_norm(x,p_list,f,n)[2])
    axes[j+2].loglog(DOF,errors)
    axes[j+2].set_xlabel("Degrees of freedom")
    axes[j+2].set_ylabel("Error in energy norm(%)")
    axes[j+2].set_title(f"Error for l=7.1,p_unif with M={Mp},pmax=6")
    print(f"the slope is:",(np.log(errors[-4])-np.log(errors[-5]))/ \
          (np.log(DOF[-4])-np.log(DOF[-5])))


# In[3]:


#f(x)=x**n-x
#n=2.1,a=0,b=1
#Graphs the error against DOF  in logaritmic axes
#First it uses h fem with uniform mesh for polynomial degrees 1 and then 2
a=0
b=1
n=2.1
N=5
f = lambda x : (x**n)-x-n*(n-1)*(x**(n-2))
fig, axes2 = plt.subplots(1,3, figsize=(15,7))
fig.tight_layout()
fig.subplots_adjust(wspace=0.3)
for j in range(1,3):
    pmax=j
    errors=[]
    DOF=[]
    for i in range(1,N+1):
        M=2**i
        x=h_unif(M,pmax,a,b)[0]
        p=h_unif(M,pmax,a,b)[1]
        errors.append(energy_norm(x,p,f,n)[1])
        DOF.append(energy_norm(x,p,f,n)[2])
    axes2[j-1].loglog(DOF,errors)
    axes2[j-1].set_xlabel("Degrees if freedom")
    axes2[j-1].set_ylabel("Error in energy norm(%)")
    axes2[j-1].set_title(f"Error for l=2.1, h_unif with N=5,p={j}")
    print(f"the slope is:",(np.log(errors[-1])-np.log(errors[-2]))/ \
          (np.log(DOF[-1])-np.log(DOF[-2])))
#Second it uses p fem with uniform mesh with 1 element and then two elements 
#for polynomial degrees 1,....,6       
M=1
pmax=6
errors=[]
DOF=[]
x=p_unif(M,pmax,a,b)[0]
p=p_unif(M,pmax,a,b)[1]
for i in p:
    p_list=[]
    for k in range(M):
        p_list.append(i)
    errors.append(energy_norm(x,p_list,f,n)[1])
    DOF.append(energy_norm(x,p_list,f,n)[2])
axes2[2].loglog(DOF,errors)
axes2[2].set_xlabel("Degrees of freedom")
axes2[2].set_ylabel("Error in energy norm(%)")
axes2[2].set_title(f"Error for l=2.1,p_unif with M={M},pmax=6")
print(f"the slope is:",(np.log(errors[-1])-np.log(errors[-2]))/ \
      (np.log(DOF[-1])-np.log(DOF[-2])))
    


# In[4]:


#f(x)=x**n-x
#n=2.1,a=0,b=1
#Graphs the error against DOF  in logaritmic axes
a=0
b=1
n=2.1
f = lambda x : (x**n)-x-n*(n-1)*(x**(n-2))
fig, axes3 = plt.subplots(1,4, figsize=(15,7))
fig.tight_layout() 
fig.subplots_adjust(wspace=0.3)
j=0

#It uses h radical fem for N=5, s=0.15 for polynomial degrees 2
N=5
s=0.15
pmax=2
errors=[]
DOF=[]
for i in range(1,N+1):
    M=2**i
    x=h_radical(M,pmax,s,a,b)[0]
    p=h_radical(M,pmax,s,a,b)[1]
    errors.append(energy_norm(x,p,f,n)[1])
    DOF.append(energy_norm(x,p,f,n)[2])
axes3[j].loglog(DOF,errors)
axes3[j].set_xlabel("Degrees of freedom")
axes3[j].set_ylabel("Error in energy norm(%)")
axes3[j].set_title(f"Error for l=2.1,h-radical with N=5,s=0.15,p=2")
print(f"the slope is:",(np.log(errors[-1])-np.log(errors[-2]))/ \
      (np.log(DOF[-1])-np.log(DOF[-2])))
#It uses p fem with geometric mesh for M=4, q=0.15 for polynomial degrees 
#p=1,....,6
M=4
q=0.15
pmax=6
errors=[]
DOF=[]
x=p_geom(M,pmax,q,a,b)[0]
p=p_geom(M,pmax,q,a,b)[1]
for i in p:
    p_list=[]
    for k in range(M):
        p_list.append(i)
    errors.append(energy_norm(x,p_list,f,n)[1])
    DOF.append(energy_norm(x,p_list,f,n)[2])
axes3[j+1].loglog(DOF,errors) 
axes3[j+1].set_xlabel("Degrees of freedom")
axes3[j+1].set_ylabel("Error in energy norm(%)")
axes3[j+1].set_title(f"Error for l=2.1,p-geom with M=4,q=0.15,p=6")
print(f"the slope is:",(np.log(errors[-5])-np.log(errors[-6]))/ \
      (np.log(DOF[-5])-np.log(DOF[-6])))
#It uses p fem with geometric mesh for M=4, q=0.15 for polynomial degrees 1,2
M=4
q=0.15
pmax=2
errors=[]
DOF=[]
x=p_geom(M,pmax,q,a,b)[0]
p=p_geom(M,pmax,q,a,b)[1]
for i in p:
    p_list=[]
    for k in range(M):
        p_list.append(i)
    errors.append(energy_norm(x,p_list,f,n)[1])
    DOF.append(energy_norm(x,p_list,f,n)[2])
axes3[j+2].loglog(DOF,errors) 
axes3[j+2].set_xlabel("Degrees of freedom")
axes3[j+2].set_ylabel("Error in energy norm(%)")
axes3[j+2].set_title(f"Error for l=2.1,p-geom with M=4,q=0.15,p=2")
print(f"the slope is:",(np.log(errors[-1])-np.log(errors[-2]))/ \
      (np.log(DOF[-1])-np.log(DOF[-2])))

#It uses hp fem with geometric mesh for N=5, q=0.15 for polynomial degrees p=M
N=5
q=0.15
errors=[]
DOF=[]
for i in range(1,N+1):
    M=2**i
    x=hp_geom(M,q,a,b)[0]
    p=hp_geom(M,q,a,b)[1] 
    for z in p:
        plist=[]
        for k in range(M):
            plist.append(z) 
        errors.append(energy_norm(x,plist,f,n)[1])
        DOF.append(energy_norm(x,plist,f,n)[2])
axes3[j+3].semilogy(DOF,errors) 
axes3[j+3].set_xlabel("Degrees of freedom")
axes3[j+3].set_ylabel("Error in energy norm(%)")
axes3[j+3].set_title(f"Error for l=2.1,hp-geom with N=6,q=0.15")
print(f"the slope is:",(np.log(errors[-4])-np.log(errors[-5]))/ \
      (np.log(DOF[-4])-np.log(DOF[-5])))


# In[5]:


import warnings
warnings.filterwarnings('ignore')

#n=2.1
#a=0,b=1
#Graphs the absolute error between the finite element solution(uFE)and the 
#exact solution (uEX)

a = 0
b = 1
n = 2.1
M = 10
q = 0.15
f = lambda x : (x**n)-x-n*(n-1)*(x**(n-2))
y = np.linspace(0,1)
u_Ex_fun = lambda x : (x-x**n)

#First;y I use the h fem with uniform mesh for M=10,p=1
x=h_unif(M,1,a,b)[0]
p=h_unif(M,1,a,b)[1]
u_Fe = -np.array(solid1d(y,x,M,p,f))
u_Ex_values = np.array([u_Ex_fun(val) for val in y])
fig , axes4 =plt.subplots(figsize=(10,10))
axes4.plot(y,abs(u_Ex_values-u_Fe)/abs(u_Ex_values),color='black')
axes4.set_xlabel("x")
axes4.set_ylabel("y")
axes4.set_title(r'Error ${|u_{FE}-u_{EX}|/|u_{EX}|}$')
#Graphs the uFE against uEX
fig , axes5 =plt.subplots(figsize=(10,10))
axes5.plot(y,u_Fe, color="red")
axes5.plot(y,u_Ex_values, color="blue")
axes5.set_xlabel("x")
axes5.set_ylabel("y")
axes5.set_title(r'Plots of $u_{FE}$ and $u_{EX}$')
axes5.legend([r'$u_{FE}$','$u_{EX}$'])

#Secondly I use the h fem with uniform mesh for M=1,p=6
x=h_unif(1,6,a,b)[0]
p=h_unif(1,6,a,b)[1]
u_Fe = -np.array(solid1d(y,x,1,p,f))
u_Ex_values = np.array([u_Ex_fun(val) for val in y])
fig , axes6 =plt.subplots(figsize=(10,10))
axes6.plot(y,abs(u_Ex_values-u_Fe)/abs(u_Ex_values),color='black')
axes6.set_xlabel("x")
axes6.set_ylabel("y")
axes6.set_title(r'Error ${|u_{FE}-u_{EX}|/|u_{EX}|}$')
#Graphs the uFE against uEX
fig , axes7 =plt.subplots(figsize=(10,10))
axes7.plot(y,u_Fe, color="red")
axes7.plot(y,u_Ex_values, color="blue")
axes7.set_xlabel("x")
axes7.set_ylabel("y")
axes7.set_title(r'Plots of $u_{FE}$ and $u_{EX}$')
axes7.legend([r'$u_{FE}$','$u_{EX}$'])

#Thirdly I use the p fem with geometric mesh for M=10,p=2 for all elements
x=p_geom(M,2,q,a,b)[0]
p=[]
for r in range(M):
    p.append(2)
u_Fe = -np.array(solid1d(y,x,M,p,f))
u_Ex_values = np.array([u_Ex_fun(val) for val in y])
fig , axes8 =plt.subplots(figsize=(10,10))
axes8.plot(y,abs(u_Ex_values-u_Fe)/abs(u_Ex_values),color='black')
axes8.set_xlabel("x")
axes8.set_ylabel("y")
axes8.set_title(r'Error ${|u_{FE}-u_{EX}|/|u_{EX}|}$')
#Graphs the uFE against uEX
fig , axes9 =plt.subplots(figsize=(10,10))
axes9.plot(y,u_Fe, color="red")
axes9.plot(y,u_Ex_values, color="blue")
axes9.set_xlabel("x")
axes9.set_ylabel("y")
axes9.set_title(r'Plots of $u_{FE}$ and $u_{EX}$')
axes9.legend([r'$u_{FE}$','$u_{EX}$'])

#Fourthly I use the p fem with geometric mesh for M=10,p=4 for all elements
x=p_geom(M,4,q,a,b)[0]
p=[]
for r in range(M):
    p.append(4)
u_Fe = -np.array(solid1d(y,x,M,p,f))
u_Ex_values = np.array([u_Ex_fun(val) for val in y])
fig , axes10 =plt.subplots(figsize=(10,10))
axes10.plot(y,abs(u_Ex_values-u_Fe)/abs(u_Ex_values),color='black')
axes10.set_xlabel("x")
axes10.set_ylabel("y")
axes10.set_title(r'Error ${|u_{FE}-u_{EX}|/|u_{EX}|}$')
#Graphs the uFE against uEX
fig , axes11 =plt.subplots(figsize=(10,10))
axes11.plot(y,u_Fe, color="red")
axes11.plot(y,u_Ex_values, color="blue")
axes11.set_xlabel("x")
axes11.set_ylabel("y")
axes11.set_title(r'Plots of $u_{FE}$ and $u_{EX}$')
axes11.legend([r'$u_{FE}$','$u_{EX}$'])

