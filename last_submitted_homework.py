# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 20:58:50 2021

@author: Panayiota Damianou
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy import integrate

# Legendre polynomial
def leg(n, x): 
    if(n == 0):
        return 1 # P0 = 1
    elif(n == 1):
        return x # P1 = x
    else:
        return (((2 * n)-1)*x * leg(n-1, x)-(n-1)*leg(n-2, x))/float(n)
  


def hbasis(i,x):
#  Evaluates the function Ni(y) at x
#  In function L(y) i-2 is the degree of the legendre polynomial

    if i==1:
        Ni=0.5 *(1-x)
    elif i==2:
        Ni=0.5 *(1+x);
    else:
        Ni=(np.sqrt(1/(4*i-6)))*(leg(i-1,x)-leg(i-3,x))
    return Ni

def dhbasis(i,x):   
#   Evaluates the derrivative of the function Ni(x) at x
#   In function L(y) i-2 is the degree of the legendre polynomial

    if i==1:
        dNi=-0.5
    elif i==2:
        dNi=0.5
    else:
        dNi=leg(i-2,x)*np.sqrt((2*i-3)/2)
    return dNi

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
    if p>=1:
        G[0,2]=G[1,2]=G[2,0]=G[2,1]=-1/np.sqrt(6)
    for i in range(2,p+1):
        G[i,i]=2/(((2*i)-1)*((2*i)-5))
    if p>=3:
        G[0,3]=G[3,0]=1/3*np.sqrt(10)
        G[1,3]=G[3,1]=-1/3*np.sqrt(10)
        for i in range(2,p+1):
            if i+2<=p+1:
               G[i,i+2]=G[i+2,i]=(-1)/(((2*i)-1)*np.sqrt(((2*i)-3)*((2*i)+1)))
    return G

def load_vector(p,f,l,c):
#Evaluates the element load vector
    b=np.zeros((1,p+1))
    for i in range(p+2):
        g = lambda x : f(x,l,c) * hbasis(i,x)
        b[i]= integrate.quad(g,-1,1)
    return b

def fem_solution(p,f,x,l,c):
# Evaluates the finite element method solution and the coifficients that generate the solution
    A=stifness_matrix(p)+mass_matrix(p)
    b=load_vector(p,f,l,c)
    a=A.solve(b)
    ufe_list=[a[i]*[hbasis(i,point) for point in x] for i in range(1,p+2)]
    ufe=sum([sum(l) for l in ufe_list])
    return a, ufe

def energy_norm(p,f,x,l,c):
# Evaluates the energy norm of the finite element method solution and it finds 
# the percentage of the error
    a=fem_solution(p,f,x,l,c)
    enorm= np.dot(a, load_vector(p,f,l,c))
    err=1
    return enorm,err

# IMPORTANT NOTE (IF ERROR): [f(x) for x in x_list]
    
def main():
    f = lambda x,l,c : (((1+x)**(l-2))*2*l*(1+x))-((1-x)*l*(l-1))+(c*(1-x)*(1+x)**2)
    l=[8.7,4.4,2.9,1.2]
    for k in l:
        c=1
        p=[2,3,4,5,6,7,8]
        error=[]
        x = np.linspace(-1,1,1001)
        for j in p:
            error.append(energy_norm(j,f,x,k,c)[1])
        fig, axes = plt.subplots(1,3, figsize=(12,3))
        
        axes[k].loglog(x,error) 
  
        axes[k].xlabel("error")
        axes[k].ylabel("N")

    l=[1.1]
    c=2
    p=4
    x = np.linspace(-1,1,1001)
    u_Ex = lambda x : (1-x)*((1+x)**l)
    u_Fe = fem_solution(p,f,x)[1]
    fig , axes =plt.subplots()
    axes[0].plot(x,u_Fe, color="red")
    axes[0].plot(x,u_Ex, color="blue")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend("Γραφική Παραστάση των u_(FE) και u_(EX)")
    
    axes[1].plot(x,abs(u_Ex-u_Fe),color='black')
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend("Γραφική Παραστάση του σφάλματος |u_(FE)-u_(EX)|")
    
if __name__ == "__main__":
    main()    
    
    

        
    
