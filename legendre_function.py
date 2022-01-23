# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:41:19 2021

@author: Panayiota Damianou
"""

def leg(n,x):
#[P] computes the legendre polynomial P_n(x)
    if n==0:
        #pk=1
    elif n==1:
        pk=np.transpose([1,0])
    else:
        pkm2=np.zeros((n+1,1))
        pkm2[n+1]=1
        pkm1=np.zeros((n+1,1))
        pkm1[n]=1
        for k in range(2,n+1):
            pk=np.zeros((n+1,1))
            for e in range (n-k+1,n+1,2):
                pk[e]=(2*k-1)*pkm1[e+1]+(1-k)*pkm2[e]
            pk[n+1]=pk[n+1]+(1-k)*pkm2[n+1]
            pk=pk/k
            if k<n:
                pkm2=pkm1
                pkm1=pk
    P = np.polyval(pk,x)
    return P