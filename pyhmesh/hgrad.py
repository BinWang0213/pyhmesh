#   Copyright (C) 2021 Bin Wang
'''
Reference
---------
Eq. 10, Fig. 4
Mesh size functions for implicit geometries and PDE-based
gradient limiting, Persson, PO. https://doi.org/10.1007/s00366-006-0014-1

Section 2.4
Automatic feature-preserving size field for 3D mesh generation, 
Arthur Bawin,  https://arxiv.org/abs/2009.03984
'''

import numpy as np
from numba import jit


def hgrad_limit(hfield, dx=1.0, hgradation=1.1, guess_h0=False):
    """Regularize the mesh size function based on length ratio of two adjacent edges

    Reference
    ---------
    Eq. 10, Fig. 4
    Mesh size functions for implicit geometries and PDE-based
    gradient limiting, Persson, PO. https://doi.org/10.1007/s00366-006-0014-1

    Section 2.4
    Automatic feature-preserving size field for 3D mesh generation, 
    Arthur Bawin,  https://arxiv.org/abs/2009.03984

    Current only support Cartesian mesh, Octree could be added in the future

    Parameters
    ----------
    hfield : object
        Input mesh size function object using cartesian mesh
        Cartesian mesh -> numpy NxNxN array
    dx : float
        unit mesh size in the Cartesian mesh
    hgradation : float
        length ratio of two adjacent edges in the final mesh
    guess_h0 : bool
        we could give an initial solution using numpy.gradient to locate sharp region
        this may acclerate the function for large problem

    Author: Bin Wang (binwang.0213@gmail.com)
    """
    grad_limit = hgradation-1.0

    aset=None
    if(guess_h0): #Using numpy gradient to prepare the initial condition
        grads = np.gradient(hfield)
        grad_limit_mask=[np.abs(dh)>grad_limit for dh in grads]
        grad_limit_mask=np.logical_or.reduce(grad_limit_mask)

        aset = np.zeros_like(grad_limit_mask,dtype=np.int32)
        aset[grad_limit_mask]=1
        aset-=1
        aset = aset.flatten()

    #Convert 2D shape list into 3D
    dim = hfield.ndim
    dims = hfield.shape
    if dim == 2: dims = (dims[0], dims[1], 1)

    return fastHJ(hfield, dims, dx=dx, hgradation=hgradation, aset0=aset)

##---------------Numba Accelerated C-like Funcs------------------

@jit(nopython=True)
def getI_J_K(ijk,shape):
    #Find index [i,j,k] from a flat 3D matrix index [i,j,k]
    NX,NY,NZ = shape
    i,j,k=0,0,0
    
    #Col major
    #i=ijk%NX
    #j=((int)(ijk / NX)) % NY
    #k=(int)(ijk / (NX*NY))

    #Row major
    k = ijk%NZ
    j = ((int)(ijk / NZ)) % NY
    i = (int)(ijk / (NZ*NY))
    
    return i,j,k

@jit(nopython=True)
def getIJK(i,j,k,shape):
    #Convert index [i,j,k] to a flat 3D matrix index [ijk]
    NX,NY,NZ = shape
    
    #Col major
    #return i + (NX)*(j + k*(NY))

    #Row major
    return k + NZ*(j+i*NY)

@jit(nopython=True)
def fastHJ(ffun, dims, dx, hgradation, imax=10000, aset0=None):
    ftol = np.min(ffun)*np.sqrt(1e-9)

    dfdx = hgradation-1.0
    elen = dx
    
    npos = np.zeros(7,dtype=np.int32)
    
    #output field, convert into 1d for generic nD indexing
    ffun_s = np.empty_like(ffun.size,dtype=np.float64)
    ffun_s = ffun.flatten()
    
    #we only search the cell near the sharp region which masked by aset
    if(aset0 is not None):
        aset = aset0
    else:
        aset = np.zeros_like(ffun_s,dtype=np.int32)

    #print('Iteration #Active cell')
    for it in range(1,imax+1):
        aidx = np.where(aset==it-1)[0] #Find gradient sharp region
        
        #print(it,'\t',len(aidx) )
        if(len(aidx)==0): break
        
        for idx in range(len(aidx)):#Check gradient and change size function
            IJK = aidx[idx]
            
            I,J,K = getI_J_K(IJK, dims)
            
            #Gather indices using 4 (6 in 3d) edge stencil centered on inod
            npos[0] = IJK

            npos[1] = getIJK(min(I+1,dims[0]-1), J, K, dims)
            npos[2] = getIJK(max(I-1,0), J, K, dims)

            npos[3] = getIJK(I, min(J+1, dims[1]-1), K, dims)
            npos[4] = getIJK(I, max(J-1,0), K, dims)

            npos[5] = getIJK(I, J, min(K+1,dims[2]-1), dims)
            npos[6] = getIJK(I, J, max(K-1,0), dims)

            
            #----------------- calc. limits about min.-value
            nod1 = npos[0]
            for p in range(1,7):
                nod2 = npos[p]
                #if(nod1==nod2): continue                
                #print(p,nod1,nod2)
                if (ffun_s[nod1] > ffun_s[nod2]):
                    fun1 = ffun_s[nod2] + elen * dfdx
                    if (ffun_s[nod1] > fun1 + ftol):
                        ffun_s[nod1] = fun1
                        aset[nod1] = it
                else:
                    fun2 = ffun_s[nod1] + elen * dfdx
                    if (ffun_s[nod2] > fun2 + ftol):
                        ffun_s[nod2] = fun2
                        aset[nod2] = it
    
    return np.reshape(ffun_s, ffun.shape)








   
