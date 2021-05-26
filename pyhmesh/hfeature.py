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


def hfeature_limit(surface_mesh, ng=4):
    """this will create a mesh size map to consider local feature size
       local feature size: a measure of the distance between nearby boundaries

    Noted that 2d and 3d has different treatment method
    Key algorithm is to find the medial axis

    Parameters
    ----------
    surface_mesh : object
        Surface Mesh defined in surface_mesh.py
    ng : int
        the number of element layers in thin gaps

    Author: Bin Wang (binwang.0213@gmail.com)
    """
    if(surface_mesh.dim==2): #image-based method
        x=1


    return fastHJ(hfield, dims, dx=dx, hgradation=hgradation, aset0=aset)








   
