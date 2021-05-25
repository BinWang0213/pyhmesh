import numpy as np

import gmsh
import pygmsh

from .pygmsh_plugin import *

class BackgroundMesh:
    #Background mesh with size field

    def __init__(self, model_fname):
        self.fname = model_fname
        self.dim = 2        
        self.geom_bbox = None
        self.geom_dimension = None

        self.hfield = None
        self.hfield_bbox = None
        self.dx=1e15

        self.hmin = 0.0
        self.hmax = 1e10

        self.loadModel(self.fname)
        self._initField()
    
    def loadModel(self,fname):
        #Load input model from a given file
        #model_fname could be any geometry type support by gmsh
        with pygmsh.occ.Geometry() as geom:
            gmsh.merge(fname)
            print('Model filename=',fname)

            self.dim = gmsh.model.getDimension()
            print("Model dimension=",self.dim)

            domain=gmsh.model.get_entities(self.dim)
            domain=gmsh2pygmsh(domain)
            self.model_bbox =  getBbox(domain)
            print("Model bbox=\n  ",self.model_bbox)

            #Scale the bounding box by a scale ratio to enclose whole model in it
            expand_ratio = 0.2
            bbox_size = self.model_bbox[1] - self.model_bbox[0]
            self.geom_dimension = bbox_size

            self.hfield_bbox = [ self.model_bbox[0] - bbox_size*expand_ratio,
                                 self.model_bbox[1] + bbox_size*expand_ratio ]
            if(self.dim==2): 
                self.hfield_bbox[0][2]=0.0
                self.hfield_bbox[1][2]=0.0
            
            print(f"Background mesh bbox (expanded by {expand_ratio})=\n  ",self.hfield_bbox)

    def _initField(self):
        #Initlize the mesh size field with default parameters
        self.hmin = np.max(self.geom_dimension)/1000
        self.hmax = np.max(self.geom_dimension)/20.0

    def setField(self, dx=None, dims=[1,1,1]):
        #Initlize the hfield by given unit cell size (dx) or dimension
        size = self.hfield_bbox[1]-self.hfield_bbox[0]

        if(dx is not None):
            dims = size/dx
            if(self.dim==2): dims=dims[0:2]  #fix for 2D case
            dims = np.array(dims,dtype=np.int32)

            self.hfield = np.empty(dims,dtype=np.float64)
        else:
            self.hfield = np.empty(dims,dtype=np.float64)

        self.dx = size[0]/dims[0]
        print(f"Set size field with dimension= {dims} cell size= {dx}")