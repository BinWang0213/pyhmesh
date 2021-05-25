import numpy as np
from collections import ChainMap

from .pygmsh_plugin import *

class SurfaceMesh:
    """Surface mesh object to extract key informations (curvature, medial axis, distance etc) 
    to define mesh size field

    Parameters
    ----------
    hfield : object
        Input mesh size function object using cartesian mesh
        Cartesian mesh -> numpy NxNxN array

    Author: Bin Wang (binwang.0213@gmail.com)
    """

    def __init__(self, model_fname=None):
        self.dim = 2  

        self.gmsh_on=False
        self._initGmsh()

        self.bbox = None
        self.bbox_size = None

        if(model_fname is not None):
            self.generate(model_fname)


    def generate(self, fname, nbulk=20, ncurv=50):
        """Create surface mesh using gmsh

        As mesh tools only support evaluate curvature and thickness for 3d model
        2d model will be extruded by 1 layer

        Parameters
        ----------
        fname : str
            model file name supported by gmsh. step, geo, etc
        nbulk : float
            the number of elements acrossing the largest dimension of model bbox
            this is used to constrain the maximum mesh size
        ncurv : float
            the number of elements nd used to accurately discretize a complete circle
        """
        gmsh.merge(fname)

        self.dim = gmsh.model.getDimension()

        print("Model dimension=",self.dim)

        #Get all regions
        domain=gmsh.model.get_entities(self.dim)
        domain=gmsh2pygmsh(domain)
        
        self.bbox = getBbox(domain)
        self.bbox_size = self.bbox[1]-self.bbox[0]
        model_length = np.max(self.bbox_size)

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", ncurv)
        gmsh.option.setNumber("Mesh.MeshSizeMax", model_length/nbulk)

        #For 2d model, we need to convert it into 3d by extrusion
        if(self.dim==2):
            top, vol_thick, lateral=extrude_plugin(gmsh.model.occ,
                                    domain, [0, 0, -model_length/20], num_layers=1)      
            gmsh.model.occ.synchronize()
        
        #generate boundary mesh only
        gmsh.model.mesh.generate(2)

        #get all boundary entities
        self.domain = domain
        self.boundary = getBoundary(domain)
        print('domain',self.boundary)
        print('boundary',self.boundary)

        gmsh.fltk.run()

    def getCurvature(self):
        """Create surface mesh using gmsh
        """
        x=1
    
    def nodes(self):
        """Get nodes from the surface mesh
        """
        print(f"Get nodes from {len(self.boundary)} boundary entites")
        
        nodes=[]
        for bd in self.boundary:
            #print('bd',bd.dim_tag)
            vtags, vxyz, _ = gmsh.model.mesh.getNodes(*bd.dim_tag, includeBoundary=True)
            nodes_bd = dict( zip(vtags, vxyz.reshape(-1,3) ) )
            nodes.append(nodes_bd)
            #print(vtags)
        
        #union duplicates from the connected boundary
        #https://stackoverflow.com/a/3495395
        nodes=dict(ChainMap(*nodes))
        numNodes = len(nodes)

        vtags, nodes = np.array(list(nodes.keys())),  np.array(list(nodes.values()))
        vtags = dict(  zip(vtags,np.arange(numNodes,dtype=np.int64))  )
        
        return vtags, nodes

    def elements(self):
        """Get elements from the surface mesh
        """
        eles = []
        for bd in self.boundary:
            #print('bd',bd.dim_tag)
            etypes, etags, ntags = gmsh.model.mesh.getElements(*bd.dim_tag)
            assert len(etypes)==1, 'Mixed-type boundary mesh'
            #print(etypes,etags,ntags)

            etypes, etags, ntags = etypes[0], etags[0], ntags[0]
            eles_bd = ntags.reshape(-1, GmshEleNodeNum[etypes])
            eles+=[eles_bd]

        #eles = np.vstack(eles)
        return eles

    def getMesh(self):
        """Get the surface mesh with renumbered index
        """
        vtags, nodes = self.nodes()
        eles=self.elements()

        #replace eles node tags based on vtags mapping
        k = np.array(list(vtags.keys()))
        v = np.array(list(vtags.values()))
        print(k.dtype,v.dtype, k.max())

        mapping_ar = np.zeros( int(k.max()+1), dtype=v.dtype) #k,v from approach #1
        mapping_ar[k] = v
        eles = [ mapping_ar[es] for es in eles]

        return nodes,eles

    def _initGmsh(self):
        if(self.gmsh_on==False):
            gmsh.initialize('',False)
            gmsh.model.add("pyhmesh_model")
            self.gmsh_on=True
    
    def close(self):
        #close gmsh instance
        if(self.gmsh_on):
            try:
                # Gmsh >= 4.7.0
                # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1036
                gmsh.model.mesh.removeSizeCallback()
            except AttributeError:
                pass
            gmsh.finalize()

def _closeGmsh():
    try:
        # Gmsh >= 4.7.0
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1036
        gmsh.model.mesh.removeSizeCallback()
    except AttributeError:
        pass
    gmsh.finalize()

def _generateMesh(fname, nd, nb):
    """Generate surface mesh using gmsh

    Parameters
    ----------
    fname : str
        model file name supported by gmsh. step, geo, etc
    nd : float
        the number of elements nd used to accurately discretize a complete circle
    nb : float
        the number of elements acrossing the largest dimension of model bbox
        this is used to constrain the maximum mesh size

    Author: Bin Wang (binwang.0213@gmail.com)
    """
    gmsh.merge(fname)
    print('Model filename=',fname)

    dim = gmsh.model.getDimension()
    dim_orig = dim #used for 2d model
    print("Model dimension=",dim)

    #Get all regions from CAD
    domain=gmsh.model.get_entities(dim)
    domain=gmsh2pygmsh(domain)
    
    bbox = getBbox(domain)
    model_length = np.linalg.norm(bbox[1]-bbox[0])

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", nd)
    gmsh.option.setNumber("Mesh.MeshSizeMax", model_length/20)

    #For 2d model, we need to convert it into 3d
    if(dim==2):
        top, vol_thick, lateral=extrude_plugin(gmsh.model.occ,
                                domain, [0, 0, -model_length/20], num_layers=1)    
        dim=3
        gmsh.model.occ.synchronize()
    
    #generate boundary mesh only
    gmsh.model.mesh.generate(dim-1)

    #gmsh.fltk.run()
    
    #Get node-wise curvaure on the boundary
    bd_dim = dim - 1 
    ent = gmsh.model.getEntities(bd_dim)

    nodes=[]
    normals=[]
    curvatures=[]
    for dim,tag in ent:
        #nodes on boundary
        tags, coord, param = gmsh.model.mesh.getNodes(dim, tag, True)
        normal = gmsh.model.getNormal(tag, param)
        curv = gmsh.model.getCurvature(dim, tag, param)

        nodes+=[coord.reshape(-1,3)]
        normals+=[normal.reshape(-1,3)]
        curvatures+=list(curv)

    nodes=np.vstack(nodes)
    normals=np.vstack(normals)
    curvatures=np.array(curvatures)
    print(nodes.shape,normals.shape,curvatures.shape)
    print('Curvature range=',np.min(curvatures),np.max(curvatures))

    if(dim_orig==2):
        mask = nodes[:,2]>-1e-5
        print(mask.shape,np.sum(mask))
        return nodes[mask,:], curvatures[mask]

    return nodes,curvatures



