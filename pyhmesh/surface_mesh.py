import numpy as np
from collections import ChainMap
import pyvista as pv

import vtk
import vtk.util.numpy_support as ns

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
        self.length = None

        if(model_fname is not None):
            self.generate(model_fname)


    def generate(self, fname, nbulk=50, ncurv=50):
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
        gmsh.clear()
        gmsh.model.add("pyhmesh_model")

        gmsh.merge(fname)

        self.dim = gmsh.model.getDimension()
        print(f"Reading {fname}...")
        print("Model dimension=",self.dim)

        #Get all regions
        domain=gmsh.model.get_entities(self.dim)
        domain=gmsh2pygmsh(domain)
        
        self.bbox = getBbox(domain)
        self.bbox_size = self.bbox[1]-self.bbox[0]
        self.length = np.max(self.bbox_size)

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", ncurv)
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.length/nbulk)

        #For 2d model, we need to convert it into 3d by extrusion
        #if(self.dim==2):
        #    top, vol_thick, lateral=extrude_plugin(gmsh.model.occ,
        #                            domain, [0, 0, -model_length/20], num_layers=1)      
        #    gmsh.model.occ.synchronize()
        
        #generate boundary mesh only
        gmsh.model.mesh.generate(2)

        #get all boundary entities
        self.domain = domain
        self.boundary = getBoundary(domain)
        print('domain',len(self.boundary))
        print('boundary', len(self.boundary))

        #gmsh.fltk.run()

    def getCurvature(self):
        """Create surface mesh using gmsh
        """
        x=1
    
    def nodes(self, bd_entities=None):
        """Get nodes from the surface mesh
        """
        if(bd_entities==None): 
            bd_entities = self.boundary
        print(f"Get nodes from {len(bd_entities)} boundary entites")
        
        nodes=[]
        for bd in bd_entities:
            #print('bd',bd.dim_tag)
            vtags, vxyz, vparam = gmsh.model.mesh.getNodes(*bd.dim_tag, includeBoundary=True)
            
            nodes_bd = dict( zip(vtags, vxyz.reshape(-1,3) ) )
            nodes.append(nodes_bd)
            #print(vtags)
        
        #union duplicates from the connected boundary
        #https://stackoverflow.com/a/3495395
        nodes=dict(ChainMap(*nodes))
        numNodes = len(nodes)

        vtags, nodes = np.array(list(nodes.keys())),  np.array(list(nodes.values()))
        vtags = dict(  zip(vtags,np.arange(numNodes,dtype=np.int64))  )
        
        self.nodeTagMap = vtags

        return vtags, nodes

    def elements(self, bd_entities=None):
        """Get elements from the surface mesh
           elements are grouped into a list for each boundary entity
        """
        if(bd_entities==None): 
            bd_entities = self.boundary
        print(f"Get eles from {len(bd_entities)} boundary entites")

        eles = []
        for bd in bd_entities:
            #print('bd',bd.dim_tag)
            etypes, etags, ntags = gmsh.model.mesh.getElements(*bd.dim_tag)
            assert len(etypes)==1, 'Mixed-type boundary mesh'
            #print(etypes,etags,ntags)

            etypes, etags, ntags = etypes[0], etags[0], ntags[0]
            eles_bd = ntags.reshape(-1, GmshEleNodeNum[etypes])
            eles+=[eles_bd]

        #eles = np.vstack(eles)
        return eles

    def getMesh(self, bd_entities=None, group_by_boundary = True, return_pyvista=False):
        """Get the surface mesh with renumbered index
        """
        vtags, nodes = self.nodes(bd_entities)
        eles=self.elements(bd_entities)

        #replace eles node tags based on vtags mapping
        k = np.array(list(vtags.keys()))
        v = np.array(list(vtags.values()))
        #print(k.dtype,v.dtype, k.max())

        mapping_ar = np.zeros( int(k.max()+1), dtype=v.dtype) #k,v from approach #1
        mapping_ar[k] = v
        eles = [ mapping_ar[es] for es in eles]

        if(group_by_boundary==False): 
            eles = np.vstack(eles)
        
        if(return_pyvista):
            eles = np.vstack(eles)
            return _toPVMesh(nodes,eles)

        return nodes,eles
    
    def voxlize(self, dx=0.0, pad_ratio=0.2, return_mesh=False):
        """Voxlize the surface mesh
        2D requires to extrude into 3d and using pyvista tool
        """
        if(dx==0.0): dx = self.length/100

        self.domain_feature = self.domain
        self.boundary_feature = self.boundary
        if(self.dim==2):
            gmsh.model.mesh.clear()
            top, vol_thick, lateral=extrude_plugin(gmsh.model.occ,
                                    self.domain, [0, 0, -dx*1.0], num_layers=3)
            gmsh.model.occ.synchronize()

            self.domain_feature = vol_thick
            self.boundary_feature = getBoundary(vol_thick)
            gmsh.model.mesh.generate(2)
            #gmsh.fltk.run()

        pvmesh = self.getMesh(self.boundary_feature, return_pyvista=True)

        img = PolyMesh2Image(pvmesh, resolution = dx, fname='test.mhd')

        if(return_mesh):
            return img, pvmesh
        return img

    def curvature(self):
        #compute curvature of surface mesh using gmsh

        curvatures=[]
        for bd in self.boundary_feature:
            #print('bd',bd.dim_tag)
            vtags, vxyz, vparam = gmsh.model.mesh.getNodes(*bd.dim_tag, includeBoundary=True)
            curv = gmsh.model.getCurvature(*bd.dim_tag, vparam)

            curvs_bd = dict( zip(vtags, curv ) )
            curvatures.append(curvs_bd)
            #print(vtags)
        
        #union duplicates from the connected boundary
        #https://stackoverflow.com/a/3495395
        curvatures=dict(ChainMap(*curvatures))

        #return np.array(list(curvatures.values()))
        curvatures = [curvatures[k] for k in self.nodeTagMap.keys()]

        return np.array(curvatures)

    
    def _initGmsh(self):
        if(self.gmsh_on==False):
            gmsh.initialize('',False)
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


def _toPVMesh(nodes,eles):
    #Create a pyvista surface mesh based on given nodes and elements
    connectivity = np.ones([len(eles),4], dtype=np.int32)*3
    connectivity[:,1:4] = eles

    return pv.PolyData(nodes, connectivity)

def PolyMesh2Image(pv_polymesh, resolution=0.5, 
                   inside_value=1, outside_value=0, 
                   fname=None):
    #Estimate image size based on resolution
    print("Convert polymesh to image.....")

    bnds = np.array(pv_polymesh.bounds)
    shape = np.ceil((bnds[1::2]-bnds[::2])/resolution).astype(int)
    extent = np.zeros(6).astype(int)
    extent[1::2] = shape-1
    
    spacing = (resolution,resolution,resolution)
    #origin = [bnds[ii*2] + spacing[ii] / 2 for ii in range(3)]
    origin = [bnds[ii*2] for ii in range(3)]
    print("Image Origin=",origin)
    print("Image Shape=",shape)
    print("Image Extent=",extent)

    # Create an white backgound image
    white_image = vtk.vtkImageData()
    white_image.SetOrigin(origin)
    white_image.SetSpacing(spacing)
    white_image.SetDimensions(shape)
    white_image.SetExtent(extent)
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    
    np_scalar=np.ones(shape[::-1]).ravel()*outside_value
    vtk_data_array = ns.numpy_to_vtk(num_array=np_scalar,deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    white_image.GetPointData().SetScalars(vtk_data_array)
    
    # polygonal data --> image stencil:
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(pv_polymesh)
    pol2stenc.SetOutputOrigin(white_image.GetOrigin())
    pol2stenc.SetOutputSpacing(white_image.GetSpacing())
    pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
    pol2stenc.Update()
    
    # cut the corresponding white image and set the background:
    imgstenc = vtk.vtkImageStencilToImage()
    imgstenc.SetInputData(pol2stenc.GetOutput())
    imgstenc.SetOutsideValue(outside_value)
    imgstenc.SetInsideValue(inside_value)
    imgstenc.SetOutput(white_image)
    imgstenc.Update()
    
    if(fname is not None):
        imageWriter = vtk.vtkMetaImageWriter()
        imageWriter.SetFileName(fname)
        imageWriter.SetInputConnection(imgstenc.GetOutputPort())
        imageWriter.Write()        
    
    # convert image data to numpy array
    img_scalar = ns.vtk_to_numpy(white_image.GetPointData().GetScalars())
    
    return img_scalar.reshape(shape[::-1])

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



