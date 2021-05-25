from typing import List, Optional, Tuple, Union

import numpy as np
from pygmsh.occ.dummy import Dummy
import gmsh
import pygmsh
import math

import matplotlib.pyplot as plt


#Current only support Simplex element
GmshEleType={
    15:'POINT',
    2:'TRIANGLE',
    4:'TETRA',
    1:'LINE'
}

GmshEleTypeID={
    'POINT':15,
    'TRIANGLE':2,
    'TETRA':4,
    'LINE':1
}

GmshEleNodeNum={
    15:1,
    2:3,
    4:4,
    1:2
}

PosEleTypeName={
    15:'SP',
    2:'ST',
    4:'SS'
}

##---------------------Improved facility--------------------
def generate_mesh(  # noqa: C901
        geom,
        dim: int = 3,
        order: Optional[int] = None,
        # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
        algorithm: Optional[int] = None,
        verbose: bool = False,
        return_meshio: bool = True,
    ):
        """Return a meshio.Mesh, storing the mesh points, cells, and data, generated by
        Gmsh from the `self`.
        """
        geom.synchronize()

        for item in geom._AFTER_SYNC_QUEUE:
            item.exec()

        for item, host in geom._EMBED_QUEUE:
            gmsh.model.mesh.embed(item.dim, [item._id], host.dim, host._id)

        # set compound entities after sync
        for c in geom._COMPOUND_ENTITIES:
            gmsh.model.mesh.setCompound(*c)

        for s in geom._RECOMBINE_ENTITIES:
            gmsh.model.mesh.setRecombine(*s)

        for t in geom._TRANSFINITE_CURVE_QUEUE:
            gmsh.model.mesh.setTransfiniteCurve(*t)

        for t in geom._TRANSFINITE_SURFACE_QUEUE:
            gmsh.model.mesh.setTransfiniteSurface(*t)

        for e in geom._TRANSFINITE_VOLUME_QUEUE:
            gmsh.model.mesh.setTransfiniteVolume(*e)

        for item, size in geom._SIZE_QUEUE:
            gmsh.model.mesh.setSize(
                gmsh.model.getBoundary(item.dim_tags, False, False, True), size
            )

        for entities, label in geom._PHYSICAL_QUEUE:
            d = entities[0].dim
            assert all(e.dim == d for e in entities)
            tag = gmsh.model.addPhysicalGroup(d, [e._id for e in entities])
            if label is not None:
                gmsh.model.setPhysicalName(d, tag, label)

        for entity in geom._OUTWARD_NORMALS:
            gmsh.model.mesh.setOutwardOrientation(entity.id)

        if order is not None:
            gmsh.model.mesh.setOrder(order)

        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)

        # set algorithm
        # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
        if algorithm:
            gmsh.option.setNumber("Mesh.Algorithm", algorithm)

        gmsh.model.mesh.generate(dim)

        if(return_meshio): return pygmsh.helpers.extract_to_meshio()

def extrude_plugin(env,input_entity,
    translation_axis: Tuple[float, float, float],
    num_layers: Optional[Union[int, List[int]]] = None,
    heights: Optional[List[float]] = None,
    recombine: bool = False):
    """Extrusion of any entity along a given translation_axis."""
    #https://github.com/nschloe/pygmsh/blob/main/src/pygmsh/common/geometry.py

    if isinstance(num_layers, int):
            num_layers = [num_layers]
    if num_layers is None:
        num_layers = []
        assert heights is None
        heights = []
    else:
        if heights is None:
            heights = []
        else:
            assert len(num_layers) == len(heights)

    assert len(translation_axis) == 3

    ie_list = input_entity if isinstance(input_entity, list) else [input_entity]

    out_dim_tags = env.extrude(
        [e.dim_tag for e in ie_list],
        *translation_axis,
        numElements=num_layers,
        heights=heights,
        recombine=recombine,
    )
    top = Dummy(*out_dim_tags[0])
    extruded = Dummy(*out_dim_tags[1])
    lateral = [Dummy(*e) for e in out_dim_tags[2:]]
    return top, extruded, lateral

##-----------------------Object query-----------------------

def gmsh2pygmsh(objs):
    return [Dummy(*o) for o in objs]

def createVolumes(tags):
    if(isinstance(tags,list)==False): tags=[tags]
    return [Dummy(3,id) for id in tags]

def createSurfaces(tags):
    if(isinstance(tags,list)==False): tags=[tags]
    return [Dummy(2,id) for id in tags]

def createCurves(tags):
    if(isinstance(tags,list)==False): tags=[tags]
    return [Dummy(1,id) for id in tags]

def gmshCoherence(objs):
    #resolve overlap geometries
    if(isinstance(objs,list)==False): objs=[objs]
    dim_tags, _ = gmsh.model.occ.fragment([d.dim_tag for d in objs],[])
    gmsh.model.occ.synchronize()

def getBoundary(objs,combined=True, oriented=True, recursive=False):
    #get the boundary of a object
    if(isinstance(objs,list)==False): objs=[objs]
    dim_tags = gmsh.model.get_boundary([d.dim_tag for d in objs],
                                        combined=False,  #gmsh has issues on combine 
                                        oriented=oriented, 
                                        recursive=recursive)
    if(combined): dim_tags=list(dict.fromkeys(dim_tags))
    return [Dummy(dim_tag[0],dim_tag[1]) for dim_tag in dim_tags]

def getBbox(objs):
    if(isinstance(objs,list)==False): objs=[objs]
    bboxs=[]
    for region in objs:
        dim,tag=region.dim_tag
        bbox=gmsh.model.get_bounding_box(dim,tag)
        bboxs+=[bbox]
    bboxs=np.array(bboxs)
    pts_min = np.min(bboxs[:,0:3],axis=0)
    pts_max = np.max(bboxs[:,3:6],axis=0)
    return pts_min,pts_max

def calcVolume(physicalGropu=-1):
    gmsh.plugin.setNumber("MeshVolume", "Dimension", gmsh.model.getDimension())
    gmsh.plugin.setNumber("MeshVolume", "PhysicalGroup", physicalGropu)
    gmsh.plugin.run("MeshVolume")

    tags=gmsh.view.getTags()
    _, _, data = gmsh.view.getListData(tags[-1])
    gmsh.view.remove(tags[-1])
    return data[0][3]

def evalMeshQuality(fname=None, tags=None, data=None, modelID=0, isPlot=False):
    #https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/demos/api/mesh_quality.py
    
    if(data is None):
        with pygmsh.occ.Geometry() as geom:
            if(fname is not None): gmsh.merge(fname)

            gmsh.plugin.setNumber("AnalyseMeshQuality", "ICNMeasure", 1.)
            gmsh.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1.)
            gmsh.plugin.run("AnalyseMeshQuality")

            #model_tags=gmsh.view.getTags()
            #print(model_tags)
            dataType, tags, data, time, numComp = gmsh.view.getModelData(modelID, 0)

        #tags=np.array(tags).flatten()
        data=np.array(data).flatten()

    if(isPlot):
        from matplotlib.ticker import PercentFormatter
        fig=plt.figure(figsize=(5,4),dpi=80)
        plt.hist(data,bins=50, color="grey",weights=np.ones(len(data)) / len(data))
        s = "Min: %.2f, Mean: %.2f, Max: %.2f" % (
            np.min(data), np.mean(data), np.max(data))
        plt.title(s)
        plt.xlabel("Mesh quality (Inverse Condition Number)")
        plt.xlim(0, 1)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    print('NumEles=',len(data))
    print('MeshQuality min/max = ',[np.min(data),np.max(data)])
    return data


##-----------------------Size Fields-----------------------

class Box:
    def __init__(
        self,
        bbox,
        lc_in,
        lc_out,
        thickness=None
    ):
        self.bbox=bbox
        self.lc_in=lc_in
        self.lc_out=lc_out
        self.thickness=thickness

        self.tag = gmsh.model.mesh.field.add("Box")
    
    def exec(self):
        gmsh.model.mesh.field.setNumber(self.tag, "VIn", self.lc_in)
        gmsh.model.mesh.field.setNumber(self.tag, "VOut", self.lc_out)
        
        if(len(self.bbox)==6):
            for i,v in enumerate(["XMin","YMin","ZMin","XMax","YMax","ZMax"]):
                gmsh.model.mesh.field.setNumber(self.tag, v, self.bbox[i])
        if(len(self.bbox)==4):
            for i,v in enumerate(["XMin","YMin","XMax","YMax"]):
                gmsh.model.mesh.field.setNumber(self.tag, v, self.bbox[i])
        
        if(self.thickness is not None):
            gmsh.model.mesh.field.setNumber(self.tag, "Thickness", self.thickness)

def add_box_field(geom,*args, **kwargs):
    size_field = Box(*args, **kwargs)
    geom._AFTER_SYNC_QUEUE.append(size_field)
    return size_field


class Restrict:
    def __init__(
        self,
        field_in,
        edges_list=None,
        faces_list=None,
        nodes_list=None,
        vols_list=None
    ):
        self.field_in = field_in
        self.edges_list = edges_list if edges_list else []
        self.faces_list = faces_list if faces_list else []
        self.nodes_list = nodes_list if nodes_list else []
        self.vols_list = vols_list if vols_list else []

        self.tag = gmsh.model.mesh.field.add("Restrict")
    
    def exec(self):
        gmsh.model.mesh.field.setNumber(self.tag, "InField", self.field_in.tag)

        # all restrict must be done down to the edges
        #http://onelab.info/pipermail/gmsh/attachments/20200427/70e5debc/attachment.geo

        if self.vols_list:
            gmsh.model.mesh.field.setNumbers(
                self.tag, "VolumesList", [v._id for v in self.vols_list]
            )
            faces = getBoundary(self.vols_list, oriented=False)
            edges = getBoundary(faces, oriented=False)
            self.faces_list+=faces
        if self.faces_list:
            gmsh.model.mesh.field.setNumbers(
                self.tag, "SurfacesList", [f._id for f in self.faces_list]
            )
            edges = getBoundary(faces, oriented=False)
            self.edges_list+=edges
        if self.edges_list:
            gmsh.model.mesh.field.setNumbers(
                self.tag, "CurvesList", [e._id for e in self.edges_list]
            )
        if self.nodes_list:
            gmsh.model.mesh.field.setNumbers(
                self.tag, "PointsList", [n._id for n in self.nodes_list]
            )

def add_restrict_field(geom, *args, **kwargs):
    size_field = Restrict(*args, **kwargs)
    geom._AFTER_SYNC_QUEUE.append(size_field)
    return size_field

class BoundaryLayer:
    def __init__(
        self,
        lcmin,
        lcmax,
        distmin,
        distmax,
        edges_list=None,
        faces_list=None,
        nodes_list=None,
        tag_dist_field=None,
        num_points_per_curve=None,
        stop_at_max=None,
        sigmoid_interp=False
    ):
        self.lcmin = lcmin
        self.lcmax = lcmax
        self.distmin = distmin
        self.distmax = distmax
        # Don't use [] as default argument, cf.
        # <https://stackoverflow.com/a/113198/353337>
        self.edges_list = edges_list if edges_list else []
        self.faces_list = faces_list if faces_list else []
        self.nodes_list = nodes_list if nodes_list else []
        self.tag_dist_field = tag_dist_field
        self.num_points_per_curve = num_points_per_curve
        self.stop_at_max=stop_at_max
        self.sigmoid_interp = sigmoid_interp

        if(self.tag_dist_field is None):
            self.tag_dist_field = gmsh.model.mesh.field.add("Distance")
        self.tag = gmsh.model.mesh.field.add("Threshold")

    def exec(self):
        tag1 = self.tag_dist_field
        tag2 = self.tag

        if self.edges_list:
            gmsh.model.mesh.field.setNumbers(
                tag1, "EdgesList", [e._id for e in self.edges_list]
            )
            # edge nodes must be specified, too, cf.
            # <https://gitlab.onelab.info/gmsh/gmsh/-/issues/812#note_9454>
            # nodes = list(set([p for e in self.edges_list for p in e.points]))
            # gmsh.model.mesh.field.setNumbers(tag1, "NodesList", [n._id for n in nodes])
        if self.faces_list:
            gmsh.model.mesh.field.setNumbers(
                tag1, "FacesList", [f._id for f in self.faces_list]
            )
        if self.nodes_list:
            gmsh.model.mesh.field.setNumbers(
                tag1, "NodesList", [n._id for n in self.nodes_list]
            )
        if self.num_points_per_curve:
            gmsh.model.mesh.field.setNumber(
                tag1, "NumPointsPerCurve", self.num_points_per_curve
            )

        gmsh.model.mesh.field.setNumber(tag2, "IField", tag1)
        gmsh.model.mesh.field.setNumber(tag2, "LcMin", self.lcmin)
        gmsh.model.mesh.field.setNumber(tag2, "LcMax", self.lcmax)
        gmsh.model.mesh.field.setNumber(tag2, "DistMin", self.distmin)
        gmsh.model.mesh.field.setNumber(tag2, "DistMax", self.distmax)

        if self.sigmoid_interp:
            gmsh.model.mesh.field.setNumber(tag2, "Sigmoid", self.sigmoid_interp)
        if self.stop_at_max:
            gmsh.model.mesh.field.setNumber(tag2, "StopAtDistMax", self.stop_at_max)


def add_boundary_layer_field(geom,*args, **kwargs):
    size_field = BoundaryLayer(*args, **kwargs)
    geom._AFTER_SYNC_QUEUE.append(size_field)
    return size_field

def twoStepBoundaryLayer(geom,sizeFields, faces_list, hmin,hmax, 
                         distmax, bl_thickness, stop_at_max):
    #sf_field=add_boundary_layer_field(geom,
    #        faces_list=faces_list,
    #        lcmin=hmin,  lcmax=hmin,
    #        distmin=0.0,  distmax=bl_thickness,
    #        num_points_per_curve=100,
    #        stop_at_max=1
    #        )
    #sizeFields+=[sf_field]

    
    sf_field=add_boundary_layer_field(geom,
            faces_list=faces_list,
            #tag_dist_field = sf_field.tag_dist_field,
            #lcmin=(hmin+hmax)/2.0,  lcmax=hmax,
            lcmin=hmax,  lcmax=hmax,
            #distmin=bl_thickness/2.0,  distmax=distmax,
            distmin=0.0,  distmax=distmax,
            num_points_per_curve=100,
            stop_at_max=stop_at_max
            )
    sizeFields+=[sf_field]
    

    sf_field=add_boundary_layer_field(geom,
            faces_list=faces_list,
            tag_dist_field = sf_field.tag_dist_field,
            #lcmin=(hmin+hmax)/2.0,  lcmax=hmax*2,
            lcmin=(hmax),  lcmax=hmax*2,
            #distmin=bl_thickness/2.0,  distmax=distmax*2,
            distmin=0.0,  distmax=distmax*2.4,
            num_points_per_curve=100,
            stop_at_max=stop_at_max
            )
    sizeFields+=[sf_field]

    return sizeFields