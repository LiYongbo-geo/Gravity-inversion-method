from geoist.inversion import geometry
from geoist.pfm import prism
from geoist.pfm import giutils
from geoist.inversion.mesh import PrismMesh
from geoist.vis import giplt
from geoist.inversion.regularization import Smoothness,Damping,TotalVariation
from geoist.inversion.pfmodel import SmoothOperator
from geoist.inversion.hyper_param import LCurve
from geoist.pfm import inv3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geoist import gridder

def greenmat(shape, nshape, density, area, narea, noisefree = True):
    
    mesh = PrismMesh(area, shape)
    kernel=[] 
    depthz = []
    xp, yp, zp = gridder.regular(narea, nshape, z=-1)
    for i, layer in enumerate(mesh.layers()):
        for j, p in enumerate(layer):
            x1 = mesh.get_layer(i)[j].x1
            x2 = mesh.get_layer(i)[j].x2
            y1 = mesh.get_layer(i)[j].y1
            y2 = mesh.get_layer(i)[j].y2
            z1 = mesh.get_layer(i)[j].z1
            z2 = mesh.get_layer(i)[j].z2
            den = mesh.get_layer(i)[j].props
            model=[geometry.Prism(x1, x2, y1, y2, z1, z2, {'density': 1000.})]
            field = prism.gz(xp, yp, zp, model)
            kernel.append(field)       
            depthz.append((z1+z2)/2.0)
    kk=np.array(kernel)        
    kk=np.transpose(kernel)  #kernel matrix for inversion, 500 cells * 400 points
    field0= np.mat(kk)*np.transpose(np.mat(density.ravel()))
    if noisefree:
        field = field0
    else:
        field = giutils.contaminate(np.array(field0).ravel(), 0.05, percent = True)

    return kk, field, np.array(depthz)