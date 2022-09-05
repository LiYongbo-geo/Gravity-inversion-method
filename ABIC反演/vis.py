from datetime import datetime
import argparse
from fileinput import filename
import numpy as np
import pathlib
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm
from torch.nn import functional as F

def Model(m, w, filename = "density"):
    m = m.T
    den_max = np.max(m)
    den_min = np.min(m)
    m = (m-den_min)/(den_max-den_min)
    L, W, H= m.shape
    c = ["#D1FEFE", "#D1FEFE", "#00FEF9", "#00FDFE", "#50FB7F", "#D3F821", "#FFDE00", "#FF9D00", "#F03A00", "#E10000"]
    x, y, z = np.indices((L, W, H))
    model = (x < 0) & (y < 0) & (z < 0)
    color = np.empty(m.shape, dtype=object)
    for i in range(L):
        for j in range(W):
            for k in range(H):
                if m[i][j][k] >= w and m[i][j][k] <=1:
                    cube = (x > i-1) & (x <= i)& (y > j-1) & (y <= j) & (z > k-1) & (z <= k)
                    color[cube] = c[int(round(10*m[i][j][k]))-1]
                    model = model | cube
    plt_model(model, color, filename)

def plt_model(model, facecolors, filename = "density"):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.gca(projection='3d')
    ax.voxels(model, facecolors=facecolors)

    ax.set_zlabel('Depth (km)', labelpad=16)
    ax.invert_zaxis()
    ax.xaxis.set_tick_params(pad=-2)
    ax.yaxis.set_tick_params(pad=-2)
    ax.zaxis.set_tick_params(pad=10)
    path = ""
    #pngpath = os.path.join(path, filename+".png")
    #pdfpath = os.path.join(path, filename+".pdf")
    #epspath = os.path.join(path, filename+".eps")
    #cb = plt.colorbar(cm.ScalarMappable(norm=plt.Normalize(0,1), cmap=colorma()),
    #                  shrink=0.5, aspect=20, pad = 0.09, label=r"density$(g cm^{-3} )$")
    
    #plt.savefig(pngpath)
    #plt.savefig(pdfpath)
    #plt.savefig(epspath)
    plt.show()

def plt_m(y, color='b'):
    x = range(1, len(y)+1)
    plt.xlim(-100, 16500)
    plt.ylim(0, 1.01)
    plt.xticks(np.arange(0, 16500, step=1024), fontsize=4)
    for i in range(len(y)):
        if y[i]==1:
            plt.scatter(x[i], y[i],linewidths=0.001)
    plt.plot(x, y, linewidth=0.8, color=color, linestyle=":")
    

def colorma():
    cdict = ["#F2F2F2", "#D1FEFE", "#00FEF9", "#00FDFE", "#50FB7F", "#D3F821", "#FFDE00", "#FF9D00", "#F03A00", "#E10000"] 
    return colors.ListedColormap(cdict, 'indexed')