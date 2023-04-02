import numpy as np
import numba as nb


def Lx(nmodel=(3, 3, 3)):
    """
    计算一阶光滑矩阵
    参数：
    nmodel: 模型形态（nz, ny, nz） 
    """
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-ylen*zlen, model_length))
    for i in range(zlen):
        for j in range(ylen):
            for k in range(xlen-1):
                index_x = i*ylen*(xlen-1) + j*(xlen-1) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+1] = 1
    return ma

def Ly(nmodel=(3, 3, 3)):
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-ylen*xlen, model_length))
    for i in range(zlen):
        for j in range(ylen-1):
            for k in range(xlen):
                index_x = i*ylen*(xlen-1) + j*(xlen) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+xlen] = 1
    return ma

def Lz(nmodel=(3, 3, 3)):
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-ylen*xlen, model_length))
    for i in range(zlen-1):
        for j in range(ylen):
            for k in range(xlen):
                index_x = i*ylen*(xlen) + j*(xlen) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+xlen*ylen] = 1
    return ma

def Lxx(nmodel=(3, 3, 3)):
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-2*ylen*zlen, model_length))
    for i in range(zlen):
        for j in range(ylen):
            for k in range(xlen-2):
                index_x = i*ylen*(xlen-2) + j*(xlen-2) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+1] = 2
                ma[index_x, index_y+2] = -1
    return ma

def Lyy(nmodel=(3, 3, 3)):
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-2*ylen*xlen, model_length))
    for i in range(zlen):
        for j in range(ylen-2):
            for k in range(xlen):
                index_x = i*ylen*(xlen-2) + j*(xlen) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+xlen] = 2
                ma[index_x, index_y+2*xlen] = -1
    return ma

def Lzz(nmodel=(3, 3, 3)):
    
    model_length = int(nmodel[0]*nmodel[1]*nmodel[2])
    zlen = nmodel[0]
    ylen = nmodel[1]
    xlen = nmodel[2]
    ma = np.zeros(shape=(model_length-2*ylen*xlen, model_length))
    for i in range(zlen-2):
        for j in range(ylen):
            for k in range(xlen):
                index_x = i*ylen*(xlen) + j*(xlen) + k
                index_y = i*ylen*xlen + j*xlen + k
                ma[index_x, index_y] = -1
                ma[index_x, index_y+xlen*ylen] = 2
                ma[index_x, index_y+2*xlen*ylen] = -1
    return ma