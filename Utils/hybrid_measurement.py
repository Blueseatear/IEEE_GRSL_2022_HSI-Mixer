# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:36:46 2022

@author: Blues
"""
import numpy as np
import sys 
def SAD(x,y):
    s = np.sum(np.dot(x,y))
    t = np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2))
    cos_value = s/t
    eps = 1e-6
    
    if 1.0 < cos_value < 1.0 + eps:
            cos_value = 1.0
    elif -1.0 - eps < cos_value < -1.0:
            cos_value = -1.0
    
    th = np.arccos(cos_value)
    return th

def SID(x,y):
    # print(np.sum(x))
    p = np.zeros_like(x,dtype=np.float)
    q = np.zeros_like(y,dtype=np.float)
    Sid = 0
    #x = abs(x)
    #y = abs(y)
    for i in range(len(x)):
        p[i] = x[i]/np.sum(x)
        q[i] = y[i]/np.sum(y)
        #print(p[i])
        # print(p[i],q[i])
    for j in range(len(x)):
        Sid += p[j]*np.log(p[j]/q[j])+q[j]*np.log(q[j]/p[j])
    return Sid

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def spectral_matrixs(inputs):
    cuboids = inputs.copy()
    cuboids = cuboids.reshape(cuboids.shape[0], cuboids.shape[1], np.prod(cuboids.shape[2:]))
    cuboids = sigmoid(cuboids)
    
    cen_idx = int((cuboids.shape[2]-1) / 2)
    
    #sad_matrix = np.zeros((cuboids.shape[0],cuboids.shape[2]))
    #sid_matrix = np.zeros((cuboids.shape[0],cuboids.shape[2]))
    hybrid_matrix = np.zeros((cuboids.shape[0],cuboids.shape[2]))

    for i in range(cuboids.shape[0]):
        for j in range(cuboids.shape[2]):
            sad_value = SAD(cuboids[i, :, cen_idx],cuboids[i, :, j])
            #sad_matrix[i,j] = SAD(cuboids[i, :, cen_idx],cuboids[i, :, j])
            sid_value = SID(cuboids[i, :, cen_idx],cuboids[i, :, j])
            #sid_matrix[i,j] = SID(cuboids[i, :, cen_idx],cuboids[i, :, j])
            hybrid_matrix[i,j] = sad_value * np.tan(sid_value)
            #hybrid_matrix[i,j] = sad_matrix[i,j] * np.tan(sid_matrix[i,j])
    #del cuboids
    return hybrid_matrix.reshape(inputs.shape[0],1,inputs.shape[2],inputs.shape[3])
