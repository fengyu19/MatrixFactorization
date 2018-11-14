# -*- coding: utf-8 -*-
from load_data import preprocess, random_create_data
import numpy as np


def MF(sensorDataMat, K, steps=5000, alpha=0.0002):
    m, n = sensorDataMat.shape
    P = np.random.rand(m, K)
    Q = np.random.rand(n, K)
    
    Q = Q.T
    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    eij = sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k])
    
        e = 0
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    e = e + pow(sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q


def MF_regularization(sensorDataMat, K, steps=5000, alpha=0.0002, beta=0.02):
    m, n = sensorDataMat.shape
    P = np.random.rand(m, K)
    Q = np.random.rand(n, K)
    
    Q = Q.T
    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    eij = sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    
        e = 0
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    e = e + pow(sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q
    
    
if __name__ == "__main__":
    contentDic = preprocess('test-data.txt')
    changed_sensorDataMat = random_create_data(contentDic, '19')
    P, Q = MF(changed_sensorDataMat, 2)