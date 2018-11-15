# -*- coding: utf-8 -*-
from load_data import preprocess, random_create_data
import numpy as np


def NMF_normal(sensorDataMat, K, steps=5000, e=0.001):

    m, n = sensorDataMat.shape
    W = np.mat(np.random.rand(m, K))
    H = np.mat(np.random.rand(K, n))
    
    
    for step in range(steps):
        a = np.dot(W.T, sensorDataMat)
        b = np.dot(W.T, np.dot(W, H))
        for i_1 in range(K):
            for j_1 in range(n):
                if b[i_1,j_1] != 0:
                    H[i_1,j_1] = H[i_1,j_1] * a[i_1,j_1] / b[i_1,j_1]
        
        
        c = sensorDataMat * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(K):
                if d[i_2, j_2] != 0:
                    W[i_2,j_2] = W[i_2,j_2] * c[i_2,j_2] / d[i_2, j_2]

        sensorDataMat_pre = W * H
        E = sensorDataMat - sensorDataMat_pre
        #print E
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i,j] * E[i,j]
        print(err)

        if err < e:
            break
        
    return W, H

def NMF_useful(sensorDataMat, K, steps=10, alpha=0.005):
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
                        if np.dot(P[i,:],Q[:,j]) > 0:
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k])
                            if P[i][k] < 0:
                                P[i][k] = 0
                            if Q[k][j] < 0:
                                Q[k][j] = 0
    
        e = 0
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    e = e + pow(sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
        print(e)
    return P, Q


def NMF_test(sensorDataMat, K, steps=50):
    m, n = sensorDataMat.shape
    P = np.random.rand(m, K)
    Q = np.random.rand(n, K)
    
    Q = Q.T
    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
#                    eij = sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        if np.dot(P[i,:],Q[:,j]) > 0:
                            P[i][k] = P[i][k]*sensorDataMat[i, j]/((np.dot(P[i,:],Q[:,j])+1) * (Q[k][j]+1))
                            Q[k][j] = Q[k][j]*sensorDataMat[i, j]/((np.dot(P[i,:],Q[:,j])+1) * (P[i][k]+1))
                        
#    for step in range(steps):
#        for i in range(m):
#            for j in range(n):
#                if sensorDataMat[i, j] > 0:
#                    eij = sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j])
#                    for k in range(K):
##                        alpha1 = P[i][k]*sensorDataMat[i, j]/(np.dot(P[i,:],Q[:,j]) * Q[k][j]+1)
##                        print(alpha1)
##                        alpha2 = Q[k][j]*sensorDataMat[i, j]/(np.dot(P[i,:],Q[:,j]) * P[j][k]+1)
#                        P[i][k] = P[i][k] + alpha1 * (2 * eij * Q[k][j])
#                        Q[k][j] = Q[k][j] + alpha2 * (2 * eij * P[i][k])
    
        e = 0
        for i in range(m):
            for j in range(n):
                if sensorDataMat[i, j] > 0:
                    e = e + pow(sensorDataMat[i, j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + pow(P[i][k],2) + pow(Q[k][j],2)
        print(e)
        if e < 0.001:
            break
    return P, Q


if __name__ == "__main__":
    contentDic = preprocess('test-data.txt')
    changed_sensorDataMat = random_create_data(contentDic, '19')
    P, Q = NMF_test(changed_sensorDataMat, 2)
