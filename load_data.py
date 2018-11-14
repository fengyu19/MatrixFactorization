# -*- coding: utf-8 -*-
import numpy as np
import random

def preprocess(path):
#    a = np.loadtxt(path)
#    print(a)
    # key: sensor number, value: data
    contentDic = dict(list())
    
    with open(path) as f:
        all_data = f.readlines()
        for line in all_data:
            line = line.strip('\n').split(' ')
            if 'NaN' in line:
                continue
            else:
                if line[1] not in contentDic.keys():
                    contentDic[line[1]] = []
                    contentDic[line[1]].append([float(line[0]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
                else:
                    contentDic[line[1]].append([float(line[0]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
    
#    print(contentDic)
    return contentDic

def random_create_data(contentDic, sensorNum):
    sensorDataMat = np.mat(contentDic[sensorNum])
    m, n = sensorDataMat.shape
    for i in range(m):
        sensorDataMat[i, random.choice([1,2,3,4])] = 0
    for i in range(m):
        sensorDataMat[i, random.choice([1,2,3,4])] = 0
#    print(sensorDataMat)
    return sensorDataMat
           

if __name__ == "__main__":
#    preprocess('labapp3-data-new.txt')
    contentDic = preprocess('test-data.txt')
    changed_sensorDataMat = random_create_data(contentDic, '19')