# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:48:02 2021

@author: ASUS
"""
import numpy as np
import pandas as pd

def coordinates_to_index(L1, L2, infilepath, outfilepath): # Index = Coordinate[0] + Coordinate[1]*L1
    coordinates = pd.read_csv(infilepath, sep = '\t', header = 'infer').values.tolist()
    Index = np.zeros(len(coordinates), dtype = int)
    for i, (x1,x2) in enumerate(coordinates):
        Index[i] = x1 + x2*L1
    output = pd.DataFrame({'index':Index})
    output.to_csv(path_or_buf = outfilepath, index=False)

def index_to_coordinates(L1, L2, infilepath, outfilepath): # Coordinate[0] = Index%L1; Coordinate[1] = int(Index/L1)
    Index = pd.read_csv(infilepath, sep = '\t', header = 'infer').values.tolist()
    Coordinates = np.zeros((len(Index),2), dtype = int)
    for i, index in enumerate(Index):
        Coordinates[i,0] = index[0] % L1
        Coordinates[i,1] = int(index[0] / L1)
    output = pd.DataFrame({'x1':Coordinates[:,0], 'x2':Coordinates[:,1]})
    output.to_csv(path_or_buf = outfilepath, sep = '\t', index=False)

infilepath_1 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.1/input_coordinates_7_1.txt'
infilepath_2 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.1/input_index_7_1.txt'
outfilepath_1 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.1/output_index_7_1.txt'
outfilepath_2 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.1/output_coordinates_7_1.txt'

coordinates_to_index(50, 57, infilepath_1, outfilepath_1)
index_to_coordinates(50, 57, infilepath_2, outfilepath_2)
