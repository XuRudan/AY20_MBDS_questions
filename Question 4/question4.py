# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:08:48 2021

@author: ASUS
"""
import numpy as np

def input_read(filepath): # Read input data from file
    flag = 0
    col = 0
    datalist = []
    file=open(filepath, 'rb')
    while (1):
        context=file.read(1)
        if context:
            if context == b'\t' or context == b' ':
                continue
            elif context == b'\n':
                flag=0
                continue
            else:
                datalist.append(int(context))
                flag += 1
                col = max(col, flag)
        else:
            break
    file.close()
    data = np.array(datalist).reshape(-1,col)
    return data

def output_write(LabelImage, filepath): # Write the label results to the file
    rows, cols = LabelImage.shape
    with open(filepath, 'w+') as f:
        for row in range(0,rows):
            for col in range(0,cols):
                f.write(str(LabelImage[row,col]))
                if col == cols-1:
                    f.write('\n')
                else:
                    f.write(' ')
    
def regu(x, minx, maxx): # Prevent indexerror
    if x < minx:
        x = minx
    if x > maxx:
        x = maxx
    return x

def grow_from_seed(inarray, seed, label, neighbors=4): # Use seed points to find connected regions by growth algorithm
    rows, cols = inarray.shape
    labelarray = np.zeros(inarray.shape, dtype=int)
    labelarray[seed] = label # Initialize the label matrix
    inarray[seed] = 0
    buf = [(seed)] # The buffer used to manage search points
    save = 1 # Saveing pointer
    chec = 0 # Searching pointer
    while(save > chec):
        row, col = buf[chec]
        for i in range(regu(row-1,0,rows-1),regu(row+2,0,rows)):
            for j in range(regu(col-1,0,cols-1),regu(col+2,0,cols)):
                if neighbors == 4:
                    if (row-i)*(col-j) != 0: # Ignore the four corners of the 8-neighborhood
                        continue
                if inarray[i,j] == 1:
                    labelarray[i,j] = label # label
                    inarray[i,j] = 0 # Ignore the labeled parts
                    buf.append((i,j))
                    save += 1
        chec += 1
    return labelarray
    
def Connected(inputimage, neighbors=4): # Find positive points and takes each point as the seed point to label
    Rows, Cols = inputimage.shape
    label = 1 # The first label
    labelimage = np.zeros(inputimage.shape, dtype=int)
    for Row in range(0,Rows):
        for Col in range(0,Cols):
            if inputimage[Row,Col] == 1:
                labelarray = grow_from_seed(inputimage, (Row,Col), label, neighbors)
                labelimage += labelarray # Merge labels for each index
                label += 1
    return labelimage

#parameters
filepath = 'F:/AY20_MBDS_questions-answer/Question 4/input_question_4' 
outputpath = 'F:/AY20_MBDS_questions-answer/Question 4/output_question_4'

#working code
InputImage = input_read(filepath) # Read the input data
LabelImage = Connected(InputImage, neighbors=8) # To label
output_write(LabelImage, outputpath) # Write the output data













