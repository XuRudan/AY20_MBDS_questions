# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:54:11 2021

@author: ASUS
"""

def Add_(row): # compute 
    num = 0
    for i in range(row):
        num += i+1
    return num

def Decompose(num, row, col):
    num = num - (col-1)
    sublist = [1]*(col-1)
    i = 0
    while(num):
        if num > (row - 1):
            sublist[i] = row
            num -= (row - 1)
        else:
            sublist[i] += num
            num = 0
        i += 1
    return sublist

def Operat(N, row, col, kind='a', filepath='F:/AY20_MBDS_questions-answer/Question 1/output_question_1'):
    if N < (Add_(row) + (col-1)*row):
        sublist = Decompose(N-Add_(row), row, col)
        numlist = [0]*row
        for subnum in sublist:
            numlist[subnum-1] += 1
        operat = ''
        for index, n in enumerate(numlist):
            operat += 'R'*n
            if index < row - 1:
                operat += 'D'
    else:
        operat = 'This figure is out of the maximum range'
    with open(filepath, kind) as f:
        f.write(str(N) + ' ' + operat + '\n')
    return operat

Operat(65, 9, 9, 'w+')
Operat(72, 9, 9)
Operat(90, 9, 9)
Operat(110, 9, 9)
with open('F:/AY20_MBDS_questions-answer/Question 1/output_question_1', 'a') as f:
    f.write('\n')
Operat(87127231192, 90000, 100000)
Operat(5994891682, 90000, 100000)






        
