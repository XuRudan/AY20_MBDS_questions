# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:29:27 2021

@author: ASUS
"""
import numpy as np

"""
read me:
    Grid sort by ordinal number:      Beads sort by their number:
        1  14 2  15 3                     B: 13  (first)
        16 4  17 5  18                    R: 12  (second)
        6  19 7  20 8
        21 9  22 10 23
        11 24 12 25 13
    1.When the maximum number of beads N_max <= ceil(L*L/2), penalty = 0
    2.Another situation, penalty = (N_max - ceil(L*L/2))*3
"""

class Grid():
    def __init__(self, L, nums): # L(int) nums(dict)
        self.L = L
        self.nums = nums
        self.kinds = len(nums)
        self.kind = list(nums.keys())
        self.grid = -np.ones((L,L))

    def Penalty(self):
        if self.grid[0,0] == -1:
            return float("inf")
        penalty = 0
        for row in range(self.L):
            for col in range(self.L):
                if row > 0:
                    penalty += self.grid[row-1,col] == self.grid[row,col]
                if row < self.L-1:
                    penalty += self.grid[row+1,col] == self.grid[row,col]
                if col > 0:
                    penalty += self.grid[row,col-1] == self.grid[row,col]
                if col < self.L-1:
                    penalty += self.grid[row,col+1] == self.grid[row,col]
        return penalty/2
    
    def Sort(self):
        S = [1]*self.L*self.L
        I = 0
        if self.L%2 == 1:
            sortlist = list(range(0,self.L*self.L,2)) + list(range(1,self.L*self.L,2))
        else:
            A = list(range(0,self.L,2)) + list(range(self.L+1,self.L*2,2))
            sortlist = list(A)
            for i in range(int(self.L/2-1)):
                B = [a + self.L*2*(i+1) for a in A]
                sortlist += B
            C = list(range(self.L*self.L))
            for n in sortlist:
                C.remove(n)
            sortlist += C
        lis = list(self.nums.values())
        value_sortlist = sorted(range(len(lis)), key=lambda k: lis[k], reverse=True)
        for k in value_sortlist:
            for j in range(self.nums.get(self.kind[k])):
                S[sortlist[I]] = k
                I += 1
        S = np.array(S)
        self.grid = S.reshape((self.L,self.L))
        
    def output_write(self,filepath): # Write the results to the file
        rows, cols = self.grid.shape
        with open(filepath, 'w+') as f:
            for row in range(0,rows):
                for col in range(0,cols):
                    f.write(self.kind[self.grid[row,col]])
                    if col == cols-1:
                        f.write('\n')
                    else:
                        f.write(' ')

A = Grid(5,{'R':12,'B':13})
A.Sort()
print(A.Penalty())
A.output_write('F:/AY20_MBDS_questions-answer/Question 5/output_question_5_1')

B = Grid(64,{'R':139,'B':1451,'G':977,'W':1072,'Y':457})
B.Sort()
print(B.Penalty())
B.output_write('F:/AY20_MBDS_questions-answer/Question 5/output_question_5_2')
