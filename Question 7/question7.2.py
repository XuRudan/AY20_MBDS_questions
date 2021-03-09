import pandas as pd

# Index = Coordinate[0] + Coordinate[1]*L1 + Coordinate[2]*L1*L2 +...+ Coordinate[n]*L1*L2*...*Ln-1
def coordinates_to_index(L, infilepath, outfilepath):
    coordinates = pd.read_csv(infilepath, sep = '\t', header = 'infer').values.tolist()
    Index = np.zeros(len(coordinates), dtype = int)
    for i, C in enumerate(coordinates):
        index = int(0)
        for ii, c in enumerate(C):
            b = 1
            for j in range(ii):
                b *= L[j]
            index += c*b
        Index[i] = index
    output = pd.DataFrame({'index':Index})
    output.to_csv(path_or_buf = outfilepath, index=False)

# Coordinate[0] = Index%L1; Coordinate[n] = int(Index/(L1*L2*...*Ln))
def index_to_coordinates(L, infilepath, outfilepath):
    Index = pd.read_csv(infilepath, sep = '\t', header = 'infer').values.tolist()
    Coordinates = np.zeros((len(Index),len(L)), dtype = int)
    for i, index in enumerate(Index):
        for ii in range(len(L)-1,0,-1):
            b = 1
            for j in range(ii):
                b *= L[j]
            Coordinates[i,ii] = int(index[0] / b)
            index -= Coordinates[i,ii]*b
        Coordinates[i,0] = index[0]
    output = pd.DataFrame(Coordinates, columns = ['x' + str(n+1) for n in range(len(L))])
    output.to_csv(path_or_buf = outfilepath, sep = '\t', index=False)

infilepath_1 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.2/input_coordinates_7_2.txt'
infilepath_2 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.2/input_index_7_2.txt'
outfilepath_1 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.2/output_index_7_2.txt'
outfilepath_2 = 'F:/AY20_MBDS_questions-answer/Question 7/Question 7.2/output_coordinates_7_2.txt'

coordinates_to_index([4,8,5,9,6,7], infilepath_1, outfilepath_1)
index_to_coordinates([4,8,5,9,6,7], infilepath_2, outfilepath_2)


